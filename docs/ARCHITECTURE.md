# Architecture

How swarm-agent is put together, and why.

## Shape of the system

```
            ┌──────────── your terminal ────────────┐
            │  swarm-agent CLI                      │
            │  fix · analyze · gen-tests · ask      │
            └──────────────┬────────────────────────┘
                           │ in-process
            ┌──────────────▼────────────────────────┐
            │  workflows (agent.py)                 │
            │  chunking · prompts · patcher · tests │
            └──────────────┬────────────────────────┘
                           │ work units + synthesis calls
            ┌──────────────▼────────────────────────┐
            │  distributor                          │
            │  peer selection · round-robin ·       │
            │  failover · per-peer accounting       │
            └──────┬──────────────┬─────────────┬───┘
                   │ HTTP (LAN)   │             │
          ┌────────▼───────┐ ┌────▼─────────┐ ┌─▼──────────────┐
          │ Ollama peer    │ │ Ollama peer  │ │ MLX peer       │
          │ (this laptop)  │ │ (workstation)│ │ mlx_lm.server  │
          └────────────────┘ └──────────────┘ └────────────────┘
```

Everything left of the HTTP boundary is one Python process with **zero
runtime dependencies** (stdlib only). Everything right of it is a model
server the user already runs. There is no third zone: no cloud endpoint,
no telemetry, no account.

## Distribution model

Two call patterns cover every workflow:

1. **Synthesis calls** — one prompt, one answer, quality matters most
   (writing the fix diff, merging the analysis, drafting the tests).
   These go to the *fastest* peer that serves the model.
2. **Fan-out calls** — many independent units (per-chunk summaries in
   `analyze`), latency-tolerant, embarrassingly parallel. These are what
   the swarm is for: a 200-chunk repository analysis takes roughly
   `200 / peers` sequential calls per machine instead of 200.

`Distributor.map_units(units, fn)` implements the fan-out:

- Units are assigned **round-robin** across the active peers; each peer
  gets exactly **one worker thread**, so a peer is never asked to run two
  generations at once (Ollama serialises them anyway — queueing locally
  would only add latency and hide load).
- Peers that serve the requested model are preferred. If none do, all
  healthy peers are used and each peer's client is bound to a model that
  peer actually has, so work is never sent into a 404.
- Results are reassembled **in input order** regardless of completion
  order, and `Distributor.counts` records how many units each peer
  completed (surfaced in debugging and tests).

## Failover

A unit that raises marks its peer **down** for the rest of the run. The
failed unit *and everything still queued behind it on that peer* are
reassigned, round-robin again, across the survivors. Only the survivors'
threads keep working; a peer that dies mid-analysis costs a few retried
chunks, not the run.

A unit that fails on *every* peer stops circulating: its result slot holds
the exception, and the workflow reports it (the analysis merge includes a
“failed to analyse” entry for that chunk) instead of crashing or silently
dropping code from the report.

Discovery is deliberately separate from distribution: `discover()` returns
latency-sorted peers (configured peers first-class; optional /24 scan of
port 11434; an MLX peer when configured or listening on localhost:8080),
and the distributor trusts but re-verifies — a peer that passed discovery
can still die mid-run, which is exactly the case failover handles.

## Chunking rationale

Models answer better about *whole units of code* than about arbitrary
1024-character slices, and Qwen2.5-Coder-class models are no exception.
Python therefore goes through `ast`: top-level functions and classes are
the units, with their real source segments; module preamble and code
between units become `module` chunks so nothing is silently skipped. A
unit larger than the token budget is split at line boundaries (never
mid-line, except a single pathological line, which is hard-split by
characters). Non-Python files use line windows with a 10-line overlap so a
construct straddling a seam still appears whole in one chunk.

The budget is a *token estimate* (`ceil(chars / 4)`), not a tokenizer —
close enough for sizing prompts, with no dependency cost.

## The patcher is the safety boundary

Models propose; the patcher disposes. `apply_unified_diff` works in two
phases:

1. **Verify** — parse the diff, then for every file, apply every hunk *in
   memory*, checking that each hunk's context and removed lines match the
   current file exactly (at the stated position, or at a unique position
   if line numbers drifted slightly). Paths that escape the repo root are
   rejected. Any mismatch raises `PatchError` and **nothing is written**.
2. **Commit** — write each file atomically (temp file + `os.replace`).

So the worst case for a bad model diff is an error message, never a
half-patched tree. `fix --apply` then runs the project's test command;
a failing suite triggers exactly one repair round-trip (test output +
current file state back to the swarm) before the failure is reported with
the test tail.

## Security posture

- **LAN-only by construction.** The only network calls are to peer base
  URLs the user configured or that answer on the local subnet scan. There
  is no update check, no telemetry, no analytics, nothing to phone home.
- **Prompts contain code — that's the point, and the boundary.** Chunk
  text goes to *your* machines. The README's peer guide covers the flip
  side: an Ollama bound to `0.0.0.0` serves anyone on the network, so
  treat swarm peers like any other LAN service (trusted network, host
  firewall).
- **Writes are conservative.** Dry-run is the default for `fix`; applying
  requires `--apply`, verification, and a test run.
- **Secrets**: swarm-agent has no credentials to leak — no API keys exist
  in this design. Config and env vars name machines and models only.

## Backends and hardware

- **Ollama peers** — the portable path, any OS, CPU or GPU.
- **MLX peers** — `mlx_lm.server` speaks an OpenAI-compatible API, so the
  MLX client is a thin second implementation of the same `chat()` surface.
  On Apple Silicon, unified memory means a 32B coder model can sit in the
  same pool the GPU reads from; `doctor` reads the chip and
  `hw.memsize` via `sysctl` and recommends model sizes by memory tier.

Honest status: v0.1.0 is tested on Linux (the full suite runs without any
real peer, against fakes and a stub HTTP server). The macOS/MLX code paths
are implemented and guarded (`sysctl` failures degrade to plain values),
but they have not yet been exercised on real Apple hardware — see the
roadmap in the README.
