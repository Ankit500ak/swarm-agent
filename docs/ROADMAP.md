# Roadmap & Workstreams

swarm-agent is built in phases, by two workstreams with **strict file
ownership** — so two people (and one AI agent) can work in parallel
without merge conflicts. This file is the contract. If it's not in
your lane, don't edit it; open an issue instead.

## The lanes

| Lane | Who | Owns |
|---|---|---|
| **Core** | Ankit (via Jaat Bhai, his agent) | `swarm_agent/discovery.py`, `swarm_agent/distributor.py`, `swarm_agent/agent.py`, `swarm_agent/patcher.py`, `swarm_agent/chunking.py` + their tests |
| **Backends & Prompts** | Dhruv (`@heywinterbell`) | `swarm_agent/backends.py`, `swarm_agent/prompts.py`, `swarm_agent/runner.py`, `benchmarks/`, `examples/` + their tests |
| **Shared contracts** | Both — coordinate first | `swarm_agent/cli.py`, `swarm_agent/config.py`, `swarm_agent/ollama.py`, `README.md`, `pyproject.toml`, `docs/` |

Rules that keep us conflict-free:

1. **One owner per file.** The other lane reviews, never edits
   directly — except shared contracts, which need an issue + a
   heads-up *before* the first edit.
2. **Everything lands via PR.** Branch `feat/<lane>-<slug>` (e.g.
   `feat/backends-mlx-probe`), PR into `main`, the *other* lane
   reviews. No direct pushes to `main`.
3. **Tests travel with code.** A PR that changes behaviour without
   tests doesn't merge. The full suite must pass.
4. **Interfaces are contracts.** `OllamaClient`, `Config`, the
   `Chunk`/`Peer` dataclasses — changing a signature means updating
   the other lane in the same PR description, before merging.
5. **Phase gates.** A phase closes with a tagged release and a
   results note (numbers, not adjectives) in `CHANGELOG.md`.

## Phases

### Phase 1 — Core engine ✅ (v0.1.0, 2026-11-08)

CLI (`fix` / `analyze` / `gen-tests` / `swarm` / `ask` / `doctor`),
Ollama client + MLX client, /24 swarm discovery, round-robin
distribution with failover, AST-aware chunking, verified patching,
50 tests. Verified against stub servers; live-hardware validation is
Phase 2's job.

### Phase 2 — Real hardware, real models (current)

Prove the swarm on real machines and real models, and harden the
backends.

- **Backends & Prompts lane:** Apple Silicon/MLX validation on real
  M-series hardware; prompt library v2 (JS/TS, Go, Rust contracts);
  test-command auto-detection; benchmark harness (1 vs 2 vs 3
  peers); demo kit (seeded-bug example + terminal recording).
- **Core lane:** latency-weighted scheduling (beyond round-robin);
  mDNS/Bonjour discovery alongside the subnet scan; live two-machine
  `analyze` run with published timings.

Exit criteria: `doctor` correct on a real Mac; a two-machine swarm
run benchmarked and written up; v0.2.0 tagged.

### Phase 3 — Repository intelligence

Persistent repo index (embeddings over chunks, reused across runs),
cross-file context retrieval for `fix`, multi-file patches, repair
loop v2 (test failure → re-diagnosis with history), session memory.

### Phase 4 — Distribution & polish

PyPI release, Homebrew formula, streaming output, docs site,
recorded end-to-end demo, v1.0.

*Phases 3–4 get their own issue sets when Phase 2 closes.*
