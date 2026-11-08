<p align="center">
  <img src="docs/assets/banner.svg" alt="swarm-agent — your code never leaves your network" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/version-0.1.0-4ade80" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license">
  <img src="https://img.shields.io/badge/backend-Ollama-000000" alt="Ollama backend">
  <img src="https://img.shields.io/badge/Apple%20Silicon-MLX-999999" alt="Apple Silicon / MLX">
  <img src="https://img.shields.io/badge/tests-50%20passing-2e7d32" alt="50 tests passing">
  <img src="https://img.shields.io/badge/dependencies-0%20runtime-b08d3e" alt="Zero runtime dependencies">
</p>

**A terminal-native coding agent that runs on machines you own.** Ask it to
fix a crash, analyse a repository, or draft a test suite — and instead of
sending your code to someone else's cloud, it splits the work across the
Ollama instances already running on your local network: your laptop, the
workstation under the desk, the Mac mini in the corner. One CLI, many
machines, zero uploads.

## Why

AI coding assistants are genuinely useful, and a lot of developers still
can't use them: the code is proprietary, the client contract says no, the
repo *is* the company. Every cloud assistant wants your source on their
servers.

Meanwhile, most teams already own plenty of compute. It just sits idle in
individual machines, one Ollama at a time — and a single laptop chokes on
the jobs that matter most: analysing a massive codebase chunk by chunk, or
generating tests for a whole tree.

swarm-agent's answer is to treat your LAN as the computer:

- **Privacy by construction.** Inference happens on peers you run. There is
  no cloud endpoint in the design — no account, no API key, no telemetry.
- **The swarm does the heavy lifting.** Repository analysis is fanned out
  chunk-by-chunk across every peer, round-robin, with automatic failover
  when a machine drops off.
- **Hardware you already paid for.** Ollama anywhere; MLX on Apple Silicon,
  where unified memory lets big coder models chew through big context.

## How it works

```mermaid
flowchart LR
    U["cat error.log | swarm-agent fix"] --> CLI["swarm-agent CLI"]
    CLI --> D{"distributor"}
    D -->|"synthesis: fastest peer"| P1["Ollama · laptop<br/>qwen2.5-coder:7b"]
    D -->|"fan-out: round-robin"| P2["Ollama · workstation<br/>qwen2.5-coder:14b"]
    D -->|"fan-out: round-robin"| P3["MLX · Mac mini<br/>mlx_lm.server"]
    P1 --> CLI
    P2 --> CLI
    P3 --> CLI
    CLI --> V["verified patch + test run"]
```

### How a fix flows

1. You pipe a crash log in. swarm-agent extracts the file paths and line
   numbers from the traceback and pulls those files as context.
2. The fastest peer gets one synthesis prompt: *error + code → a unified
   diff, nothing else* (the prompts are written bluntly, the way
   Qwen2.5-Coder-class models like them).
3. The diff comes back to **your** machine, where the patcher verifies
   every hunk's context against the current files *before writing
   anything*. A stale diff is an error message, never a half-patched tree.
4. With `--apply`, the patch lands and your test suite runs. If it fails,
   the swarm gets exactly one repair attempt with the test output before
   the failure is reported to you.

Analysis inverts the shape: hundreds of small per-chunk summaries are the
fan-out, and a single synthesis call on the fastest peer merges them into
the report.

## Quickstart

```bash
# 1. Install (stdlib-only — nothing else to fetch)
git clone <this repo> && cd swarm-agent
pip install -e .

# 2. Start a model server on this machine (or any machine on your LAN)
ollama pull qwen2.5-coder:7b
ollama serve

# 3. Check the machine and the swarm
swarm-agent doctor
swarm-agent swarm --scan        # find every Ollama on your /24

# 4. Use it
cat error.log | swarm-agent fix
```

## Usage

### `fix` — from crash log to verified patch

```console
$ cat error.log | swarm-agent fix --apply
swarm-agent: context from 1 file(s): calc.py
swarm-agent: applied changes to calc.py
swarm-agent: tests pass (python -m pytest -q)
```

Without `--apply` it's a dry run: you get the proposed diff, plus a check
that it would apply cleanly. The classic zsh workflow just works:

```bash
python app.py 2>&1 | swarm-agent fix            # propose
python app.py 2>&1 | swarm-agent fix --apply    # apply + run tests
swarm-agent fix --file crash.txt --path ~/code/myrepo
```

### `analyze` — a whole repo, read by the whole swarm

```bash
swarm-agent analyze ~/code/bigrepo
swarm-agent analyze . --question "where is retry logic implemented, and is it safe?"
```

Files are walked (skipping `.git`, venvs, `node_modules`, binaries and
anything over 1 MB), chunked code-aware, summarised in parallel across
your peers, and merged into a structured report: purpose, architecture,
key modules, risks, suggested next steps.

### `gen-tests` — draft a pytest suite for a module

```bash
swarm-agent gen-tests src/pricing.py            # print the suite
swarm-agent gen-tests src/pricing.py --write --run
# wrote tests/test_pricing_generated.py, and ran it
```

### `ask`, `swarm`, `doctor`

```bash
swarm-agent ask "why would a dict lose insertion order in a pickle round-trip?"
swarm-agent swarm            # peers, models, latency
swarm-agent doctor           # hardware report + backend recommendations
```

Exit codes are script-friendly: `0` success, `1` the swarm or workflow
failed, `2` bad usage.

## Add a machine to your swarm

Any computer on your network can become a peer in about a minute:

```bash
# on the new machine
ollama pull qwen2.5-coder:7b
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

```bash
# on yours — pick it up in the scan, or name it explicitly
swarm-agent swarm --scan
export SWARM_PEERS="http://localhost:11434,http://192.168.1.40:11434"
```

> **Firewall note:** binding Ollama to `0.0.0.0` serves *anyone on the
> network* who can reach port 11434, with no auth. Do this on networks
> you trust, keep your host firewall on, and don't port-forward it. Your
> code stays on the LAN — make sure the LAN deserves it.

Persistent setup lives in `~/.config/swarm-agent/config.json`:

```json
{
  "model": "qwen2.5-coder:7b",
  "peers": ["http://localhost:11434", "http://192.168.1.40:11434"],
  "test_command": "python -m pytest -q"
}
```

## Apple Silicon: MLX and unified memory

A Mac is often the best seat in the swarm. Ollama already uses Metal, and
for serving a dedicated coder model, `mlx_lm.server` is frequently faster
still — swarm-agent speaks to it as a first-class **MLX peer**
(OpenAI-compatible API, auto-detected on `localhost:8080`, or set
`mlx_url` / `SWARM_MLX_URL`).

Unified memory is the reason this works so well: the model lives in one
pool that CPU and GPU share, so size is mostly a question of gigabytes:

| Unified memory | Comfortable coder model | Notes |
|---|---|---|
| 8 GB | `qwen2.5-coder:7b` (small context) | lean on bigger peers for analysis |
| 16 GB | `qwen2.5-coder:7b`, 14B quantised | the laptop sweet spot |
| 32 GB | `qwen2.5-coder:14b` | 32B quantised also fits |
| 64 GB+ | `qwen2.5-coder:32b` | run the big synthesis calls here |

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --port 8080
swarm-agent doctor     # reports chip + unified memory, and sees the MLX peer
```

## Prompts & chunking

The defaults are tuned for **Qwen2.5-Coder-class open-weight models**, which
follow blunt format contracts far better than polite suggestions:

- **`fix`** — the model must reply with *only* a unified diff in a
  ` ```diff ` fence, context lines character-exact, or the single line
  `NO_FIX_POSSIBLE`. Parsing is therefore trivial and strict.
- **`analyze`** — each chunk gets a three-line contract (`PURPOSE:` /
  `KEY:` / `RISKS:`); the merge prompt assembles a fixed report structure.
- **`gen-tests`** — *only* Python in a ` ```python ` fence, real assertions,
  no mocks-of-mocks.

Chunking is code-aware: Python files are split with `ast` into top-level
function and class units (never arbitrary character slices), everything
else uses overlapping line windows, and every chunk respects the token
budget (`chunk_tokens`, default 1500, estimated at ~4 chars/token).
Oversized units split at line boundaries.

## Configuration reference

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `model` | `SWARM_MODEL` | `qwen2.5-coder:7b` | model requested from peers |
| `peers` | `SWARM_PEERS` | `http://localhost:11434` | comma-separated peer base URLs |
| — | `OLLAMA_HOST` | — | primary peer (Ollama convention), prepended to `peers` |
| `mlx_url` | `SWARM_MLX_URL` | auto-detect `:8080` | MLX server base URL |
| `chunk_tokens` | — | `1500` | per-chunk token budget |
| `test_command` | — | `python -m pytest -q` | run after `fix --apply` (set `null` to skip) |
| `request_timeout` | — | `120` s | per-call model timeout |
| `auto_discover` | `--scan` on `swarm` | `false` | scan the local /24 for Ollama |

Precedence: built-in defaults → `~/.config/swarm-agent/config.json` →
environment → CLI flags (`--model`, `--peers`).

## Roadmap

**v0.1.0 (this release)** — the three workflows, Ollama + MLX peers,
subnet discovery, round-robin distribution with failover, verified
patching, the test-run loop. Tested on Linux; the suite runs entirely
against fakes and a stub server, so it passes with no model installed.

**Planned** — exercise and tune the macOS/MLX path on real Apple hardware;
streaming output for synthesis calls; peer capability weighting (send the
32B peer the hard chunks); `git worktree` sandbox mode for `fix --apply`;
more languages for AST chunking (tree-sitter); a `watch` mode for CI logs.

## Contributing

Issues and PRs welcome. The codebase is deliberately small and stdlib-only
— please keep runtime dependencies at zero. Run the tests with:

```bash
python -m pytest
```

Design notes, the distribution model, the failover policy, and the
security posture live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT — see [LICENSE](LICENSE). © 2026 Ankit Pal.
