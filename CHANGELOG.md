# Changelog

## v0.1.0 — 2026-11-08

First release. A terminal-native CLI coding agent whose inference runs on a
*swarm* of Ollama (and MLX) peers on the local network — source code never
leaves the LAN.

- **Workflows**: `fix` (pipe a crash log in, get a verified unified diff out;
  `--apply` writes it and runs the tests, with one automatic repair attempt),
  `analyze` (repo analysis fanned out across the swarm and merged into a
  structured report), `gen-tests` (pytest generation for a module),
  `ask`, `swarm`, `doctor`.
- **Local network orchestration**: configured peers, optional /24 subnet
  scan for Ollama on port 11434, latency-sorted peer selection, per-model
  peer preference, round-robin distribution with mid-run failover and
  reassignment.
- **Backends**: Ollama over HTTP (stdlib `urllib` only — zero runtime
  dependencies), plus an MLX peer client for `mlx_lm.server` on Apple
  Silicon; `doctor` reports hardware (chip, unified memory) and recommends
  model sizes.
- **Code-aware chunking**: Python files split by `ast` into top-level
  function/class units under a token budget; other files use overlapping
  line windows.
- **Prompts** tuned for Qwen2.5-Coder-class models with blunt output
  contracts (diff-only fixes, structured summaries, code-only tests).
- **Safe patching**: diffs are verified hunk-by-hunk against the current
  files before anything is written; a mismatch changes nothing.
- 50 automated tests covering chunking, patching, discovery, distribution
  and failover, config layering, the Ollama client, and an end-to-end
  `fix` run against a fake Ollama server.
