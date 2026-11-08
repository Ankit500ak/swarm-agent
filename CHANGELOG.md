# Changelog

## Unreleased — Phase 2 + Phase 3 (merged on `main`, not yet tagged)

All of the Phase 2 and Phase 3 code below is merged on `main`. No new
version is tagged: per the roadmap's phase gates, v0.2.0 / v0.3.0 wait
on live-hardware results (real-Mac MLX validation, peer benchmarks),
which do not exist yet — so no benchmark or hardware numbers are
claimed here either. Test suite on `main`: 163 tests.

### Engine (core lane)

- **Persistent repo index** (#20, issue #13): chunk embeddings are
  computed once and stored in `<repo>/.swarm/index.json`, reused until
  a file's mtime/size changes; only changed files are re-embedded.
  Embeddings via an Ollama peer's `/api/embed` (new `OllamaClient.embed`),
  with a local TF-IDF fallback when no peer answers.
- **Cross-file retrieval in `fix`** (#22, issue #14): the index ranks
  chunks from files the traceback never named; the best join the fix
  prompt as related context, budget-capped so traceback files are
  never trimmed to make room.
- **Multi-file atomic patches** (#18, issue #15): `apply_multi_file_diff`
  verifies every hunk of every file in memory before writing anything —
  all-or-nothing across files, with per-file create/modify/delete
  results.
- **Repair loop v2** (#25, issue #16): failed verifications are parsed
  into structured failure records and re-diagnosed (original error +
  previous diff + the failures), applied against current file state,
  and bounded by `SWARM_MAX_REPAIRS` (default 2; `0` disables). Ends
  with a printed attempt history.
- **Latency-weighted scheduling** (#21, issue #6): fan-out units go to
  the peer with the lowest predicted wait (`in-flight × EWMA` of real
  unit durations, seeded from discovery latency), replacing plain
  round-robin in `Distributor.map_units`. Public API unchanged.
- **mDNS discovery** (#24, issue #7): a pure-stdlib DNS-SD browser
  joins the subnet scan under `auto_discover`. Honest yield: stock
  Ollama does not advertise over mDNS, so this mainly finds wrapped /
  third-party `_ollama._tcp` services.

### Prompts & backends (Dhruv's lane)

- **Prompt contracts v3** (#19, issue #9): multi-file diff format,
  repair re-diagnosis (consumes the structured failure records), and
  index-summary contracts — purely additive; the v1 contracts and
  default builder outputs stay byte-for-byte stable.
- **Prompt library v2** (#27, issue #2): fix + test-generation
  contracts for JavaScript, TypeScript, Go and Rust with per-language
  guardrails (no `any`/`@ts-ignore` escapes in TS, no `unwrap()` /
  `unsafe` escapes in Rust, Go errors returned not swallowed), plus
  extension-based selectors (`language_for_path`, `fix_system_for`,
  `testgen_system_for`).
- **Structured test-failure parsing** (#17, issue #10): `runner`
  turns pytest and jest output into frozen `{file, test, message}`
  records for the repair loop; unparseable failure output degrades to
  a single raw-tail record, never an exception.
- **Backends hardening** (#8, issue #1 — code half): MLX/hardware
  reporting hardened (empty `sysctl` replies, memory parsing). The
  issue's real-Mac validation step is still open.

### Runner

- **Test-command auto-detection** (#29, issue #3): `detect_test_command`
  picks `npm test` (package.json test script), `go test ./...`
  (go.mod), `cargo test` (Cargo.toml) or pytest from repo markers,
  in that priority order; an explicit command always wins and
  detection never raises.

### Wiring (core follow-ups)

- **Language selectors wired into the flows** (#30, issue #28): `fix`
  selects its contract from the deepest traceback file's extension and
  threads it through repairs; `gen-tests` selects by target extension
  (JS/TS write `tests/<name>.test.js` / `.test.ts`). Python behaviour
  is byte-identical to before.
- **Detection wired into the verify steps** (#32, issue #31): `fix`
  and `gen-tests --run` resolve their verify command as
  explicit `--test-cmd` > customised `config.test_command` >
  auto-detection > the pytest default — so non-Python repos are no
  longer verified with pytest.

### Examples

- **Two multi-file example projects** (#23, issue #12):
  `examples/cross_file_crash` (defect in a different file from the
  crash site) and `examples/signature_change` (fix spans a signature
  and its caller), each failing exactly as documented (2 tests each) —
  demo targets for cross-file retrieval and multi-file patching, and
  the seeded-bug repos for the future fix-benchmark.

### Fixes

- **Restored three files stored as literal base64 text** (#33):
  `swarm_agent/agent.py`, `tests/test_language_wiring.py` and
  `tests/test_detect_wiring.py` had landed on `main` as base64 blobs
  via a faulty push transport in #30/#32; restored byte-identical to
  the verified contents and the full suite re-run against `main`
  itself (163 passed). Pushes from the core side now get a read-back
  content check before their PR is created.

### Docs

- Phased roadmap and lane contract (`docs/ROADMAP.md`) and `CODEOWNERS`
  added at Phase 2/3 setup; this changelog and the README brought
  current with the merged Phase 2/3 state.

### Still pending (validation, not code)

- Real-Mac `doctor` / MLX validation (issue #1), the 1-vs-2-vs-3-peer
  benchmark harness (#4), the seeded-bug fix-benchmark (#11), and the
  demo kit (#5) — and the v0.2.0 / v0.3.0 tags that wait on them.

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
