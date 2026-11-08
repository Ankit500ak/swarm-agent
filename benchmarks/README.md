# Benchmarks — fix benchmark

This directory holds the fix benchmark (issue #11): five tiny repos,
each with one seeded bug, fixed by `swarm-agent fix --apply` against
a local model. (The analyse benchmark — 1 vs 2 vs 3 peers — lives
here too once its own PR lands; its methodology will be documented
in this README alongside this section.)

## The five repos

| Repo | Bug | Tests |
|---|---|---|
| `examples/cross_file_crash/` | Two-file: prices stored as strings in `catalog.py`, crash in `checkout.py`. | 2, both fail |
| `examples/signature_change/` | Two-file: signature changed in `store.py`, caller in `report.py` not updated. | 2, both fail |
| `fix_repos/off_by_one/` | One-line slice overshoot: `items[-n - 1:]` returns `n+1` items. | 2 of 3 fail |
| `fix_repos/empty_average/` | Missing empty-list guard: `ZeroDivisionError` on `average([])`. | 1 of 2 fails |
| `fix_repos/threshold/` | Boundary bug: `is_adult(18)` returns `False` (`>` instead of `>=`). | 1 of 3 fails |

The first two are the existing multi-file examples (issue #12),
reused rather than duplicated. The last three live in `fix_repos/`
and follow the same pattern: source, tests, `pytest.ini` +
`conftest.py`, a real captured `crash.log`, and a README stating the
symptom, not the answer. Each repo's fix was proven in a scratch
copy before it was added: the intended one-line (or few-line) fix
makes its tests pass.

## How to run

From the repo root, with an Ollama peer up and serving the model:

```bash
python benchmarks/fixbench.py --peers http://localhost:11434
```

- `--model` selects the model (default `qwen2.5-coder:7b`).
- `--peers` takes comma-separated Ollama peer URLs; it can also
  come from `SWARM_PEERS`. With neither, the default peer
  `http://localhost:11434` is used.
- `--repos` runs a subset, e.g. `--repos broken-demo,off_by_one`.
- `--work-dir` sets the parent directory for the scratch copies
  (default: the system temp dir).

For each repo the runner copies it to a scratch directory (the
committed repo is never mutated), runs its tests to capture the
failure, pipes the failure to `swarm-agent fix --apply`, and then
independently re-runs the tests. Results are appended to
`FIX_RESULTS.md`.

## Reading the results

| column | meaning |
|---|---|
| date | when the run happened (local time) |
| repo | which seeded-bug repo |
| bug | one-line description of the seeded bug |
| outcome | `fixed attempt 1`, `fixed after repair`, or `failed` |
| attempts | how many fix attempts the run took (1 + repair rounds) |
| model | model requested from the peers |
| machine | chip + memory of the machine that ran the benchmark |
| notes | why a run failed, if it did |

## Methodology notes

- **The outcome is grounded in the independent re-run**, not in
  what the fix run claimed. If the fix exits 0 while the tests
  still fail, the row says `failed` and the note says so.
- **Attempt 1 vs after repair:** the distinction comes from fix's
  own output — `tests pass after repair - fixed on attempt N` — and
  the repair loop's `repair R/M` lines. A run that never entered
  the repair loop and passed is `fixed attempt 1`.
- **Machines, models, network:** the table records the machine and
  the model for every row. The model runs locally, so the network
  matters little here — but record the machine (a MacBook's MLX
  peer behaves differently from a Linux box's Ollama peer).
- **Reproduce a row:** same repo, same model, peers already warm
  (`ollama pull` the model first), then run the command above.
  Model output is non-deterministic, so a single row is one sample,
  not a law — re-run before concluding a repo is unfixable.
- **No runs have been recorded yet** — the table is empty until the
  first real run on hardware with a local model.
