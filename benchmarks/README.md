# Benchmarks — is the swarm actually faster?

The project's core claim is that splitting analysis across the
machines on your LAN beats running it on one. This directory
measures that claim, with honest numbers only: the harness records
what actually ran, and a failed run is recorded as failed, never
invented or dropped.

## Workload

The workload is fixed and pinned: `sample_repo/` is a small Python
project (4 files — `calc.py`, `store.py`, `cli.py`, `test_calc.py`)
that analyses into **16 chunks** at the default 1500-token budget.
It never changes between runs, so 1-peer and 3-peer numbers are
comparable. The chunk count is pinned by a test
(`tests/test_benchmarks.py`).

The workload is the real `analyze` flow
(`swarm_agent.agent.analyze_flow`): chunk summaries fanned out
across the swarm, then one synthesis call on the fastest peer. Wall
time covers the whole call, including synthesis.

## How to run

From the repo root, with Ollama peers up and serving the model:

```bash
python benchmarks/bench.py --peers http://localhost:11434,http://192.168.1.5:11434,http://192.168.1.6:11434
```

- `--peers` takes comma-separated Ollama peer URLs. It can also come
  from the `SWARM_PEERS` environment variable; with neither, the
  harness uses the default peer `http://localhost:11434`.
- `--model` selects the model (default `qwen2.5-coder:7b`).
- `--max-peers` caps the comparison (default 3).

The harness runs the workload with the first 1 peer, then the first
2, then the first 3, and appends one row per run to `RESULTS.md`.

## Reading the results

| column | meaning |
|---|---|
| date | when the run happened (local time) |
| machine | chip + memory of the machine that ran the benchmark |
| model | model requested from the peers |
| ollama peers | how many Ollama peers were configured for the run |
| wall (s) | wall time of the whole `analyze` call |
| chunks | chunks actually summarised (distributor counts summed) |
| chunks/min | chunks ÷ wall time, in minutes |
| per-peer tasks | `host:count` pairs in peer order — who did how much work |

## Methodology notes

- **Machines, models, network:** the table records the machine and
  the model for every row. The network is the LAN the peers sit on;
  record the setup (wired vs Wi-Fi, which machines) alongside the
  run, because a slow link swamps the peer-count effect. No runs
  have been recorded yet — the table is empty until the first real
  run on hardware with Ollama peers.
- **MLX joins automatically:** `discover()` adds an MLX peer when a
  server is listening on `localhost:8080`. It will show up in the
  per-peer column. Stop the MLX server for a pure Ollama comparison.
- **The distributor picks by latency:** a fast peer gets more chunks
  (latency-weighted scheduling), so per-peer counts are rarely even —
  that is the system working, not noise. Re-run a row before
  concluding a peer is slow: the first run also warms the model.
- **Reproduce a row:** same sample repo, same peer list order, same
  model, peers already warm (`ollama pull` the model on each peer
  first), then run the command above.
