# Benchmark results — analyse, 1 vs 2 vs 3 peers

Methodology: see [README.md](README.md). Each row is one run of
`analyze` over the pinned sample repo (4 files, 16 chunks at the
default 1500-token budget). Wall time covers the whole analyse call,
including the final synthesis. Chunks are chunks actually summarised
(the distributor's per-peer counts summed). Per-peer tasks are
`host:count` pairs in peer order.

| date | machine | model | ollama peers | wall (s) | chunks | chunks/min | per-peer tasks |
|---|---|---|---|---|---|---|---|
