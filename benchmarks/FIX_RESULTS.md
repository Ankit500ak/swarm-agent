# Fix-benchmark results — swarm-agent fix --apply on seeded bugs

Methodology: see [README.md](README.md). Each row is one run of
`swarm-agent fix --apply` against a seeded-bug repo, copied to a
scratch directory first (the committed repo is never mutated). The
outcome is grounded in an independent re-run of the repo's tests
after the fix, not in what the fix run claimed: `fixed attempt 1`
means the first fix made the tests pass, `fixed after repair` means
the repair loop had to run, `failed` means the tests still fail.

| date | repo | bug | outcome | attempts | model | machine | notes |
|---|---|---|---|---|---|---|---|
