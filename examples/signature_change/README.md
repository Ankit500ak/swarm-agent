# signature_change

A tiny ledger with a bug that genuinely spans two files: fixing only
one of them is not enough.

## Symptom

`monthly_total()` crashes with a `TypeError` — a required
keyword-only argument is missing. The traceback points at
`ledger/report.py`. See `crash.log` for the exact failure.

## Reproduce

```bash
cd examples/signature_change
python -m pytest
```

## Expected test command

```bash
python -m pytest
```

Both tests in `tests/test_report.py` should pass: monthly totals in
USD, with EUR entries converted at 1.10. Right now both fail with the
crash in `crash.log`.

## What this is for

A demo target for swarm-agent's cross-file work: the fix has to change
a function's signature and its caller together, then patch both files
in one diff. (Phase 3, issue #12; used by the Core lane's cross-file
retrieval and multi-file patching, and by the fix-benchmark.)
