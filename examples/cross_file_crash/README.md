# cross_file_crash

A tiny shop package with a bug that cannot be found by staring at the
crash site alone.

## Symptom

`checkout_total()` crashes with a `TypeError` whenever it totals a
basket. The traceback points at `shop/checkout.py` — but the code that
is wrong lives somewhere else. See `crash.log` for the exact failure.

## Reproduce

```bash
cd examples/cross_file_crash
python -m pytest
```

## Expected test command

```bash
python -m pytest
```

Both tests in `tests/test_checkout.py` should pass: the basket total
with the 10% member discount applied. Right now both fail with the
crash in `crash.log`.

## What this is for

A demo target for swarm-agent's cross-file work: the fix has to trace
the crash in one module back to the defect in another, then patch it.
(Phase 3, issue #12; used by the Core lane's cross-file retrieval
and by the fix-benchmark.)
