# off_by_one — a one-line slicing bug

`recent_items(items, n)` should return the last `n` items of
`items`, in order. Instead it returns one too many:

```bash
$ python -c "from recent import recent_items; print(recent_items([1, 2, 3, 4, 5], 3))"
[2, 3, 4, 5]
```

The tests in `tests/test_recent.py` fail — see `crash.log`,
the real captured output of `python -m pytest` from this directory
(2 failed, 1 passed: the "all items" case passes by luck, because
the overshoot slice clamps to the start of the list).
The defect is a single slice expression in `recent.py`.
