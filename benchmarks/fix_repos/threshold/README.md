# threshold — a boundary bug

`is_adult(age)` should return `True` when `age` is 18 or older.
Instead it excludes exactly 18:

```bash
$ python -c "from check import is_adult; print(is_adult(18))"
False
```

One of the three tests in `tests/test_check.py` fails — see
`crash.log`, the real captured output of `python -m pytest` from
this directory (1 failed, 2 passed: only the boundary case is
wrong). The defect is a single comparison operator in `check.py`.
