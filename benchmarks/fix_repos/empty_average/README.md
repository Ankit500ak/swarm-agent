# empty_average — a missing guard

`average(values)` should return the mean of the numbers, and `0.0`
for an empty list. Instead it crashes on the empty case:

```bash
$ python -c "from stats import average; print(average([]))"
Traceback (most recent call last):
  ...
ZeroDivisionError: division by zero
```

One of the two tests in `tests/test_stats.py` fails — see `crash.log`,
the real captured output of `python -m pytest` from this directory
(1 failed, 1 passed: the non-empty case is fine).
The defect is a missing guard in `stats.py`.
