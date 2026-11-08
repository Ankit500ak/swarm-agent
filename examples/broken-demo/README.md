# broken-demo — the quick-start demo for `swarm-agent fix`

A tiny gradebook summary: `demo.py` reads `data.json` and prints
the average score, the best score and how many scores there are.

## Symptom

It crashes instead of printing anything:

```bash
$ python demo.py
Traceback (most recent call last):
  ...
KeyError: 'score'
```

The same failure shows up as 3 failed tests — see `crash.log`,
which is the real captured output of `python -m pytest` from this
directory. The README states the symptom, not the answer: the defect
is one wrong word in `demo.py`.

## Try the fix

With an Ollama peer up and serving `qwen2.5-coder:7b` (see the repo
README for the setup), run from this directory:

```bash
cat crash.log | swarm-agent fix
```

The agent reads the crash, finds the defect, and proposes the fix.
Add `--apply` to have it patch `demo.py` and re-run the tests:

```bash
cat crash.log | swarm-agent fix --apply
```

Success looks like this — all three tests green:

```bash
$ python -m pytest
3 passed
```

## Recording

A short terminal recording of a real fix run (asciinema cast or
GIF) belongs here, linked from this README. It is not recorded yet:
it needs a machine with a local Ollama peer running
`qwen2.5-coder:7b`. Once captured, drop the file in this directory
and link it here.
