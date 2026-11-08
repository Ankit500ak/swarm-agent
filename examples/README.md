# examples

Demo projects for swarm-agent. `broken-demo/` is the quick-start
demo for `swarm-agent fix` (Phase 2, issue #5): one tiny project,
one seeded bug, one crash log. The other two are the cross-file
demos (Phase 3, issue #12). Each one is intentionally broken: run
its tests and they fail exactly as documented in its `crash.log`.
The cross-file projects are the demo targets for the Core lane's
cross-file retrieval (#14) and multi-file patching (#15), and the
seeded-bug repos for the fix-benchmark (#11).

| Project | Bug | Run it |
| --- | --- | --- |
| `broken-demo/` | A single wrong word in `demo.py` — the gradebook summary crashes with a `KeyError` before printing anything. The `cat crash.log \| swarm-agent fix` demo. | `cd broken-demo && python -m pytest` |
| `cross_file_crash/` | The crash surfaces in `shop/checkout.py`, but the defect lives in `shop/catalog.py` (wrong units/contract across the boundary). | `cd cross_file_crash && python -m pytest` |
| `signature_change/` | The fix genuinely spans two files: a function's signature and its caller (`ledger/store.py` + `ledger/report.py`). | `cd signature_change && python -m pytest` |

Each project ships its own `pytest.ini` and `conftest.py`, so the
test command above runs that project's tests from its own directory.
The repository's own test suite (`python -m pytest` from the repo
root) is unaffected — it only collects `tests/`.
