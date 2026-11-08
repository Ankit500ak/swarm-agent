# examples

Demo projects for swarm-agent's cross-file work (Phase 3, issue #12).
Each one is intentionally broken: run its tests and they fail exactly
as documented in its `crash.log`. They are the demo targets for the
Core lane's cross-file retrieval (#14) and multi-file patching (#15),
and the seeded-bug repos for the fix-benchmark (#11).

| Project | Bug | Run it |
| --- | --- | --- |
| `cross_file_crash/` | The crash surfaces in `shop/checkout.py`, but the defect lives in `shop/catalog.py` (wrong units/contract across the boundary). | `cd cross_file_crash && python -m pytest` |
| `signature_change/` | The fix genuinely spans two files: a function's signature and its caller (`ledger/store.py` + `ledger/report.py`). | `cd signature_change && python -m pytest` |

Each project ships its own `pytest.ini` and `conftest.py`, so the
test command above runs that project's tests from its own directory.
The repository's own test suite (`python -m pytest` from the repo
root) is unaffected — it only collects `tests/`.
