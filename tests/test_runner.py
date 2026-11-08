"""Tests for swarm_agent.runner.parse_failures (Issue #10, Phase 3).

The two big fixtures are REAL captured outputs, not hand-written
approximations:

- PYTEST_OUTPUT: `python -m pytest tests/test_calc.py -q` on a tiny
  project with three failing tests (two plain functions, one method
  in a class). Captured 2026-11-08, pytest 9.2.0. (One memory address
  in a `self = <...>` line is normalised to 0x... — it varies per run
  and the parser never reads it.)
- JEST_OUTPUT: `npx jest --no-coverage` on a tiny project with two
  failing tests (one top-level, one inside describe('calculator')).
  Captured 2026-11-08, jest 29. The FAIL line, the ✕ summary lines,
  the ● detail sections and the code frames are all as jest printed
  them.
"""
from swarm_agent.runner import parse_failures


PYTEST_OUTPUT = """
FFF                                                                      [100%]
=================================== FAILURES ===================================
___________________________________ test_add ___________________________________

    def test_add():
>       assert add(1, 2) == 4
E       assert 3 == 4
E        +  where 3 = add(1, 2)

tests/test_calc.py:6: AssertionError
________________________________ test_subtract _________________________________

    def test_subtract():
>       assert add(5, 3) == 1
E       assert 8 == 1
E        +  where 8 = add(5, 3)

tests/test_calc.py:10: AssertionError
____________________________ TestDivide.test_divide ____________________________

self = <test_calc.TestDivide object at 0x...>

    def test_divide(self):
>       assert 10 / 2 == 4
E       assert (10 / 2) == 4

tests/test_calc.py:15: AssertionError
=========================== short test summary info ============================
FAILED tests/test_calc.py::test_add - assert 3 == 4
FAILED tests/test_calc.py::test_subtract - assert 8 == 1
FAILED tests/test_calc.py::TestDivide::test_divide - assert (10 / 2) == 4
3 failed in 0.03s
"""

JEST_OUTPUT = """
FAIL ./sum.test.js
  ✕ adds 1 + 2 to equal 3 (6 ms)
  calculator
    ✕ subtracts 5 - 3 to equal 1 (1 ms)

  ● adds 1 + 2 to equal 3

    expect(received).toBe(expected) // Object.is equality

    Expected: 4
    Received: 3

      1 | function sum(a, b) { return a + b; }
    > 2 | test('adds 1 + 2 to equal 3', () => { expect(sum(1, 2)).toBe(4); });
        |                                                         ^
      3 | describe('calculator', () => {
      4 |   test('subtracts 5 - 3 to equal 1', () => { expect(sum(5, -3)).toBe(1); });
      5 | });

      at Object.toBe (sum.test.js:2:57)

  ● calculator › subtracts 5 - 3 to equal 1

    expect(received).toBe(expected) // Object.is equality

    Expected: 1
    Received: 2

      2 | test('adds 1 + 2 to equal 3', () => { expect(sum(1, 2)).toBe(4); });
      3 | describe('calculator', () => {
    > 4 |   test('subtracts 5 - 3 to equal 1', () => { expect(sum(5, -3)).toBe(1); });
        |                                                                 ^
      5 | });
      6 |

      at Object.toBe (sum.test.js:4:65)

Test Suites: 1 failed, 1 total
Tests:       2 failed, 2 total
Snapshots:   0 total
Time:        0.355 s, estimated 1 s
Ran all test suites.
"""


# --- pytest ---------------------------------------------------------------

def test_pytest_three_failures():
    records = parse_failures(PYTEST_OUTPUT, "pytest")
    assert len(records) == 3
    by_test = {r["test"]: r for r in records}
    assert by_test["test_add"]["file"] == "tests/test_calc.py"
    assert by_test["test_add"]["message"] == "assert 3 == 4"
    assert by_test["test_subtract"]["message"] == "assert 8 == 1"
    assert by_test["TestDivide::test_divide"]["file"] == "tests/test_calc.py"
    assert by_test["TestDivide::test_divide"]["message"] == "assert (10 / 2) == 4"


def test_pytest_record_shape_is_frozen_contract():
    # Issue #10 freezes this shape; the Core lane (#16) builds on it.
    for record in parse_failures(PYTEST_OUTPUT, "pytest"):
        assert set(record.keys()) == {"file", "test", "message"}
        assert all(isinstance(v, str) for v in record.values())


def test_pytest_passing_output_is_empty():
    assert parse_failures("3 passed in 0.04s\n", "pytest") == []
    assert parse_failures("", "pytest") == []
    assert parse_failures("   \n", "pytest") == []


def test_pytest_verbose_failed_lines_use_section_detail():
    output = (
        "tests/test_calc.py::test_add FAILED [ 33%]\n"
        "=================================== FAILURES ===================================\n"
        "___________________________________ test_add ___________________________________\n"
        "\n"
        "    def test_add():\n"
        ">       assert add(1, 2) == 4\n"
        "E       assert 3 == 4\n"
        "\n"
        "tests/test_calc.py:6: AssertionError\n"
    )
    records = parse_failures(output, "pytest")
    assert len(records) == 1
    assert records[0]["file"] == "tests/test_calc.py"
    assert records[0]["test"] == "test_add"
    assert records[0]["message"] == "assert 3 == 4"


def test_pytest_error_lines_are_failures_too():
    output = "ERROR tests/test_x.py::test_setup - fixture \'db\' not found\n"
    records = parse_failures(output, "pytest")
    assert len(records) == 1
    assert records[0]["test"] == "test_setup"
    assert "not found" in records[0]["message"]


# --- jest -----------------------------------------------------------------

def test_jest_two_failures():
    records = parse_failures(JEST_OUTPUT, "jest")
    assert len(records) == 2
    by_test = {r["test"]: r for r in records}
    assert by_test["adds 1 + 2 to equal 3"]["file"] == "sum.test.js"
    assert "Expected: 4" in by_test["adds 1 + 2 to equal 3"]["message"]
    assert "Received: 3" in by_test["adds 1 + 2 to equal 3"]["message"]
    nested = by_test["calculator \u203a subtracts 5 - 3 to equal 1"]
    assert nested["file"] == "sum.test.js"
    assert "Expected: 1" in nested["message"]
    assert "Received: 2" in nested["message"]
    # The code frame is noise, not the message.
    for record in records:
        assert "function sum" not in record["message"]
        assert "at Object.toBe" not in record["message"]


def test_jest_record_shape_is_frozen_contract():
    for record in parse_failures(JEST_OUTPUT, "jest"):
        assert set(record.keys()) == {"file", "test", "message"}
        assert all(isinstance(v, str) for v in record.values())


def test_jest_npm_alias_and_passing_output():
    assert len(parse_failures(JEST_OUTPUT, "npm")) == 2
    assert parse_failures("Test Suites: 1 passed, 1 total\nTests: 2 passed, 2 total\n", "jest") == []
    assert parse_failures("", "jest") == []


def test_jest_cross_only_output_still_records():
    # A truncated tail may keep only the suite summary (✕ lines), no ●
    # sections. Those tests still get a record, with message "failed".
    output = (
        "FAIL ./sum.test.js\n"
        "  \u2715 adds 1 + 2 to equal 3 (5 ms)\n"
        "  calculator\n"
        "    \u2715 subtracts 5 - 3 to equal 1 (1 ms)\n"
    )
    records = parse_failures(output, "jest")
    assert len(records) == 2
    assert {r["test"] for r in records} == {
        "adds 1 + 2 to equal 3",
        "subtracts 5 - 3 to equal 1",
    }
    assert all(r["message"] == "failed" for r in records)
    assert all(r["file"] == "sum.test.js" for r in records)


def test_jest_two_files_do_not_bleed_into_each_other():
    # Regression: the message collector used to swallow the next suite's
    # FAIL line into the previous test's message, and the second test
    # then inherited the first file. A new FAIL/PASS line ends a section.
    output = (
        "FAIL ./a.test.js\n"
        "  \u25cf first test\n"
        "\n"
        "    Expected: 1\n"
        "    Received: 2\n"
        "\n"
        "FAIL ./b.test.js\n"
        "  \u25cf second test\n"
        "\n"
        "    boom happened\n"
    )
    records = parse_failures(output, "jest")
    assert [(r["file"], r["test"]) for r in records] == [
        ("a.test.js", "first test"),
        ("b.test.js", "second test"),
    ]
    assert records[0]["message"] == "Expected: 1\nReceived: 2"
    assert "FAIL" not in records[0]["message"]
    assert records[1]["message"] == "boom happened"


# --- degradation: never an exception --------------------------------------

def test_unparseable_failure_output_degrades_to_raw_tail():
    output = "something broke badly\nFAILED to start the thing\n"
    records = parse_failures(output, "pytest")
    assert len(records) == 1
    assert records[0] == {"file": "", "test": "", "message": output.strip()}


def test_unknown_ecosystem_never_raises():
    output = "FAILED tests/test_x.py::test_y - boom\n"
    records = parse_failures(output, "go")
    assert len(records) == 1
    assert records[0]["message"] == output.strip()
    assert parse_failures("all good here\n", "go") == []
