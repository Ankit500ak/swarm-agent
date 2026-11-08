from swarm_agent import prompts


def test_fix_prompt_contract():
    # Qwen-class models need the output contract stated bluntly.
    assert "```diff" in prompts.FIX_SYSTEM
    assert "ONLY" in prompts.FIX_SYSTEM
    assert "NO_FIX_POSSIBLE" in prompts.FIX_SYSTEM


def test_testgen_prompt_contract():
    assert "```python" in prompts.TESTGEN_SYSTEM
    assert "pytest" in prompts.TESTGEN_SYSTEM


def test_analyze_prompt_contract():
    for marker in ("PURPOSE:", "KEY:", "RISKS:"):
        assert marker in prompts.ANALYZE_SYSTEM
    for section in ("## Purpose", "## Architecture", "## Key modules",
                    "## Risks", "## Suggested next steps"):
        assert section in prompts.ANALYZE_SYNTHESIS_SYSTEM


def test_builders():
    msg = prompts.build_fix_prompt("boom", [("a.py", "x = 1")])
    assert "boom" in msg and "a.py" in msg and "x = 1" in msg
    msg = prompts.build_testgen_prompt("calc", "def add(a, b): ...")
    assert "import calc" in msg
    msg = prompts.build_synthesis_prompt(["[a.py] PURPOSE: x"], "what is this?")
    assert "what is this?" in msg


# --- v3 contracts (Issue #9, Phase 3) ------------------------------------------
# Hard rule: the v1/v2 contracts above stay byte-stable. These tests pin
# that the v3 additions did not leak into them, and pin the new contracts.

def test_v1_v2_contracts_unchanged_by_v3():
    # v1 FIX_SYSTEM is the single-file contract: no multi-file language.
    assert "spans several files" not in prompts.FIX_SYSTEM
    assert "One fence for ALL files" not in prompts.FIX_SYSTEM
    assert "file section" not in prompts.FIX_SYSTEM
    # v1 repair builder still takes the raw test-output tail, not records.
    msg = prompts.build_repair_prompt("boom", "FAILED x", [("a.py", "x = 1")])
    assert "TEST OUTPUT AFTER YOUR FIX" in msg
    assert "STRUCTURED FAILURES" not in msg
    assert "PREVIOUS DIFF" not in msg


def test_multi_file_fix_system_contract():
    system = prompts.MULTI_FILE_FIX_SYSTEM
    assert "```diff" in system
    assert "One fence for ALL files" in system
    assert "one file section per changed file" in system
    assert "--- a/<path>" in system
    assert "at least one hunk" in system
    assert "Never use absolute paths" in system
    assert "NO_FIX_POSSIBLE" in system
    assert "Do not add `diff --git` headers" in system


def test_rediagnosis_system_contract():
    system = prompts.REDIAGNOSIS_SYSTEM
    assert "structured test failures" in system
    assert "Do not repeat the previous diff" in system
    assert "```diff" in system
    assert "current file state" in system
    assert "NO_FIX_POSSIBLE" in system


def test_index_summary_system_contract():
    system = prompts.INDEX_SUMMARY_SYSTEM
    assert "exactly one paragraph" in system
    assert "3-6 sentences" in system
    assert "No headings" in system
    assert "Do not invent behaviour" in system


def test_build_multi_file_fix_prompt():
    msg = prompts.build_multi_file_fix_prompt(
        "crash in a.py", [("a.py", "x = 1"), ("b.py", "y = 2")]
    )
    assert "crash in a.py" in msg
    assert "a.py" in msg and "b.py" in msg
    assert "single ```diff fence" in msg
    assert "one section per changed file" in msg
    # Distinct from the v1 builder's closing line.
    assert msg != prompts.build_fix_prompt(
        "crash in a.py", [("a.py", "x = 1"), ("b.py", "y = 2")]
    )


def test_build_rediagnosis_prompt_uses_frozen_failure_shape():
    # Records in exactly the shape issue #10 froze: {file, test, message}.
    failures = [
        {"file": "tests/test_calc.py", "test": "test_add", "message": "assert 3 == 4"},
        {"file": "tests/test_calc.py", "test": "test_sub", "message": "assert 8 == 1"},
    ]
    diff = "--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-x\n+y\n"
    msg = prompts.build_rediagnosis_prompt(
        "boom", diff, failures, [("calc.py", "def add(a, b): return a + b")],
        attempt=2,
    )
    assert "repair attempt 2" in msg
    assert "ORIGINAL ERROR" in msg and "boom" in msg
    assert "PREVIOUS DIFF" in msg and "--- a/calc.py" in msg
    assert "STRUCTURED FAILURES" in msg
    assert "- tests/test_calc.py :: test_add — assert 3 == 4" in msg
    assert "- tests/test_calc.py :: test_sub — assert 8 == 1" in msg
    assert "FILE (current state): calc.py" in msg
    assert "corrected unified diff" in msg
    assert "TEST OUTPUT AFTER YOUR FIX" not in msg  # that is the v1 builder


def test_build_rediagnosis_prompt_attempt_optional_and_empty_failures():
    msg = prompts.build_rediagnosis_prompt("boom", "--- a/x\n+++ b/x", [], [("x.py", "x")])
    assert "repair attempt" not in msg
    assert "No structured failures were recorded." in msg
    # A record missing keys degrades gracefully instead of raising.
    msg2 = prompts.build_rediagnosis_prompt(
        "boom", "d", [{"file": "a.py"}], [("a.py", "x")]
    )
    assert "- a.py ::  — " in msg2


def test_build_index_summary_prompt():
    summaries = [
        "PURPOSE: parses test output\nKEY: parse_failures\nRISKS: none",
        "PURPOSE: runs the suite\nKEY: run_tests\nRISKS: timeouts",
    ]
    msg = prompts.build_index_summary_prompt("swarm_agent/runner.py", summaries)
    assert "MODULE: swarm_agent/runner.py" in msg
    assert "parses test output" in msg and "runs the suite" in msg
    assert "one-paragraph index summary" in msg
