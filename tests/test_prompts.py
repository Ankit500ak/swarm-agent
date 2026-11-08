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
