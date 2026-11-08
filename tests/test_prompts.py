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


# --- v2 language contracts (Issue #2) -----------------------------------------
# Fix + test-generation contracts for JS/TS, Go, Rust, selectable by
# file extension. Hard rule: the Python contracts stay byte-stable.

def test_language_selectors_by_extension():
    cases = {
        "a.py": "python",
        "src/app.js": "javascript",
        "src/app.jsx": "javascript",
        "src/app.mjs": "javascript",
        "src/app.cjs": "javascript",
        "src/app.ts": "typescript",
        "src/app.tsx": "typescript",
        "src/app.mts": "typescript",
        "src/app.cts": "typescript",
        "cmd/main.go": "go",
        "src/lib.rs": "rust",
    }
    for path, expected in cases.items():
        assert prompts.language_for_path(path) == expected, path
        assert prompts.fix_system_for_path(path) is prompts.FIX_SYSTEMS[expected]
        assert prompts.testgen_system_for_path(path) is prompts.TESTGEN_SYSTEMS[expected]
    # Unknown extensions, no extension, and empty paths fall back to Python.
    for path in ("notes.txt", "README.md", "Makefile", "noext", "", "a.PY"):
        assert prompts.language_for_path(path) == "python", path
    # Extension match is case-insensitive.
    assert prompts.language_for_path("src/App.JS") == "javascript"
    assert prompts.language_for_path("src/Lib.RS") == "rust"


def test_language_aliases_and_fallback():
    assert prompts.fix_system_for("js") is prompts.JS_FIX_SYSTEM
    assert prompts.fix_system_for("ts") is prompts.TS_FIX_SYSTEM
    assert prompts.fix_system_for("rs") is prompts.RUST_FIX_SYSTEM
    assert prompts.fix_system_for("go") is prompts.GO_FIX_SYSTEM
    assert prompts.fix_system_for("golang") is prompts.GO_FIX_SYSTEM
    assert prompts.fix_system_for("py") is prompts.FIX_SYSTEM
    assert prompts.testgen_system_for("typescript") is prompts.TS_TESTGEN_SYSTEM
    assert prompts.testgen_system_for("javascript") is prompts.JS_TESTGEN_SYSTEM
    # None, unknown names, and mixed case degrade to the Python contracts.
    assert prompts.fix_system_for(None) is prompts.FIX_SYSTEM
    assert prompts.fix_system_for("cobol") is prompts.FIX_SYSTEM
    assert prompts.fix_system_for("JS") is prompts.JS_FIX_SYSTEM
    assert prompts.testgen_system_for("") is prompts.TESTGEN_SYSTEM
    assert prompts.normalize_language(None) == "python"
    assert prompts.normalize_language("  Go  ") == "go"


def test_new_fix_system_contracts():
    checks = {
        "javascript": (prompts.JS_FIX_SYSTEM, "CommonJS"),
        "typescript": (prompts.TS_FIX_SYSTEM, "as any"),
        "go": (prompts.GO_FIX_SYSTEM, "gofmt"),
        "rust": (prompts.RUST_FIX_SYSTEM, "unwrap()"),
    }
    for lang, (system, marker) in checks.items():
        assert "```diff" in system, lang
        assert "ONLY" in system, lang
        assert "NO_FIX_POSSIBLE" in system, lang
        assert marker in system, lang
        assert system != prompts.FIX_SYSTEM, lang
        assert system is prompts.FIX_SYSTEMS[lang], lang


def test_new_testgen_system_contracts():
    checks = {
        "javascript": (prompts.JS_TESTGEN_SYSTEM, "```javascript", "jest"),
        "typescript": (prompts.TS_TESTGEN_SYSTEM, "```typescript", "jest"),
        "go": (prompts.GO_TESTGEN_SYSTEM, "```go", "TestXxx"),
        "rust": (prompts.RUST_TESTGEN_SYSTEM, "```rust", "#[cfg(test)]"),
    }
    for lang, (system, fence, marker) in checks.items():
        assert fence in system, lang
        assert marker in system, lang
        assert "real assertion" in system, lang
        assert system != prompts.TESTGEN_SYSTEM, lang
        assert system is prompts.TESTGEN_SYSTEMS[lang], lang


def test_build_fix_prompt_language():
    # Python is the default and reproduces the exact v1 output.
    v1 = prompts.build_fix_prompt("boom", [("a.py", "x = 1")])
    assert "```python" in v1
    assert prompts.build_fix_prompt("boom", [("a.py", "x = 1")], language="python") == v1
    # The fence follows the first code block's extension.
    js = prompts.build_fix_prompt("boom", [("app.js", "const x = 1;")])
    assert "```javascript" in js and "```python" not in js
    assert prompts.build_fix_prompt("boom", [("main.go", "x := 1")]).count("```go") == 1
    assert "```rust" in prompts.build_fix_prompt("boom", [("lib.rs", "let x = 1;")])
    assert "```typescript" in prompts.build_fix_prompt("boom", [("app.ts", "const x: number = 1;")])
    # An explicit language wins over the file extension.
    forced = prompts.build_fix_prompt("boom", [("a.py", "x = 1")], language="go")
    assert "```go" in forced and "```python" not in forced


def test_build_testgen_prompt_language():
    # Python reproduces the exact v1 output.
    v1 = prompts.build_testgen_prompt("calc", "def add(a, b): ...")
    assert prompts.build_testgen_prompt("calc", "def add(a, b): ...") == v1
    assert "import calc" in v1 and "pytest" in v1
    js = prompts.build_testgen_prompt("calc", "export function add(a, b) {}", language="js")
    assert "```javascript" in js and "jest" in js
    assert "import calc" not in js  # the Python import line is Python-only
    ts = prompts.build_testgen_prompt("calc", "export function add(a: number, b: number) {}", language="ts")
    assert "```typescript" in ts and "jest" in ts
    go = prompts.build_testgen_prompt("calc", "func Add(a, b int) int { return a + b }", language="go")
    assert "```go" in go and "go test" in go
    rs = prompts.build_testgen_prompt("calc", "pub fn add(a: i32, b: i32) -> i32 { a + b }", language="rust")
    assert "```rust" in rs and "cargo test" in rs


def test_builders_v3_language():
    # The v3 fix-family builders take the same language parameter.
    msg = prompts.build_multi_file_fix_prompt(
        "boom", [("main.go", "x := 1"), ("util.go", "y := 2")]
    )
    assert "```go" in msg and "```python" not in msg
    msg = prompts.build_rediagnosis_prompt(
        "boom", "--- a/lib.rs\n+++ b/lib.rs", [], [("lib.rs", "let x = 1;")]
    )
    assert "```rust" in msg
    msg = prompts.build_repair_prompt(
        "boom", "FAILED x", [("app.ts", "const x = 1;")]
    )
    assert "```typescript" in msg
    # And their Python output is unchanged.
    assert prompts.build_multi_file_fix_prompt(
        "boom", [("a.py", "x = 1")]
    ) == prompts.build_multi_file_fix_prompt("boom", [("a.py", "x = 1")], language="python")


def test_python_contracts_unchanged_by_v2():
    # No v2 language leaks into the Python contracts.
    for marker in ("CommonJS", "as any", "gofmt", "unwrap()", "jest", "cargo test"):
        assert marker not in prompts.FIX_SYSTEM, marker
        assert marker not in prompts.TESTGEN_SYSTEM, marker
    # The v3 contracts are untouched by the v2 additions.
    assert "One fence for ALL files" in prompts.MULTI_FILE_FIX_SYSTEM
    assert "structured test failures" in prompts.REDIAGNOSIS_SYSTEM
    assert "exactly one paragraph" in prompts.INDEX_SUMMARY_SYSTEM
