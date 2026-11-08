"""Language wiring (issue #28): the flows select prompt contracts by
file extension, via the selectors landed in issue #2.

- fix: the working language comes from the traceback's deepest (last)
  gathered file; it drives the system contract (`fix_system_for`),
  the prompt fences (`build_fix_prompt(language=...)`) and, on
  repairs, the re-diagnosis prompt (`build_rediagnosis_prompt`).
- gen-tests: the contract comes from the target module's extension.
- Python is byte-identical to the pre-wiring behaviour: the language
  "python" resolves to the very same constants and builder outputs
  the flows used before the selectors existed.
"""

import hashlib
import re

import pytest

from swarm_agent import agent, prompts
from swarm_agent.config import Config
from swarm_agent.ollama import SwarmError
from swarm_agent.runner import parse_failures


# --------------------------------------------------------------------------
# fakes (same shape as test_repair_loop's: scripted chat + fake embed)
# --------------------------------------------------------------------------

class ScriptedClient:
    """Chat returns the next canned reply; embed is bag-of-words."""

    DIM = 64

    def __init__(self, replies: list[str]):
        self._replies = list(replies)

    def chat(self, messages, model=None):
        if self._replies:
            return self._replies.pop(0)
        return "NO_FIX_POSSIBLE"

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * ScriptedClient.DIM
        for tok in re.findall(r"[a-z0-9_]+", text.lower()):
            bucket = int.from_bytes(
                hashlib.md5(tok.encode()).digest()[:4], "little"
            ) % ScriptedClient.DIM
            vec[bucket] += 1.0
        return vec

    def embed(self, texts, model=None):
        return [self._vector(t) for t in texts]


class FakeDistributor:
    """Stands in for Distributor: `ask` chats, `client_for` embeds."""

    def __init__(self, client: ScriptedClient):
        self._client = client
        self.asked: list = []

    def ask(self, messages, model=None):
        self.asked.append(messages)
        return self._client.chat(messages, model=model)

    def best_peer(self):
        return "fake-peer"

    def client_for(self, peer):
        return self._client


def _install(monkeypatch, replies) -> FakeDistributor:
    dist = FakeDistributor(ScriptedClient(replies))
    monkeypatch.setattr(agent, "get_distributor", lambda cfg: dist)
    return dist


def _reply(diff_body: str) -> str:
    return "Here is the fix:\n\n```diff\n" + diff_body + "\n```\n"


# --------------------------------------------------------------------------
# (a) a Python fix produces exactly the pre-wiring prompts
# --------------------------------------------------------------------------

CALC = "def add(a, b):\n    return a - b\n"

PY_ERROR = (
    "Traceback (most recent call last):\n"
    '  File "calc.py", line 2, in add\n'
    "    return a - b\n"
    "ValueError: boom\n"
)

PY_DIFF = _reply(
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b"
)


def test_python_fix_prompts_are_byte_identical(monkeypatch, tmp_path):
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8")
    dist = _install(monkeypatch, [PY_DIFF])

    rc = agent.fix_flow(PY_ERROR, tmp_path, Config(), apply=False)

    assert rc == 0
    assert len(dist.asked) == 1
    system, user = dist.asked[0][0]["content"], dist.asked[0][1]["content"]
    # The system contract is the very constant the flow used before.
    assert system == prompts.FIX_SYSTEM
    assert system is prompts.fix_system_for("python")
    # The user prompt equals the builders' output with NO language
    # argument (the pre-wiring call) - and, equivalently, with the
    # language the wiring now passes explicitly.
    blocks = agent._gather_traceback_files(PY_ERROR, tmp_path, Config())
    assert user == prompts.build_fix_prompt(PY_ERROR, blocks)
    assert user == prompts.build_fix_prompt(PY_ERROR, blocks, language="python")


# --------------------------------------------------------------------------
# (b) a JS fix takes its contract from the traceback's deepest file
# --------------------------------------------------------------------------

APP_JS = "function add(a, b) {\n  return a - b;\n}\n"
RUNNER_PY = "def main():\n    pass\n"

# Outermost frame is a Python file; the deepest (where it blew up) is JS.
JS_ERROR = (
    "Traceback (most recent call last):\n"
    '  File "runner.py", line 10, in main\n'
    "    main()\n"
    '  File "app.js", line 2, in add\n'
    "    return a - b;\n"
    "TypeError: boom\n"
)

JS_DIFF = _reply(
    "--- a/app.js\n"
    "+++ b/app.js\n"
    "@@ -1,2 +1,2 @@\n"
    " function add(a, b) {\n"
    "-  return a - b;\n"
    "+  return a + b;"
)


def test_js_fix_uses_js_contract_from_deepest_file(monkeypatch, tmp_path):
    (tmp_path / "runner.py").write_text(RUNNER_PY, encoding="utf-8")
    (tmp_path / "app.js").write_text(APP_JS, encoding="utf-8")
    dist = _install(monkeypatch, [JS_DIFF])

    rc = agent.fix_flow(JS_ERROR, tmp_path, Config(), apply=False)

    assert rc == 0
    system, user = dist.asked[0][0]["content"], dist.asked[0][1]["content"]
    # System: the JS contract, not the Python one. The distinguishing
    # rules are read from the prompts module, not hardcoded here.
    assert system == prompts.JS_FIX_SYSTEM
    assert system != prompts.FIX_SYSTEM
    js_only = [
        line for line in prompts.JS_FIX_SYSTEM.splitlines()
        if line.strip() and line not in prompts.FIX_SYSTEM
    ]
    assert js_only and all(line in system for line in js_only)
    # User: built with the working language - JS fences throughout,
    # where the builders' old default (first file = runner.py) would
    # have produced Python fences.
    blocks = agent._gather_traceback_files(JS_ERROR, tmp_path, Config())
    assert blocks[-1][0] == "app.js"  # deepest frame is last
    assert user == prompts.build_fix_prompt(
        JS_ERROR, blocks, language="javascript"
    )
    assert "```javascript" in user
    assert "```python" not in user
    assert user != prompts.build_fix_prompt(JS_ERROR, blocks)


# --------------------------------------------------------------------------
# (b2) repairs re-diagnose in the fix's working language too
# --------------------------------------------------------------------------

APP_JS_TWO_BUGS = (
    "function add(a, b) {\n  return a - b;\n}\n"
    "\n"
    "function sub(a, b) {\n  return a + b;\n}\n"
)

JS_ERROR_ONE_FILE = (
    "Error: boom\n"
    '  File "app.js", line 2, in add\n'
    "    return a - b;\n"
)

DIFF1 = _reply(
    "--- a/app.js\n"
    "+++ b/app.js\n"
    "@@ -1,2 +1,2 @@\n"
    " function add(a, b) {\n"
    "-  return a - b;\n"
    "+  return a + b;"
)

DIFF2 = _reply(
    "--- a/app.js\n"
    "+++ b/app.js\n"
    "@@ -5,2 +5,2 @@\n"
    " function sub(a, b) {\n"
    "-  return a + b;\n"
    "+  return a - b;"
)

JEST_TAIL = (
    "FAIL tests/app.test.js\n"
    "  ● sub › subtracts\n"
    "    expect(received).toBe(expected)\n"
    "    Expected: 2\n"
    "    Received: 8\n"
)


def test_repair_rediagnosis_gets_the_working_language(monkeypatch, tmp_path):
    (tmp_path / "app.js").write_text(APP_JS_TWO_BUGS, encoding="utf-8")
    dist = _install(monkeypatch, [DIFF1, DIFF2])

    # Script the test runs: first verification fails, the repair's passes.
    runs = iter([(1, JEST_TAIL), (0, "all green")])
    monkeypatch.setattr(agent, "run_tests", lambda cmd, root: next(runs))

    # Spy on the builder (delegating to the real one) to capture how
    # the repair loop calls it.
    seen: dict = {}
    real_builder = prompts.build_rediagnosis_prompt

    def spy(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(prompts, "build_rediagnosis_prompt", spy)

    rc = agent.fix_flow(
        JS_ERROR_ONE_FILE, tmp_path, Config(), apply=True, test_cmd="npm test"
    )

    assert rc == 0
    assert len(dist.asked) == 2  # initial fix + one repair
    assert seen["kwargs"].get("language") == "javascript"
    assert seen["kwargs"].get("attempt") == 2
    # ...and it was fed the structured failures parsed from the run.
    assert seen["args"][2] == parse_failures(JEST_TAIL, "jest")
    repair_user = dist.asked[1][1]["content"]
    assert dist.asked[1][0]["content"] == prompts.REDIAGNOSIS_SYSTEM
    assert "FILE (current state): app.js\n```javascript" in repair_user


# --------------------------------------------------------------------------
# (c) gen-tests for a Go module selects the Go contract
# --------------------------------------------------------------------------

CALC_GO = "package calc\n\nfunc Add(a, b int) int { return a - b }\n"

GO_SUITE = (
    "package calc\n\nimport \"testing\"\n\n"
    "func TestAdd(t *testing.T) {\n"
    "\tif Add(2, 3) != 5 {\n"
    "\t\tt.Fatal(\"bad add\")\n"
    "\t}\n"
    "}"
)


def test_gen_tests_go_selects_go_contract(monkeypatch, tmp_path):
    target = tmp_path / "calc.go"
    target.write_text(CALC_GO, encoding="utf-8")
    dist = _install(monkeypatch, ["```go\n" + GO_SUITE + "\n```\n"])

    code = agent.gen_tests_flow(target, Config())

    assert code == GO_SUITE
    system, user = dist.asked[0][0]["content"], dist.asked[0][1]["content"]
    assert system == prompts.GO_TESTGEN_SYSTEM
    assert system is prompts.testgen_system_for("go")
    assert user == prompts.build_testgen_prompt(
        "calc", CALC_GO, language="go"
    )
    assert "go test" in user


# --------------------------------------------------------------------------
# (d) an unknown extension falls back to the Python contracts
# --------------------------------------------------------------------------

LEGACY = "       IDENTIFICATION DIVISION.\n       PROGRAM-ID. LEGACY.\n"

PY_SUITE = "def test_x():\n    assert True"


def test_gen_tests_unknown_extension_falls_back_to_python(
    monkeypatch, tmp_path
):
    target = tmp_path / "legacy.cob"
    target.write_text(LEGACY, encoding="utf-8")
    dist = _install(monkeypatch, ["```python\n" + PY_SUITE + "\n```\n"])

    code = agent.gen_tests_flow(target, Config())

    assert code == PY_SUITE
    system, user = dist.asked[0][0]["content"], dist.asked[0][1]["content"]
    assert system == prompts.TESTGEN_SYSTEM
    # Exactly the pre-wiring (no-language) builder output.
    assert user == prompts.build_testgen_prompt("legacy", LEGACY)


def test_fix_language_helper():
    assert agent._fix_language([]) == "python"
    assert agent._fix_language([("a.py", "")]) == "python"
    # The deepest (last) file decides, in both directions.
    assert agent._fix_language([("a.py", ""), ("b.js", "")]) == "javascript"
    assert agent._fix_language([("a.go", ""), ("b.py", "")]) == "python"
    assert agent._fix_language([("a.unknown", "")]) == "python"


# --------------------------------------------------------------------------
# gen-tests write path: JS/TS targets get jest-style file names
# --------------------------------------------------------------------------

JS_SUITE = (
    "const { add } = require('../app');\n\n"
    "test('adds', () => {\n"
    "  expect(add(1, 2)).toBe(3);\n"
    "});"
)


def test_gen_tests_js_write_uses_jest_filename(monkeypatch, tmp_path):
    target = tmp_path / "app.js"
    target.write_text(APP_JS, encoding="utf-8")
    dist = _install(monkeypatch, ["```javascript\n" + JS_SUITE + "\n```\n"])
    monkeypatch.chdir(tmp_path)

    code = agent.gen_tests_flow(target, Config(), write=True)

    assert code == JS_SUITE
    assert dist.asked[0][0]["content"] == prompts.JS_TESTGEN_SYSTEM
    out = tmp_path / "tests" / "app.test.js"
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == JS_SUITE + "\n"


# --------------------------------------------------------------------------
# gen-tests write paths (issue #38): Go in-package, Rust under tests/,
# and no destination is ever silently overwritten
# --------------------------------------------------------------------------

LEDGER_GO = "package store\n\nfunc Balance() int { return 0 }\n"

LEDGER_GO_SUITE = (
    "package store\n\nimport \"testing\"\n\n"
    "func TestBalance(t *testing.T) {\n"
    "\tif Balance() != 0 {\n"
    "\t\tt.Fatal(\"bad balance\")\n"
    "\t}\n"
    "}"
)

LEDGER_RS = "pub fn balance() -> i64 { 0 }\n"

RUST_SUITE = (
    "use ledger::balance;\n\n"
    "#[test]\n"
    "fn balance_starts_at_zero() {\n"
    "    assert_eq!(balance(), 0);\n"
    "}"
)

HANDWRITTEN_GO_TEST = (
    "package store\n\nimport \"testing\"\n\n"
    "func TestHandwritten(t *testing.T) {}\n"
)


def test_gen_tests_python_write_path_unchanged(monkeypatch, tmp_path):
    target = tmp_path / "calc.py"
    target.write_text(CALC, encoding="utf-8")
    _install(monkeypatch, ["```python\n" + PY_SUITE + "\n```\n"])
    monkeypatch.chdir(tmp_path)

    code = agent.gen_tests_flow(target, Config(), write=True)

    assert code == PY_SUITE
    out = tmp_path / "tests" / "test_calc_generated.py"
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == PY_SUITE + "\n"


def test_gen_tests_go_write_is_in_package(monkeypatch, tmp_path):
    pkg = tmp_path / "pkg" / "store"
    pkg.mkdir(parents=True)
    target = pkg / "ledger.go"
    target.write_text(LEDGER_GO, encoding="utf-8")
    dist = _install(monkeypatch, ["```go\n" + LEDGER_GO_SUITE + "\n```\n"])
    monkeypatch.chdir(tmp_path)

    code = agent.gen_tests_flow(target, Config(), write=True)

    assert code == LEDGER_GO_SUITE
    assert dist.asked[0][0]["content"] == prompts.GO_TESTGEN_SYSTEM
    # In-package, next to the target - NOT the legacy Python name.
    out = pkg / "ledger_test.go"
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == LEDGER_GO_SUITE + "\n"
    assert not (tmp_path / "tests" / "test_ledger_generated.py").exists()


def test_gen_tests_rust_write_uses_cargo_tests_dir(monkeypatch, tmp_path):
    target = tmp_path / "ledger.rs"
    target.write_text(LEDGER_RS, encoding="utf-8")
    dist = _install(monkeypatch, ["```rust\n" + RUST_SUITE + "\n```\n"])
    monkeypatch.chdir(tmp_path)

    code = agent.gen_tests_flow(target, Config(), write=True)

    assert code == RUST_SUITE
    assert dist.asked[0][0]["content"] == prompts.RUST_TESTGEN_SYSTEM
    out = tmp_path / "tests" / "ledger_test.rs"
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == RUST_SUITE + "\n"
    assert not (tmp_path / "tests" / "test_ledger_generated.py").exists()


def test_gen_tests_go_write_refuses_to_clobber_existing(
    monkeypatch, tmp_path
):
    pkg = tmp_path / "pkg" / "store"
    pkg.mkdir(parents=True)
    target = pkg / "ledger.go"
    target.write_text(LEDGER_GO, encoding="utf-8")
    existing = pkg / "ledger_test.go"
    existing.write_text(HANDWRITTEN_GO_TEST, encoding="utf-8")
    _install(monkeypatch, ["```go\n" + LEDGER_GO_SUITE + "\n```\n"])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SwarmError, match="file already exists"):
        agent.gen_tests_flow(target, Config(), write=True)

    # The hand-written suite is untouched.
    assert existing.read_text(encoding="utf-8") == HANDWRITTEN_GO_TEST


def test_gen_tests_rust_write_refuses_to_clobber_existing(
    monkeypatch, tmp_path
):
    target = tmp_path / "ledger.rs"
    target.write_text(LEDGER_RS, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    existing = tests_dir / "ledger_test.rs"
    existing.write_text("// hand-written\n", encoding="utf-8")
    _install(monkeypatch, ["```rust\n" + RUST_SUITE + "\n```\n"])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SwarmError, match="file already exists"):
        agent.gen_tests_flow(target, Config(), write=True)

    assert existing.read_text(encoding="utf-8") == "// hand-written\n"


def test_gen_tests_go_write_fills_an_empty_existing_file(
    monkeypatch, tmp_path
):
    # Mirrors the patcher's rule exactly: only a non-empty existing
    # file blocks creation; an empty placeholder is safe to fill.
    pkg = tmp_path / "pkg" / "store"
    pkg.mkdir(parents=True)
    target = pkg / "ledger.go"
    target.write_text(LEDGER_GO, encoding="utf-8")
    out = pkg / "ledger_test.go"
    out.write_text("", encoding="utf-8")
    _install(monkeypatch, ["```go\n" + LEDGER_GO_SUITE + "\n```\n"])
    monkeypatch.chdir(tmp_path)

    code = agent.gen_tests_flow(target, Config(), write=True)

    assert code == LEDGER_GO_SUITE
    assert out.read_text(encoding="utf-8") == LEDGER_GO_SUITE + "\n"


# --------------------------------------------------------------------------
# (e) related context fences each chunk with its own file's language
# --------------------------------------------------------------------------

RELATED_PY_TEXT = (
    "def discounted_total(prices, rate):\n"
    "    return sum(prices) * rate"
)
RELATED_JS_TEXT = (
    "function total(prices) {\n"
    "  return prices.length;\n"
    "}"
)


def test_related_context_fences_chunks_by_language(monkeypatch, tmp_path):
    # A Python fix whose index-supplied related context mixes files:
    # each related block must open with its own language's fence.
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8")
    dist = _install(monkeypatch, [PY_DIFF])
    related = [
        ("pricing.py", 1, RELATED_PY_TEXT),
        ("widget.js", 4, RELATED_JS_TEXT),
    ]
    monkeypatch.setattr(
        agent, "_gather_related_context", lambda *args: related
    )

    rc = agent.fix_flow(PY_ERROR, tmp_path, Config(), apply=False)

    assert rc == 0
    user = dist.asked[0][1]["content"]
    label = "Related context (from repo index)"
    assert label in user
    section = user[user.index(label):]
    # The Python chunk still opens with the python fence...
    assert "FILE: pricing.py (lines from 1)\n```python\n" in section
    assert RELATED_PY_TEXT in section
    # ...and the JavaScript chunk with the javascript fence, not python.
    assert "FILE: widget.js (lines from 4)\n```javascript\n" in section
    assert RELATED_JS_TEXT in section
    assert "FILE: widget.js (lines from 4)\n```python" not in section


def test_format_related_context_mixed_languages():
    section = agent._format_related_context([
        ("calc.go", 7, "func Add(a, b int) int { return a - b }"),
        ("lib.rs", 2, "pub fn add(a: i32, b: i32) -> i32 { a - b }"),
        ("notes.txt", 1, "not code at all"),
    ])
    assert "FILE: calc.go (lines from 7)\n```go\n" in section
    assert "FILE: lib.rs (lines from 2)\n```rust\n" in section
    # Unknown extensions degrade to the Python fence, as the
    # selectors themselves fall back to Python.
    assert "FILE: notes.txt (lines from 1)\n```python\n" in section


def test_format_related_context_python_is_byte_identical():
    # The pre-fix formatter hardcoded ```python for every chunk; an
    # all-Python related list must produce exactly that output still.
    related = [
        ("pricing.py", 1, RELATED_PY_TEXT),
        ("calc.py", 9, "x = 1"),
    ]
    expected = "\n".join([
        "--- Related context (from repo index) ---",
        "The chunks below are from other files in this repository that "
        "the index ranked as most related to the error. They were not "
        "named in the traceback, but the root cause may live in one of "
        "them. If the fix belongs in one of these files, use its path "
        "exactly as shown.",
        "",
        "FILE: pricing.py (lines from 1)\n```python\n"
        + RELATED_PY_TEXT + "\n```",
        "FILE: calc.py (lines from 9)\n```python\nx = 1\n```",
        "Fix the root cause. Reply with the unified diff only, "
        "in a ```diff fence.",
    ])
    assert agent._format_related_context(related) == expected
