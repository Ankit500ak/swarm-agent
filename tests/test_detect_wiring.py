"""Test-command wiring (issue #31): the verify steps resolve their
command through `agent._verify_command`, which wires
`runner.detect_test_command` into the flows.

Precedence under test (documented on `_verify_command`):

1. an explicit per-run ``test_cmd`` wins verbatim;
2. a *customised* ``config.test_command`` (anything but the built-in
   default - including ``None``/``""``, verification off) beats
   detection;
3. detection picks from the repo's markers, falling back to the
   default pytest command - so a plain Python repo verifies with
   exactly the command it used before detection existed.

The fix flow resolves the command once and hands it to the repair
loop, so every verify run in a fix uses the same command.
"""

import hashlib
import json
import re
import sys

from swarm_agent import agent, prompts
from swarm_agent.config import Config


# --------------------------------------------------------------------------
# fakes (same shape as test_language_wiring's: scripted chat + fake embed)
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


class IndexStubClient:
    """The index-side client handed out by `client_for`.

    Embeds bag-of-words (same vectors as ScriptedClient, so index
    ranking is unchanged) and answers any chat - i.e. index summary
    generation - with a constant stub paragraph, so building the repo
    index never consumes the scripted conversation replies that
    `ask` depends on.
    """

    def chat(self, messages, model=None):
        return "PURPOSE: stub summary"

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
        return IndexStubClient()


def _install(monkeypatch, replies) -> FakeDistributor:
    dist = FakeDistributor(ScriptedClient(replies))
    monkeypatch.setattr(agent, "get_distributor", lambda cfg: dist)
    return dist


def _capture_runs(monkeypatch, results=None):
    """Replace agent.run_tests with a recorder.

    Returns the list of (command, cwd) it was called with. ``results``
    scripts the (exit_code, tail) replies; default is a single pass,
    repeated for any further calls.
    """
    calls: list = []
    scripted = list(results) if results else [(0, "all green")]

    def fake_run_tests(cmd, root):
        calls.append((cmd, root))
        if len(scripted) > 1:
            return scripted.pop(0)
        return scripted[0]

    monkeypatch.setattr(agent, "run_tests", fake_run_tests)
    return calls


def _reply(diff_body: str) -> str:
    return "Here is the fix:\n\n```diff\n" + diff_body + "\n```\n"


# --------------------------------------------------------------------------
# fixtures: one small buggy repo per ecosystem
# --------------------------------------------------------------------------

APP_JS = "function add(a, b) {\n  return a - b;\n}\n"

JS_ERROR = (
    "Error: boom\n"
    '  File "app.js", line 2, in add\n'
    "    return a - b;\n"
)

JS_DIFF = _reply(
    "--- a/app.js\n"
    "+++ b/app.js\n"
    "@@ -1,2 +1,2 @@\n"
    " function add(a, b) {\n"
    "-  return a - b;\n"
    "+  return a + b;"
)

CALC_GO = "package calc\nfunc Add(a, b int) int { return a - b }\n"

GO_ERROR = (
    "panic: boom\n"
    '  File "calc.go", line 2, in Add\n'
)

GO_DIFF = _reply(
    "--- a/calc.go\n"
    "+++ b/calc.go\n"
    "@@ -1,2 +1,2 @@\n"
    " package calc\n"
    "-func Add(a, b int) int { return a - b }\n"
    "+func Add(a, b int) int { return a + b }"
)

CALC_PY = "def add(a, b):\n    return a - b\n"

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


def _js_repo(tmp_path):
    (tmp_path / "app.js").write_text(APP_JS, encoding="utf-8")
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "app", "scripts": {"test": "jest"}}),
        encoding="utf-8",
    )
    return tmp_path


def _go_repo(tmp_path):
    (tmp_path / "calc.go").write_text(CALC_GO, encoding="utf-8")
    (tmp_path / "go.mod").write_text("module calc\n\ngo 1.22\n", encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------
# (a) + (b) detection reaches the fix flow's verify step
# --------------------------------------------------------------------------

def test_fix_in_js_repo_verifies_with_npm_test(monkeypatch, tmp_path):
    repo = _js_repo(tmp_path)
    _install(monkeypatch, [JS_DIFF])
    runs = _capture_runs(monkeypatch)

    rc = agent.fix_flow(JS_ERROR, repo, Config(), apply=True)

    assert rc == 0
    assert runs == [("npm test", repo)]


def test_fix_in_go_repo_verifies_with_go_test(monkeypatch, tmp_path):
    repo = _go_repo(tmp_path)
    _install(monkeypatch, [GO_DIFF])
    runs = _capture_runs(monkeypatch)

    rc = agent.fix_flow(GO_ERROR, repo, Config(), apply=True)

    assert rc == 0
    assert runs == [("go test ./...", repo)]


# --------------------------------------------------------------------------
# the repair loop reuses the detected command for every verify run
# --------------------------------------------------------------------------

APP_JS_TWO_BUGS = (
    "function add(a, b) {\n  return a - b;\n}\n"
    "\n"
    "function sub(a, b) {\n  return a + b;\n}\n"
)

JS_DIFF_2 = _reply(
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


def test_repair_loop_verifies_with_the_detected_command(
    monkeypatch, tmp_path
):
    (tmp_path / "app.js").write_text(APP_JS_TWO_BUGS, encoding="utf-8")
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "app", "scripts": {"test": "jest"}}),
        encoding="utf-8",
    )
    _install(monkeypatch, [JS_DIFF, JS_DIFF_2])
    runs = _capture_runs(
        monkeypatch, results=[(1, JEST_TAIL), (0, "all green")]
    )

    rc = agent.fix_flow(JS_ERROR, tmp_path, Config(), apply=True)

    assert rc == 0
    # Initial verify + the repair's verify, both detected - the loop
    # never falls back to a hardcoded pytest.
    assert [cmd for cmd, _ in runs] == ["npm test", "npm test"]


# --------------------------------------------------------------------------
# (c) an explicit test_cmd beats the repo's markers
# --------------------------------------------------------------------------

def test_fix_explicit_test_cmd_beats_markers(monkeypatch, tmp_path):
    repo = _js_repo(tmp_path)
    _install(monkeypatch, [JS_DIFF])
    runs = _capture_runs(monkeypatch)

    rc = agent.fix_flow(
        JS_ERROR, repo, Config(), apply=True, test_cmd="make check"
    )

    assert rc == 0
    assert runs == [("make check", repo)]


# --------------------------------------------------------------------------
# (d) a plain Python repo gets exactly the pre-detection command
# --------------------------------------------------------------------------

def test_fix_plain_python_repo_verifies_with_pytest_default(
    monkeypatch, tmp_path
):
    (tmp_path / "calc.py").write_text(CALC_PY, encoding="utf-8")
    _install(monkeypatch, [PY_DIFF])
    runs = _capture_runs(monkeypatch)

    rc = agent.fix_flow(PY_ERROR, tmp_path, Config(), apply=True)

    assert rc == 0
    assert runs == [("python -m pytest -q", tmp_path)]


# --------------------------------------------------------------------------
# (e) a customised config.test_command beats detection
# --------------------------------------------------------------------------

def test_fix_customised_config_beats_detection(monkeypatch, tmp_path):
    repo = _js_repo(tmp_path)
    _install(monkeypatch, [JS_DIFF])
    runs = _capture_runs(monkeypatch)

    rc = agent.fix_flow(
        JS_ERROR, repo, Config(test_command="make verify"), apply=True
    )

    assert rc == 0
    assert runs == [("make verify", repo)]


def test_fix_config_none_disables_verification(monkeypatch, tmp_path):
    repo = _js_repo(tmp_path)
    _install(monkeypatch, [JS_DIFF])
    runs = _capture_runs(monkeypatch)

    # Verification was switched off in the config: detection must not
    # resurrect it just because the repo has markers.
    rc = agent.fix_flow(
        JS_ERROR, repo, Config(test_command=None), apply=True
    )

    assert rc == 0
    assert runs == []


# --------------------------------------------------------------------------
# the precedence, pinned at the helper
# --------------------------------------------------------------------------

def test_verify_command_precedence(tmp_path):
    js_dir = tmp_path / "js"
    js_dir.mkdir()
    js_root = _js_repo(js_dir)
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    # 1. explicit wins over markers and over a customised config.
    assert agent._verify_command(js_root, Config(), "make check") == "make check"
    assert agent._verify_command(
        js_root, Config(test_command="make verify"), "make check"
    ) == "make check"
    # 2. a customised config beats detection...
    assert agent._verify_command(
        js_root, Config(test_command="make verify")
    ) == "make verify"
    #    ...including "off": None/"" are returned, not detected past.
    assert agent._verify_command(js_root, Config(test_command=None)) is None
    assert agent._verify_command(js_root, Config(test_command="")) == ""
    # 3. with the default config, detection decides.
    assert agent._verify_command(js_root, Config()) == "npm test"
    assert agent._verify_command(empty_root, Config()) == "python -m pytest -q"
    # A config value equal to the default counts as uncustomised.
    assert agent._verify_command(
        js_root, Config(test_command="python -m pytest -q")
    ) == "npm test"


# --------------------------------------------------------------------------
# gen-tests: run=True uses the detected command for non-Python targets,
# and keeps the exact historical command for Python ones
# --------------------------------------------------------------------------

GO_SUITE = (
    "package calc\n\nimport \"testing\"\n\n"
    "func TestAdd(t *testing.T) {\n"
    "\tif Add(2, 3) != 5 {\n"
    "\t\tt.Fatal(\"bad add\")\n"
    "\t}\n"
    "}"
)

PY_SUITE = "def test_add():\n    assert add(1, 2) == 3"


def test_gen_tests_run_uses_detected_command_for_go(
    monkeypatch, tmp_path
):
    repo = _go_repo(tmp_path)
    target = repo / "calc.go"
    _install(monkeypatch, ["```go\n" + GO_SUITE + "\n```\n"])
    runs = _capture_runs(monkeypatch)
    monkeypatch.chdir(repo)

    code = agent.gen_tests_flow(target, Config(), write=True, run=True)

    assert code == GO_SUITE
    assert runs == [("go test ./...", repo)]


def test_gen_tests_run_for_python_is_unchanged(monkeypatch, tmp_path):
    target = tmp_path / "calc.py"
    target.write_text(CALC_PY, encoding="utf-8")
    _install(monkeypatch, ["```python\n" + PY_SUITE + "\n```\n"])
    runs = _capture_runs(monkeypatch)
    monkeypatch.chdir(tmp_path)

    code = agent.gen_tests_flow(target, Config(), write=True, run=True)

    assert code == PY_SUITE
    out = tmp_path / "tests" / "test_calc_generated.py"
    assert runs == [(f"{sys.executable} -m pytest {out} -q", tmp_path)]
