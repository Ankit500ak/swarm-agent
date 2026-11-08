"""Repair loop v2 (issue #16): re-diagnosis from structured failures.

These tests drive `fix_flow` with a scripted fake swarm (a queue of
canned replies) but the *real* patcher, the *real* test runner and
the *real* `parse_failures` - so the loop is exercised exactly as it
runs in production: apply, run pytest, parse the failures, ask for a
corrected diff against the current file state, apply that, re-run.
"""

import hashlib
import re
import sys

import pytest

from swarm_agent import agent, prompts
from swarm_agent.config import Config


# --------------------------------------------------------------------------
# fakes
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


# --------------------------------------------------------------------------
# the scenario: two bugs in calc.py, the traceback names only `add`
# --------------------------------------------------------------------------

CALC = (
    "def add(a, b):\n"
    "    return a - b\n"
    "\n"
    "\n"
    "def subtract(a, b):\n"
    "    return a + b\n"
)

TESTS = (
    "from calc import add, subtract\n"
    "\n"
    "\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "\n"
    "\n"
    "def test_subtract():\n"
    "    assert subtract(5, 3) == 2\n"
)

ERROR = (
    "Traceback (most recent call last):\n"
    '  File "calc.py", line 2, in <module>\n'
    "    assert add(2, 3) == 5\n"
    "AssertionError\n"
)

TEST_CMD = f"{sys.executable} -m pytest -q"


def _reply(diff_body: str) -> str:
    return "Here is the fix:\n\n```diff\n" + diff_body + "\n```\n"


# Fixes `add` only; `subtract` keeps failing - the classic partial fix.
DIFF_FIX_ADD = _reply(
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b"
)

# Written against the post-DIFF_FIX_ADD state (subtract untouched by it).
DIFF_FIX_SUBTRACT = _reply(
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -5,2 +5,2 @@\n"
    " def subtract(a, b):\n"
    "-    return a + b\n"
    "+    return a - b"
)

# Written against the ORIGINAL calc.py: by repair time `add` already
# returns a + b, so this diff's context no longer exists on disk.
DIFF_STALE = _reply(
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b + 0"
)


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8")
    (tmp_path / "test_calc.py").write_text(TESTS, encoding="utf-8")
    return tmp_path


def _run_fix(monkeypatch, repo, replies, max_repairs="2"):
    if max_repairs is None:
        monkeypatch.delenv("SWARM_MAX_REPAIRS", raising=False)
    else:
        monkeypatch.setenv("SWARM_MAX_REPAIRS", max_repairs)
    # The scripted patches are size-preserving and land within the same
    # second, so a __pycache__ written by one inner pytest run would be
    # considered fresh by the next (mtime + size both match) and the
    # loop would verify against stale bytecode. Keep the inner runs
    # from writing bytecode at all.
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    dist = FakeDistributor(ScriptedClient(replies))
    monkeypatch.setattr(agent, "get_distributor", lambda cfg: dist)
    rc = agent.fix_flow(
        ERROR, repo, Config(), apply=True, test_cmd=TEST_CMD
    )
    return rc, dist


# --------------------------------------------------------------------------
# (a) wrong first patch, re-diagnosis fixes it on attempt 2
# --------------------------------------------------------------------------

def test_repair_loop_fixes_on_second_attempt(monkeypatch, repo, capsys):
    rc, dist = _run_fix(monkeypatch, repo, [DIFF_FIX_ADD, DIFF_FIX_SUBTRACT])

    assert rc == 0
    out = capsys.readouterr().out
    assert "fixed on attempt 2" in out
    assert len(dist.asked) == 2  # initial fix + exactly one repair

    # The repair conversation is the re-diagnosis contract, and it was
    # fed the structured failures and the previous attempt's diff.
    repair_messages = dist.asked[1]
    assert repair_messages[0]["content"] == prompts.REDIAGNOSIS_SYSTEM
    user = repair_messages[1]["content"]
    assert "This is repair attempt 2." in user
    assert "STRUCTURED FAILURES" in user
    assert "test_subtract" in user  # the parsed failing test, by name
    previous = user[user.index("PREVIOUS DIFF"):]
    assert "+    return a + b" in previous  # attempt 1's diff, verbatim
    # ...and the code blocks are the files' CURRENT (post-attempt-1) state.
    assert "FILE (current state): calc.py" in user

    final = (repo / "calc.py").read_text(encoding="utf-8")
    assert "return a + b" in final  # add fixed by attempt 1
    assert "def subtract(a, b):\n    return a - b" in final  # by attempt 2


# --------------------------------------------------------------------------
# (b) every attempt wrong: the loop gives up at the bound, honestly
# --------------------------------------------------------------------------

def test_repair_loop_gives_up_after_the_bound(monkeypatch, repo, capsys):
    wrong1 = _reply(
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b\n+    return a * b"
    )
    wrong2 = _reply(
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a * b\n+    return a - b - 1"
    )
    wrong3 = _reply(
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b - 1\n+    return a + b + 10"
    )
    rc, dist = _run_fix(monkeypatch, repo, [wrong1, wrong2, wrong3])

    assert rc == 1
    assert len(dist.asked) == 3  # 1 initial + SWARM_MAX_REPAIRS=2 repairs
    out = capsys.readouterr().out
    assert "giving up after 3 attempt(s)" in out
    for n in (1, 2, 3):
        assert f"attempt {n}:" in out  # the full history is reported
    assert "last failures:" in out
    assert "test_add" in out
    # The last (still wrong) state is what remains on disk.
    assert "return a + b + 10" in (repo / "calc.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# (c) a repair diff written against the ORIGINAL contents fails
#     verification: counted as a failed attempt, nothing is corrupted
# --------------------------------------------------------------------------

def test_stale_repair_diff_counts_and_writes_nothing(monkeypatch, repo, capsys):
    rc, dist = _run_fix(monkeypatch, repo, [DIFF_FIX_ADD, DIFF_STALE])

    assert rc == 1
    assert len(dist.asked) == 2  # the loop stops after the stale diff
    out = capsys.readouterr().out
    assert "did not apply" in out
    assert "attempt 2" in out  # ...and the attempt is in the history
    # The file is exactly the post-attempt-1 state: no force-write, no
    # partial application of the stale diff.
    expected = CALC.replace("    return a - b\n", "    return a + b\n", 1)
    assert (repo / "calc.py").read_text(encoding="utf-8") == expected


# --------------------------------------------------------------------------
# (d) max_repairs = 0 keeps the single-attempt behaviour
# --------------------------------------------------------------------------

def test_no_repairs_when_bound_is_zero(monkeypatch, repo, capsys):
    rc, dist = _run_fix(monkeypatch, repo, [DIFF_FIX_ADD], max_repairs="0")

    assert rc == 1
    assert len(dist.asked) == 1  # the swarm is never asked for a repair
    assert "repairs disabled" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the bound and the ecosystem inference, unit-level
# --------------------------------------------------------------------------

def test_max_repairs_sources(monkeypatch):
    cfg = Config()
    monkeypatch.delenv("SWARM_MAX_REPAIRS", raising=False)
    assert agent._max_repairs(cfg) == 2  # default
    monkeypatch.setenv("SWARM_MAX_REPAIRS", "3")
    assert agent._max_repairs(cfg) == 3
    monkeypatch.setenv("SWARM_MAX_REPAIRS", "junk")
    assert agent._max_repairs(cfg) == 2  # unparseable degrades to default
    monkeypatch.setenv("SWARM_MAX_REPAIRS", "-4")
    assert agent._max_repairs(cfg) == 0  # negative clamps to zero
    cfg.max_repairs = 5  # a config value, if one ever exists, wins
    assert agent._max_repairs(cfg) == 5


def test_infer_ecosystem():
    assert agent._infer_ecosystem("python -m pytest -q") == "pytest"
    assert agent._infer_ecosystem("npm test") == "jest"
    assert agent._infer_ecosystem("npx jest --watchAll=false") == "jest"
    assert agent._infer_ecosystem("make test") == ""
