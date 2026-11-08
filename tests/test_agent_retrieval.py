"""Cross-file context retrieval for `fix`: the repo index feeds the prompt.

These tests drive `fix_flow` directly with a fake distributor whose
client can both chat (returning a canned diff) and embed (deterministic
bag-of-words vectors, like tests/test_index.py). The scenario is the
classic split bug: the crash surfaces in one file, the cause lives in
another file the traceback never names.
"""

import hashlib
import re

import pytest

from swarm_agent import agent
from swarm_agent import index as index_module
from swarm_agent.chunking import estimate_tokens
from swarm_agent.config import Config


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeSwarmClient:
    """Chat + embed in one fake; records the chat messages it receives."""

    DIM = 64

    def __init__(self, reply: str):
        self.reply = reply

    def chat(self, messages, model=None):
        return self.reply

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * FakeSwarmClient.DIM
        for tok in re.findall(r"[a-z0-9_]+", text.lower()):
            bucket = int.from_bytes(
                hashlib.md5(tok.encode()).digest()[:4], "little"
            ) % FakeSwarmClient.DIM
            vec[bucket] += 1.0
        return vec

    def embed(self, texts, model=None):
        return [self._vector(t) for t in texts]


class FakeDistributor:
    """Stands in for Distributor: `ask` chats, `client_for` embeds."""

    def __init__(self, client: FakeSwarmClient):
        self._client = client
        self.asked: list = []

    def ask(self, messages, model=None):
        self.asked.append(messages)
        return self._client.chat(messages, model=model)

    def best_peer(self):
        return "fake-peer"

    def client_for(self, peer):
        return self._client


def _run_fix(monkeypatch, repo, error_text, reply, config=None):
    dist = FakeDistributor(FakeSwarmClient(reply))
    monkeypatch.setattr(agent, "get_distributor", lambda cfg: dist)
    rc = agent.fix_flow(error_text, repo, config or Config(), apply=False)
    return rc, dist


def _user_prompt(dist: FakeDistributor) -> str:
    assert dist.asked, "the fix flow never asked the swarm"
    messages = dist.asked[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    return messages[1]["content"]


# --------------------------------------------------------------------------
# the two-file scenario: crash in orders.py, cause in pricing.py
# --------------------------------------------------------------------------

ORDERS = (
    "from pricing import discounted_total\n"
    "\n"
    "\n"
    "def checkout(prices):\n"
    "    return discounted_total(prices)\n"
)

PRICING = (
    "def discounted_total(prices, rate):\n"
    "    subtotal = sum(prices)\n"
    "    return subtotal - subtotal * rate\n"
)

ERROR = (
    "Traceback (most recent call last):\n"
    '  File "orders.py", line 5, in checkout\n'
    "    return discounted_total(prices)\n"
    "TypeError: discounted_total() missing 1 required positional argument: 'rate'\n"
)

ERROR_TWO_FRAMES = (
    "Traceback (most recent call last):\n"
    '  File "orders.py", line 5, in checkout\n'
    "    return discounted_total(prices)\n"
    '  File "pricing.py", line 2, in discounted_total\n'
    "    subtotal = sum(prices)\n"
    "TypeError: discounted_total() missing 1 required positional argument: 'rate'\n"
)

REPLY = """Here is the fix:

```diff
--- a/orders.py
+++ b/orders.py
@@ -4,2 +4,2 @@
 def checkout(prices):
-    return discounted_total(prices)
+    return discounted_total(prices, 0.1)
```
"""


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "orders.py").write_text(ORDERS, encoding="utf-8")
    (tmp_path / "pricing.py").write_text(PRICING, encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_related_context_adds_the_cause_file(monkeypatch, repo, capsys):
    rc, dist = _run_fix(monkeypatch, repo, ERROR, REPLY)
    assert rc == 0

    prompt = _user_prompt(dist)
    label = "Related context (from repo index)"
    assert label in prompt
    # The traceback file leads; the related section follows it.
    assert prompt.index("FILE: orders.py") < prompt.index(label)
    section = prompt[prompt.index(label):]
    assert "pricing.py" in section
    assert "def discounted_total(prices, rate):" in section

    # The run built the persistent index for next time.
    assert (repo / ".swarm" / "index.json").exists()
    assert "related context from index" in capsys.readouterr().out

    # A second run reuses the saved index and still gets the context.
    rc2, dist2 = _run_fix(monkeypatch, repo, ERROR, REPLY)
    assert rc2 == 0
    assert "def discounted_total(prices, rate):" in _user_prompt(dist2)


def test_index_failure_degrades_to_plain_fix(monkeypatch, repo, capsys):
    def exploding_build(*args, **kwargs):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(index_module.RepoIndex, "build", exploding_build)

    rc, dist = _run_fix(monkeypatch, repo, ERROR, REPLY)
    assert rc == 0  # the fix itself is unaffected

    prompt = _user_prompt(dist)
    assert "Related context (from repo index)" not in prompt
    assert "FILE: orders.py" in prompt  # traceback context still there
    assert "continuing without related context" in capsys.readouterr().out


def test_traceback_files_are_not_repeated_as_related(monkeypatch, repo):
    # Both files are already gathered from the traceback, so the index
    # has nothing new to offer: no related section, no duplicates.
    rc, dist = _run_fix(monkeypatch, repo, ERROR_TWO_FRAMES, REPLY)
    assert rc == 0

    prompt = _user_prompt(dist)
    assert "Related context (from repo index)" not in prompt
    assert prompt.count("def discounted_total(prices, rate):") == 1


# --------------------------------------------------------------------------
# budget: the related section shrinks before anything else does
# --------------------------------------------------------------------------

F1 = (
    "def discounted_total(prices, rate):\n"
    "    subtotal = sum(prices)\n"
    "    return subtotal - subtotal * rate"
)
F2 = (
    "def express_total(prices, rate):\n"
    "    base = sum(prices)\n"
    "    return base * rate"
)
F3 = (
    "def gift_wrap(item):\n"
    '    paper = "kraft"\n'
    '    ribbon = "gold"\n'
    "    total = paper + ribbon + item"
)
BIG_PRICING = F1 + "\n\n" + F2 + "\n\n" + F3 + "\n"


def test_related_context_shrinks_to_the_chunk_budget(monkeypatch, tmp_path):
    (tmp_path / "orders.py").write_text(ORDERS, encoding="utf-8")
    (tmp_path / "pricing.py").write_text(BIG_PRICING, encoding="utf-8")

    budget = 50
    # Fixture arithmetic: F1 + F2 fit the budget, F3 on top does not.
    assert estimate_tokens(F1) + estimate_tokens(F2) <= budget
    assert estimate_tokens(F1) + estimate_tokens(F2) + estimate_tokens(F3) > budget

    rc, dist = _run_fix(
        monkeypatch, tmp_path, ERROR, REPLY,
        config=Config(chunk_tokens=budget),
    )
    assert rc == 0

    prompt = _user_prompt(dist)
    label = "Related context (from repo index)"
    assert label in prompt
    section = prompt[prompt.index(label):]
    assert "discounted_total" in section
    assert "express_total" in section
    # The lowest-ranked chunk was dropped to fit the budget...
    assert "gift_wrap" not in prompt
    # ...while the traceback file itself is never trimmed.
    assert "return discounted_total(prices)" in prompt[:prompt.index(label)]
