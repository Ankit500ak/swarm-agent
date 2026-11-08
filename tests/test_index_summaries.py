"""Per-module summaries in the persistent repo index (issue #37).

The index accepts an optional summariser callable (dependency
injection - the index itself stays model-agnostic). The production
summariser is :func:`swarm_agent.index.chat_summariser`, the two-stage
composition the analyze flow uses: per-chunk ANALYZE_SYSTEM summaries
combined via INDEX_SUMMARY_SYSTEM, both through the real prompt
builders. These tests pin: summaries are stored, incrementality is
keyed on the same mtime/size fingerprint as the chunks, failures are
isolated per module, absence is honest (never fabricated), and
pre-summary index files load unchanged.
"""

import hashlib
import json
import re

import pytest

from swarm_agent import prompts
from swarm_agent.chunking import chunk_file
from swarm_agent.index import RepoIndex, chat_summariser


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeEmbedClient:
    """Deterministic bag-of-words embedder (as in test_index.py)."""

    DIM = 64

    def __init__(self):
        self.calls: list[list[str]] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * FakeEmbedClient.DIM
        for tok in re.findall(r"[a-z0-9_]+", text.lower()):
            bucket = int.from_bytes(
                hashlib.md5(tok.encode()).digest()[:4], "little"
            ) % FakeEmbedClient.DIM
            vec[bucket] += 1.0
        return vec

    def embed(self, texts, model=None):
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]


class RecordingChat:
    """A chat callable: canned replies per stage, records every call.

    Chunk stage (ANALYZE_SYSTEM) answers a fixed PURPOSE line; the
    combine stage (INDEX_SUMMARY_SYSTEM) answers the paragraph.
    """

    def __init__(self, paragraph="A stored summary paragraph."):
        self.calls: list[list[dict]] = []
        self.paragraph = paragraph

    def __call__(self, messages, model=None):
        self.calls.append(messages)
        if messages[0]["content"] == prompts.INDEX_SUMMARY_SYSTEM:
            return self.paragraph
        return "PURPOSE: fake chunk purpose"

    def stage_calls(self, system):
        return [c for c in self.calls if c[0]["content"] == system]


class CountingSummariser:
    """DI-level fake: records the module paths it was asked to summarise."""

    def __init__(self, text="Summary text."):
        self.calls: list[str] = []
        self.text = text

    def __call__(self, module_path, chunks):
        self.calls.append(module_path)
        return self.text


# --------------------------------------------------------------------------
# fixture (same tiny repo as test_index.py)
# --------------------------------------------------------------------------

def _write_repo(root):
    (root / "animals.py").write_text(
        "def zebra():\n    return 'striped horse of the savanna'\n",
        encoding="utf-8",
    )
    (root / "fruit.py").write_text(
        "def banana():\n    return 'yellow curved tropical snack'\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture()
def repo(tmp_path):
    return _write_repo(tmp_path)


def _build(repo, summariser, client=None):
    return RepoIndex.build(
        repo,
        client=client if client is not None else FakeEmbedClient(),
        model="fake-model",
        summariser=summariser,
    )


# --------------------------------------------------------------------------
# (a) build stores a summary per module, via the real prompt builders
# --------------------------------------------------------------------------

def test_build_stores_summaries_via_real_prompt_builders(repo):
    chat = RecordingChat()
    index = _build(repo, chat_summariser(chat))

    assert index.module_summary("animals.py") == "A stored summary paragraph."
    assert index.module_summary("fruit.py") == "A stored summary paragraph."
    assert set(index.summaries()) == {"animals.py", "fruit.py"}
    assert index.module_summary("missing.py") is None

    # Chunk stage: exactly the analyze flow's call shape, built by the
    # real build_chunk_prompt for the module's only chunk.
    chunk_calls = chat.stage_calls(prompts.ANALYZE_SYSTEM)
    assert len(chunk_calls) == 2  # one chunk per tiny module
    animals_chunk = chunk_file(repo / "animals.py")[0]
    expected_user = prompts.build_chunk_prompt(
        "animals.py",
        animals_chunk.start_line,
        animals_chunk.kind,
        animals_chunk.text,
    )
    assert any(c[1]["content"] == expected_user for c in chunk_calls)

    # Combine stage: the real build_index_summary_prompt over the
    # chunk-stage replies.
    combine_calls = chat.stage_calls(prompts.INDEX_SUMMARY_SYSTEM)
    assert len(combine_calls) == 2
    expected_combine = prompts.build_index_summary_prompt(
        "animals.py", ["PURPOSE: fake chunk purpose"]
    )
    assert any(c[1]["content"] == expected_combine for c in combine_calls)

    # And it is what actually got persisted.
    stored = json.loads(index.index_path.read_text(encoding="utf-8"))
    assert (
        stored["files"]["animals.py"]["summary"]
        == "A stored summary paragraph."
    )


# --------------------------------------------------------------------------
# (b) refresh with unchanged files does NOT regenerate
# --------------------------------------------------------------------------

def test_refresh_unchanged_does_not_regenerate(repo):
    first = CountingSummariser()
    _build(repo, first)
    assert sorted(first.calls) == ["animals.py", "fruit.py"]

    second = CountingSummariser(text="Should never be used.")
    index = _build(repo, second)
    assert second.calls == []
    assert index.module_summary("animals.py") == "Summary text."
    assert index.module_summary("fruit.py") == "Summary text."


# --------------------------------------------------------------------------
# (c) a changed file regenerates only its own summary
# --------------------------------------------------------------------------

def test_changed_file_regenerates_only_its_summary(repo):
    _build(repo, CountingSummariser(text="Original summary."))

    (repo / "fruit.py").write_text(
        "def banana():\n"
        "    return 'yellow curved tropical snack, now with extra words'\n",
        encoding="utf-8",
    )

    second = CountingSummariser(text="Regenerated summary.")
    index = _build(repo, second)
    assert second.calls == ["fruit.py"]
    assert index.module_summary("fruit.py") == "Regenerated summary."
    assert index.module_summary("animals.py") == "Original summary."


# --------------------------------------------------------------------------
# (d) no summariser: index builds fine, summaries honestly absent
# --------------------------------------------------------------------------

def test_no_summariser_build_has_no_summaries(repo):
    index = RepoIndex.build(repo, client=None, model="fake-model")
    assert index.summaries() == {}
    assert index.module_summary("animals.py") is None
    assert index.summary_failures == []
    # The index itself is fully usable.
    assert index.search("zebra savanna", k=1)[0]["path"] == "animals.py"


def test_summariser_returning_nothing_stores_nothing(repo):
    index = _build(repo, lambda path, chunks: None)
    assert index.summaries() == {}
    assert index.summary_failures == []  # absence is not a failure
    assert len(index) == 2


# --------------------------------------------------------------------------
# (e) a failing summariser leaves the other modules (and index) intact
# --------------------------------------------------------------------------

def test_failing_summariser_leaves_others_intact(repo):
    def flaky(path, chunks):
        if path == "fruit.py":
            raise RuntimeError("peer went away mid-summary")
        return "Animals summary."

    index = _build(repo, flaky)
    assert index.module_summary("animals.py") == "Animals summary."
    assert index.module_summary("fruit.py") is None
    assert index.summary_failures == ["fruit.py"]
    assert len(index) == 2
    assert index.search("zebra", k=1)[0]["path"] == "animals.py"


# --------------------------------------------------------------------------
# (f) persistence round-trip
# --------------------------------------------------------------------------

def test_summaries_survive_save_load_roundtrip(repo):
    built = _build(repo, CountingSummariser(text="Persisted summary."))
    assert built.summaries()

    loaded = RepoIndex(repo, client=FakeEmbedClient(), model="fake-model")
    assert loaded.load() is True
    assert loaded.summaries() == built.summaries()
    assert loaded.module_summary("animals.py") == "Persisted summary."


# --------------------------------------------------------------------------
# (g) a pre-summary (old-format) index file loads and backfills lazily
# --------------------------------------------------------------------------

def test_old_format_index_loads_and_backfills(repo):
    _build(repo, None)  # a build with no summariser at all

    # Rewrite the stored file the way a pre-summary build wrote it:
    # file entries without any "summary" key.
    stored = json.loads(
        (repo / ".swarm" / "index.json").read_text(encoding="utf-8")
    )
    for entry in stored["files"].values():
        entry.pop("summary", None)
    (repo / ".swarm" / "index.json").write_text(
        json.dumps(stored), encoding="utf-8"
    )

    loaded = RepoIndex(repo, client=FakeEmbedClient(), model="fake-model")
    assert loaded.load() is True  # loads without error
    assert loaded.summaries() == {}
    assert loaded.search("zebra", k=1)[0]["path"] == "animals.py"

    # A later build with a summariser backfills the missing summaries
    # without re-embedding the unchanged chunks.
    embed = FakeEmbedClient()
    backfill = CountingSummariser(text="Backfilled summary.")
    index = RepoIndex.build(
        repo, client=embed, model="fake-model", summariser=backfill
    )
    assert sorted(backfill.calls) == ["animals.py", "fruit.py"]
    assert index.summaries() == {
        "animals.py": "Backfilled summary.",
        "fruit.py": "Backfilled summary.",
    }
    embedded = [t for call in embed.calls for t in call]
    assert not any(t.startswith("def ") for t in embedded)
