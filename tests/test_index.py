import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from swarm_agent.index import TFIDF_DIM, RepoIndex
from swarm_agent.ollama import OllamaClient, SwarmError


# --------------------------------------------------------------------------
# fake embed clients
# --------------------------------------------------------------------------

class FakeEmbedClient:
    """Deterministic bag-of-words embedder; records every call.

    Vectors are token counts in hashed buckets, so a query ranks chunks
    that share its tokens first - enough semantics for ranking tests.
    """

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

    def embedded_texts(self) -> list[str]:
        return [t for call in self.calls for t in call]


class FailingEmbedClient:
    def embed(self, texts, model=None):
        raise SwarmError("peer went away")


# --------------------------------------------------------------------------
# helpers
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


# --------------------------------------------------------------------------
# OllamaClient.embed (the shared-contract addition)
# --------------------------------------------------------------------------

class _EmbedHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/api/embed":
            self.send_response(404)
            self.end_headers()
            return
        assert payload["model"] == "embed-model"
        texts = payload["input"]
        if texts == ["too", "many"]:
            vectors = [[0.1, 0.2]]  # wrong count on purpose
        else:
            vectors = [[float(len(t)), 1.0] for t in texts]
        body = json.dumps({"embeddings": vectors}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def embed_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _EmbedHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_ollama_embed_roundtrip(embed_server):
    client = OllamaClient(embed_server, default_model="embed-model")
    vectors = client.embed(["ab", "c"])
    assert vectors == [[2.0, 1.0], [1.0, 1.0]]


def test_ollama_embed_count_mismatch_raises(embed_server):
    client = OllamaClient(embed_server, default_model="embed-model")
    with pytest.raises(SwarmError):
        client.embed(["too", "many"])


# --------------------------------------------------------------------------
# building + searching with a peer
# --------------------------------------------------------------------------

def test_build_and_search_with_fake_client(repo):
    client = FakeEmbedClient()
    index = RepoIndex.build(repo, client=client, model="fake-model")
    assert index.embedder == "ollama"
    assert len(index) == 2
    assert index.index_path.exists()

    stored = json.loads(index.index_path.read_text(encoding="utf-8"))
    assert stored["embedder"] == "ollama"
    assert stored["model"] == "fake-model"
    assert set(stored["files"]) == {"animals.py", "fruit.py"}
    entry = stored["files"]["animals.py"]
    assert entry["size"] > 0 and entry["mtime"] > 0
    chunk = entry["chunks"][0]
    assert chunk["start_line"] == 1
    assert chunk["kind"] in ("function", "file", "module")
    assert "zebra" in chunk["text"]
    assert len(chunk["vector"]) == FakeEmbedClient.DIM

    results = index.search("zebra savanna", k=1)
    assert results[0]["path"] == "animals.py"
    assert results[0]["score"] > 0


def test_second_build_is_a_noop(repo):
    first = RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")
    assert len(first) == 2

    client = FakeEmbedClient()
    second = RepoIndex.build(repo, client=client, model="fake-model")
    # Nothing changed: the client is never even probed, let alone asked
    # to embed chunk text again.
    assert client.calls == []
    assert len(second) == 2
    assert second.search("banana", k=1)[0]["path"] == "fruit.py"


def test_changed_file_is_the_only_one_reembedded(repo):
    RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")

    target = repo / "fruit.py"
    target.write_text(
        "def banana():\n"
        "    return 'yellow curved tropical snack, now with extra words'\n",
        encoding="utf-8",
    )

    client = FakeEmbedClient()
    RepoIndex.build(repo, client=client, model="fake-model")
    # Ignore the one-off mode probe; look only at embedded chunk texts.
    embedded = [t for t in client.embedded_texts() if t.startswith("def ")]
    assert embedded, "the changed file should have been re-embedded"
    assert all("banana" in text for text in embedded)
    assert not any("zebra" in text for text in embedded)


def test_removed_file_drops_out(repo):
    RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")
    (repo / "fruit.py").unlink()

    index = RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")
    stored = json.loads(index.index_path.read_text(encoding="utf-8"))
    assert set(stored["files"]) == {"animals.py"}
    assert all(r["path"] != "fruit.py" for r in index.search("banana", k=5))


def test_search_ranking_puts_matching_chunk_first(repo):
    index = RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")
    results = index.search("yellow banana snack", k=2)
    assert [r["path"] for r in results] == ["fruit.py", "animals.py"]
    assert results[0]["score"] > results[1]["score"]


def test_save_load_roundtrip(repo):
    built = RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")
    expected = built.search("zebra", k=1)

    loaded = RepoIndex(repo, client=FakeEmbedClient(), model="fake-model")
    assert loaded.load() is True
    assert len(loaded) == len(built)
    assert loaded.search("zebra", k=1) == expected


def test_load_without_index_returns_false(tmp_path):
    assert RepoIndex(tmp_path).load() is False


# --------------------------------------------------------------------------
# TF-IDF fallback
# --------------------------------------------------------------------------

def test_tfidf_fallback_without_client(repo):
    index = RepoIndex.build(repo, client=None, model="fake-model")
    assert index.embedder == "tfidf"
    results = index.search("zebra savanna", k=1)
    assert results[0]["path"] == "animals.py"
    chunk = json.loads(index.index_path.read_text(encoding="utf-8"))
    vector = chunk["files"]["animals.py"]["chunks"][0]["vector"]
    assert len(vector) == TFIDF_DIM


def test_tfidf_fallback_when_embed_fails(repo):
    index = RepoIndex.build(
        repo, client=FailingEmbedClient(), model="fake-model"
    )
    assert index.embedder == "tfidf"
    assert index.search("banana", k=1)[0]["path"] == "fruit.py"


def test_tfidf_index_survives_load_and_search(repo):
    RepoIndex.build(repo, client=None, model="fake-model")
    loaded = RepoIndex(repo)  # no client: TF-IDF search needs none
    assert loaded.load() is True
    assert loaded.search("yellow banana", k=1)[0]["path"] == "fruit.py"


def test_ollama_index_without_client_cannot_search(repo):
    RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")
    loaded = RepoIndex(repo)
    assert loaded.load() is True
    with pytest.raises(SwarmError):
        loaded.search("zebra")


# --------------------------------------------------------------------------
# corrupt index handling
# --------------------------------------------------------------------------

def test_corrupt_index_rebuilds_cleanly(repo):
    index_dir = repo / ".swarm"
    index_dir.mkdir()
    (index_dir / "index.json").write_text("{ not json at all", encoding="utf-8")

    index = RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")
    assert len(index) == 2
    assert index.search("zebra", k=1)[0]["path"] == "animals.py"


def test_wrong_version_index_rebuilds_cleanly(repo):
    index_dir = repo / ".swarm"
    index_dir.mkdir()
    (index_dir / "index.json").write_text(
        json.dumps({"version": 999, "files": {}}), encoding="utf-8"
    )

    index = RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")
    assert len(index) == 2


def test_index_directory_is_not_indexed(repo):
    index = RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")
    stored = json.loads(index.index_path.read_text(encoding="utf-8"))
    assert not any(p.startswith(".swarm") for p in stored["files"])
    # And a rebuild does not ingest the index file it just wrote.
    again = RepoIndex.build(repo, client=FakeEmbedClient(), model="fake-model")
    assert len(again) == len(index)
