"""A persistent, searchable index of a repository's chunks.

``analyze`` re-reads and re-chunks a repository on every run. The index
exists so the *embeddings* of those chunks survive between runs: they are
computed once, stored in ``<repo_root>/.swarm/index.json``, and reused
until a file changes. Rebuilds are incremental - a file is only
re-chunked and re-embedded when its ``mtime`` or ``size`` differs from
what the index recorded, and files that disappear drop out.

Embeddings come from an Ollama peer (``POST /api/embed``) when one is
reachable. When no peer answers - or embedding fails mid-build - the
index falls back to a small local TF-IDF embedder, so it still works
fully offline. The fallback hashes tokens into a fixed
:data:`TFIDF_DIM`-dimensional space (hashing keeps the vocabulary
unbounded and the stored statistics tiny: one document-frequency count
per column). Two simplifications follow, both deliberate:

- Hash collisions merge a few unrelated tokens into one column. At
  this dimension that is rare enough to ignore for code search.
- TF-IDF vectors are recomputed for the *whole* corpus whenever it
  changes, because idf is a corpus statistic and every vector in the
  index must live in the same space. This is local arithmetic - the
  incrementality that matters (network calls to a peer) still only
  ever touches changed files, and in TF-IDF mode there are no network
  calls at all.

An index's vectors are only comparable within the space that produced
them, so the file records its embedder kind and model; a build that
finds a different space (or a different model) discards the stored
vectors and re-embeds. A missing or corrupt index file is never an
error - it simply rebuilds from scratch.

Each file entry may also carry a ``summary``: one paragraph describing
the module, generated at build time by an optional *summariser*
callable injected by the caller (see :func:`chat_summariser`). The
index itself stays model-agnostic - it only knows the callable's
shape, ``summariser(module_path, chunks) -> str | None``. Summaries
are strictly optional: with no summariser the index is exactly what it
was before they existed, a summariser failure leaves that one module
without a summary (never a fabricated one), and a module's summary is
regenerated only when the module's content fingerprint changes - a
summary stored by an older build is carried verbatim while its file
is unchanged. The field is an additive, optional key in the existing
version-1 format, so index files written before summaries existed
load unchanged, their modules simply having no summary yet.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Callable, Optional

from . import prompts
from .agent import BINARY_EXTS, MAX_FILE_BYTES, SKIP_DIRS
from .chunking import chunk_file
from .config import DEFAULT_MODEL
from .ollama import SwarmError

INDEX_DIRNAME = ".swarm"
INDEX_FILENAME = "index.json"
INDEX_VERSION = 1

#: Dimension of the hashed TF-IDF space used by the offline fallback.
TFIDF_DIM = 1024

#: Texts per ``/api/embed`` call when embedding through a peer.
EMBED_BATCH = 64

_PROBE_TEXT = "swarm-agent index probe"
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
# The index directory itself is never indexed.
_SKIP_DIRS = SKIP_DIRS | {INDEX_DIRNAME}


# --------------------------------------------------------------------------
# vector helpers
# --------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _column(token: str) -> int:
    """Deterministic hash bucket for a token (never ``hash()``: it is
    salted per process, which would corrupt a persisted index)."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % TFIDF_DIM


class _TfidfSpace:
    """Corpus statistics for the hashed TF-IDF fallback embedder."""

    def __init__(self, num_docs: int = 0, df: Optional[list[int]] = None) -> None:
        self.num_docs = num_docs
        self.df = df if df is not None else [0] * TFIDF_DIM

    @classmethod
    def fit(cls, texts: list[str]) -> "_TfidfSpace":
        space = cls(num_docs=len(texts))
        for text in texts:
            for col in {_column(tok) for tok in _tokens(text)}:
                space.df[col] += 1
        return space

    def _idf(self, col: int) -> float:
        return math.log((1 + self.num_docs) / (1 + self.df[col])) + 1.0

    def vector(self, text: str) -> list[float]:
        vec = [0.0] * TFIDF_DIM
        counts: dict[int, int] = {}
        for tok in _tokens(text):
            col = _column(tok)
            counts[col] = counts.get(col, 0) + 1
        for col, count in counts.items():
            vec[col] = (1.0 + math.log(count)) * self._idf(col)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm:
            vec = [x / norm for x in vec]
        return vec

    def to_json(self) -> dict:
        return {"num_docs": self.num_docs, "df": self.df}

    @classmethod
    def from_json(cls, data: object) -> Optional["_TfidfSpace"]:
        if not isinstance(data, dict):
            return None
        df = data.get("df")
        num_docs = data.get("num_docs")
        if (
            not isinstance(df, list)
            or len(df) != TFIDF_DIM
            or not isinstance(num_docs, int)
        ):
            return None
        return cls(num_docs=num_docs, df=[int(x) for x in df])


# --------------------------------------------------------------------------
# repository walking (same rules as analyze, plus .swarm itself)
# --------------------------------------------------------------------------

def _iter_indexable_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in BINARY_EXTS:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                path.read_text(encoding="utf-8")  # probe: must decode as text
            except (OSError, UnicodeDecodeError):
                continue
            yield path


# --------------------------------------------------------------------------
# module summaries (optional, injected - the index stays model-agnostic)
# --------------------------------------------------------------------------

#: A module summariser: given a module's repo-relative path and its
#: chunk records (``{"start_line", "kind", "text", ...}``), return the
#: one-paragraph summary to store, or None to store nothing.
Summariser = Callable[[str, list], Optional[str]]


def chat_summariser(chat: Callable) -> Summariser:
    """A summariser backed by a ``chat(messages) -> str`` callable.

    This is the two-stage composition the analyze flow uses, per
    module: each chunk is summarised with
    :data:`prompts.ANALYZE_SYSTEM` (PURPOSE / KEY / RISKS, via
    :func:`prompts.build_chunk_prompt`), then the chunk summaries are
    combined into the single stored paragraph with
    :data:`prompts.INDEX_SUMMARY_SYSTEM` (via
    :func:`prompts.build_index_summary_prompt`). Any exception from
    ``chat`` propagates - the index treats it as "this module gets no
    summary"; an empty final reply likewise stores nothing.
    """

    def summarise(module_path: str, chunks: list) -> Optional[str]:
        chunk_summaries = []
        for chunk in chunks:
            reply = chat([
                {"role": "system", "content": prompts.ANALYZE_SYSTEM},
                {"role": "user", "content": prompts.build_chunk_prompt(
                    module_path,
                    chunk["start_line"],
                    chunk["kind"],
                    chunk["text"],
                )},
            ])
            chunk_summaries.append(reply.strip())
        reply = chat([
            {"role": "system", "content": prompts.INDEX_SUMMARY_SYSTEM},
            {"role": "user", "content": prompts.build_index_summary_prompt(
                module_path, chunk_summaries
            )},
        ])
        return reply.strip() or None

    return summarise


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------

class RepoIndex:
    """Chunk embeddings for one repository, persisted under ``.swarm/``."""

    def __init__(
        self,
        repo_root: str | Path,
        client=None,
        model: Optional[str] = None,
        budget_tokens: int = 1500,
        summariser: Optional[Summariser] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.client = client
        self.model = model or DEFAULT_MODEL
        self.budget_tokens = budget_tokens
        self.summariser = summariser
        # "ollama" | "tfidf" | None (nothing built or loaded yet)
        self.embedder: Optional[str] = None
        self._tfidf: Optional[_TfidfSpace] = None
        # relpaths whose summary generation failed in the last build
        self.summary_failures: list[str] = []
        # relpath -> {"mtime": float, "size": int, "chunks": [record, ...],
        #             "summary": str | None}
        # record = {"start_line", "kind", "text", "vector"}
        self._files: dict[str, dict] = {}

    # -- paths -------------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.repo_root / INDEX_DIRNAME / INDEX_FILENAME

    def __len__(self) -> int:
        return sum(len(entry["chunks"]) for entry in self._files.values())

    # -- persistence ---------------------------------------------------------
    def save(self) -> None:
        """Write the index to disk (atomically, via a temp file)."""
        payload = {
            "version": INDEX_VERSION,
            "model": self.model,
            "embedder": self.embedder,
            "tfidf": self._tfidf.to_json() if self._tfidf else None,
            "files": self._files,
        }
        path = self.index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

    def _read_stored(self) -> Optional[dict]:
        """The persisted payload, or None if missing/corrupt/foreign."""
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("version") != INDEX_VERSION:
            return None
        if not isinstance(data.get("files"), dict):
            return None
        return data

    def load(self) -> bool:
        """Load a previously saved index. False if there isn't a usable one."""
        stored = self._read_stored()
        if stored is None or stored.get("embedder") not in ("ollama", "tfidf"):
            return False
        self.model = stored.get("model") or self.model
        self.embedder = stored["embedder"]
        self._tfidf = _TfidfSpace.from_json(stored.get("tfidf"))
        self._files = stored["files"]
        return True

    # -- building ------------------------------------------------------------
    @classmethod
    def build(
        cls,
        repo_root: str | Path,
        client=None,
        model: Optional[str] = None,
        budget_tokens: int = 1500,
        summariser: Optional[Summariser] = None,
    ) -> "RepoIndex":
        """Build (or incrementally refresh) the index for ``repo_root``.

        Only files whose ``mtime``/``size`` changed are re-chunked and
        re-embedded; the refreshed index is saved before returning.
        When ``summariser`` is given, modules that lack a summary (new,
        changed, or never summarised) get one generated; modules whose
        content is unchanged keep their stored summary untouched.
        """
        index = cls(
            repo_root,
            client=client,
            model=model,
            budget_tokens=budget_tokens,
            summariser=summariser,
        )
        index._build()
        index.save()
        return index

    def _scan(self) -> dict[str, tuple[Path, float, int]]:
        """relpath -> (absolute path, mtime, size) for every indexable file."""
        found: dict[str, tuple[Path, float, int]] = {}
        for path in _iter_indexable_files(self.repo_root):
            try:
                stat = path.stat()
            except OSError:
                continue
            rel = path.relative_to(self.repo_root).as_posix()
            found[rel] = (path, stat.st_mtime, stat.st_size)
        return found

    def _probe_mode(self) -> str:
        """'ollama' if the client can embed right now, else 'tfidf'."""
        if self.client is None:
            return "tfidf"
        try:
            vectors = self.client.embed([_PROBE_TEXT], model=self.model)
        except Exception:
            return "tfidf"
        return "ollama" if vectors and vectors[0] else "tfidf"

    def _build(self) -> None:
        stored = self._read_stored()
        stored_files: dict = stored["files"] if stored else {}
        stored_mode = stored.get("embedder") if stored else None
        same_space = (
            stored is not None
            and stored_mode in ("ollama", "tfidf")
            and stored.get("model") == self.model
        )

        # Assemble chunk records: reuse what is still valid, re-chunk the
        # rest. Stored vectors survive only if the embedding space does.
        # A stored summary survives whenever the file itself is unchanged
        # (it describes the text, not the vectors); a changed file's old
        # summary is dropped here and regenerated below if possible.
        files: dict[str, dict] = {}
        pending: list[dict] = []  # chunk records still needing a vector
        for rel, (abspath, mtime, size) in sorted(self._scan().items()):
            old = stored_files.get(rel)
            chunks: Optional[list[dict]] = None
            summary: Optional[str] = None
            if old and old.get("mtime") == mtime and old.get("size") == size:
                chunks = [
                    {
                        "start_line": c["start_line"],
                        "kind": c["kind"],
                        "text": c["text"],
                        "vector": list(c["vector"]) if same_space else None,
                    }
                    for c in old.get("chunks", [])
                ]
                # Absent on pre-summary index files: treated as missing.
                stored_summary = old.get("summary")
                if isinstance(stored_summary, str) and stored_summary:
                    summary = stored_summary
            if chunks is None:
                try:
                    fresh = chunk_file(abspath, budget_tokens=self.budget_tokens)
                except (OSError, ValueError):
                    continue
                chunks = [
                    {
                        "start_line": c.start_line,
                        "kind": c.kind,
                        "text": c.text,
                        "vector": None,
                    }
                    for c in fresh
                ]
            files[rel] = {
                "mtime": mtime,
                "size": size,
                "chunks": chunks,
                "summary": summary,
            }
            pending.extend(c for c in chunks if c["vector"] is None)
        self._files = files

        # Modules that will need a summary generated in this build:
        # anything whose entry has none (new, changed, or stored by a
        # build that had no summariser). Unchanged modules with a
        # stored summary are never in this list.
        needs_summary = (
            [rel for rel, entry in files.items() if entry["summary"] is None]
            if self.summariser is not None
            else []
        )

        if not pending and same_space:
            # Pure reuse - nothing to embed, keep the stored space as-is.
            self.embedder = stored_mode
            self._tfidf = _TfidfSpace.from_json(stored.get("tfidf"))
        else:
            mode = self._probe_mode()
            if stored_mode is not None and mode != stored_mode:
                # The space changed under us: every stored vector is invalid.
                pending = [
                    c for entry in files.values() for c in entry["chunks"]
                ]
                for chunk in pending:
                    chunk["vector"] = None

            if mode == "ollama" and pending:
                try:
                    self._embed_via_client(pending)
                except Exception:
                    mode = "tfidf"  # peer failed mid-build: fall back locally
                    pending = [
                        c for entry in files.values() for c in entry["chunks"]
                    ]
            if mode == "tfidf":
                self._embed_via_tfidf()
            self.embedder = mode

        if needs_summary:
            self._generate_summaries(needs_summary)

    def _generate_summaries(self, relpaths: list[str]) -> None:
        """Fill in missing module summaries via the injected summariser.

        Per-module failure isolation: a module whose summariser raises
        (or returns nothing) keeps no summary and is recorded in
        :attr:`summary_failures`; the rest of the index is unaffected.
        """
        assert self.summariser is not None  # guarded by the caller
        for rel in relpaths:
            entry = self._files[rel]
            try:
                summary = self.summariser(rel, entry["chunks"])
            except Exception:
                self.summary_failures.append(rel)
                continue
            if isinstance(summary, str) and summary.strip():
                entry["summary"] = summary.strip()

    def _embed_via_client(self, pending: list[dict]) -> None:
        for i in range(0, len(pending), EMBED_BATCH):
            batch = pending[i : i + EMBED_BATCH]
            vectors = self.client.embed(
                [c["text"] for c in batch], model=self.model
            )
            if len(vectors) != len(batch):
                raise SwarmError(
                    f"embed returned {len(vectors)} vectors for "
                    f"{len(batch)} texts"
                )
            for chunk, vector in zip(batch, vectors):
                chunk["vector"] = [float(x) for x in vector]

    def _embed_via_tfidf(self) -> None:
        texts = [
            c["text"] for entry in self._files.values() for c in entry["chunks"]
        ]
        self._tfidf = _TfidfSpace.fit(texts)
        for entry in self._files.values():
            for chunk in entry["chunks"]:
                chunk["vector"] = self._tfidf.vector(chunk["text"])

    # -- searching -------------------------------------------------------------
    def _query_vector(self, query: str) -> list[float]:
        if self.embedder == "ollama":
            if self.client is None:
                raise SwarmError(
                    "this index was built with Ollama embeddings; "
                    "a client is required to search it"
                )
            vectors = self.client.embed([query], model=self.model)
            if not vectors:
                raise SwarmError("embed returned no vector for the query")
            return [float(x) for x in vectors[0]]
        if self.embedder == "tfidf" and self._tfidf is not None:
            return self._tfidf.vector(query)
        raise SwarmError("index has not been built or loaded yet")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """The ``k`` chunks closest to ``query``, best first.

        Each result is ``{"path", "start_line", "text", "score"}`` with
        ``path`` relative to the repository root.
        """
        if k <= 0 or not self._files:
            return []
        qvec = self._query_vector(query)
        scored: list[tuple[float, str, int, str]] = []
        for rel, entry in self._files.items():
            for chunk in entry["chunks"]:
                score = _cosine(qvec, chunk["vector"])
                scored.append((score, rel, chunk["start_line"], chunk["text"]))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            {"path": rel, "start_line": start, "text": text, "score": score}
            for score, rel, start, text in scored[:k]
        ]

    # -- summaries -----------------------------------------------------------
    def module_summary(self, path: str | Path) -> Optional[str]:
        """The stored summary paragraph for one module, or None.

        ``path`` is the module's repo-relative path (the same key the
        index stores, e.g. ``"pkg/mod.py"``). Modules that were never
        summarised - no summariser at build time, generation failed,
        or the file is not in the index - simply return None.
        """
        rel = path.as_posix() if isinstance(path, Path) else str(path)
        entry = self._files.get(rel)
        if not entry:
            return None
        summary = entry.get("summary")
        return summary if isinstance(summary, str) and summary else None

    def summaries(self) -> dict[str, str]:
        """All stored module summaries, ``{relpath: paragraph}``."""
        return {
            rel: entry["summary"]
            for rel, entry in sorted(self._files.items())
            if isinstance(entry.get("summary"), str) and entry["summary"]
        }
