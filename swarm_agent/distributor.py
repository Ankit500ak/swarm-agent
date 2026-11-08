"""Distributing work across the swarm.

The distributor owns the peer list and the failure policy:

- Peers that serve the requested model are preferred — first under the
  exact name, then under a spelling :func:`models_match` recognises as
  the same model (an MLX server's ``mlx-community/…-4bit`` name for
  Ollama's ``…:7b``); if none do, any healthy peer is used (its client
  resolves to a model it actually has).
- :meth:`Distributor.map_units` schedules *latency-weighted*: each peer
  carries an exponentially-weighted moving average (EWMA) of how long
  its units actually take, seeded from the latency discovery measured.
  Whenever a unit is handed out, :func:`pick_peer` gives it to the peer
  with the lowest predicted wait — ``in-flight units x EWMA`` — so
  faster peers naturally cycle through more units than slower ones,
  and the balance shifts mid-run as real timings replace the seeds.
- A peer whose work raises is marked *down*; the failed unit goes back
  on the pending queue and is reassigned to the surviving peers. A unit
  that has failed on every peer ends up holding its exception as its
  result, so callers can report it instead of crashing.
- Results always come back in input order, and ``counts`` records how
  many units each peer completed.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .discovery import Peer
from .ollama import SwarmError, client_for_peer

# EWMA smoothing factor for per-peer unit durations: high enough to react
# within a handful of units, low enough that one slow call is not destiny.
_EWMA_ALPHA = 0.3
# Floor so a peer that answered in microseconds still has a positive
# predicted wait while it is busy (an idle peer's wait is exactly 0).
_MIN_EWMA_MS = 0.01
# Seed for peers discovery could not time (latency_ms == 0): deliberately
# pessimistic, so measured peers win the first picks and this peer's own
# first timings quickly replace the guess.
_DEFAULT_EWMA_MS = 1000.0

# -- model-name matching -------------------------------------------------------
#
# Peers report model names in whatever spelling their server uses: Ollama
# says ``qwen2.5-coder:7b`` while an MLX server says
# ``mlx-community/Qwen2.5-Coder-7B-Instruct-4bit``. Those are the same
# model, so exact string equality would never select the MLX peer under
# the default configuration (issue #47). ``model_key`` reduces a name to
# a comparable ``(base, size)`` pair; ``models_match`` compares keys.

# A hyphen token stating the parameter count: 7b, 14b, 0.5b, 135m, 8x7b.
_SIZE_RE = re.compile(r"^(?:\d+x)?\d+(?:\.\d+)?[bm]$")
# A hyphen token stating only how the weights are packed: 4bit, 8bit,
# fp16, bf16, f16/f32, GGUF quant names (q4_K_M, q8_0, iq4_xs, ...).
_QUANT_RE = re.compile(r"^(?:\d+bit|fp\d+|bf\d+|f16|f32|i?q\d+.*)$")
_FORMAT_TOKENS = frozenset({"awq", "gptq", "gguf", "safetensors", "exl2"})
# Fine-tune flavour tokens: they say how a model was tuned, not which
# model it is, and the two spellings of the same model disagree on them
# (Ollama's ``:7b`` tag implies the instruct build without saying so).
_FLAVOUR_TOKENS = frozenset({"instruct", "chat"})


def model_key(name: Optional[str]) -> Optional[tuple[str, Optional[str]]]:
    """Reduce a model name to a comparable ``(base, size)`` key, or None.

    Normalisation, applied identically to the requested model and to
    every name a peer reports:

    - case is folded and any registry/namespace prefix is dropped
      (everything up to the last ``/``, e.g. ``mlx-community/``);
    - an Ollama ``:tag`` is split off and read like the rest of the name
      (``:7b``, ``:7b-instruct-q4_K_M``); a bare ``:latest`` tag carries
      no information and is ignored;
    - tokens stating the parameter size (``7b``, ``14b``, ``0.5b``,
      ``8x7b`` ...) are pulled out as ``size``, wherever they appear;
    - quantisation/format tokens (``4bit``, ``fp16``, ``q4_K_M``,
      ``gguf`` ...) and the flavour tokens ``instruct``/``chat`` are
      dropped;
    - what remains, rejoined with ``-``, is the ``base``: family and
      version survive intact, so ``qwen2.5-coder`` never keys equal to
      ``qwen3-coder`` or ``llama3.1``.

    Returns ``None`` when nothing recognisable remains (an empty name,
    or one made only of size/quant tokens): such names are unparseable
    and callers fall back to exact string equality.
    """
    if not name or not isinstance(name, str):
        return None
    text = name.strip().lower().rsplit("/", 1)[-1]
    if not text:
        return None
    head, sep, tag = text.partition(":")
    tag_tokens = tag.split("-") if sep and tag != "latest" else []
    size: Optional[str] = None
    base_tokens: list[str] = []
    for token in head.split("-") + tag_tokens:
        if not token:
            continue
        if _SIZE_RE.match(token):
            if size is None:
                size = token
        elif (
            token in _FLAVOUR_TOKENS
            or token in _FORMAT_TOKENS
            or _QUANT_RE.match(token)
        ):
            continue
        else:
            base_tokens.append(token)
    if not base_tokens:
        return None
    return "-".join(base_tokens), size


def models_match(requested: str, available: str) -> bool:
    """Whether a peer serving ``available`` can satisfy ``requested``.

    True when the names are exactly equal (always — even for names the
    normaliser cannot parse), or when both parse to the same key: equal
    base, and sizes that agree whenever *both* sides state one. A 14B
    peer therefore never satisfies a 7B request, while a request that
    names no size accepts any size of the right model.

    This is a heuristic over naming conventions, not proof of identity:
    quantisation is ignored (a peer serving a 4-bit build matches) and
    ``-instruct`` versus base builds of the same model and size are
    treated as interchangeable. Underscore-joined name variants are not
    normalised. When the distinction matters, use the exact name — the
    distributor always prefers exact matches over normalised ones.
    """
    if requested == available:
        return True
    req = model_key(requested)
    avl = model_key(available)
    if req is None or avl is None:
        return False
    if req[0] != avl[0]:
        return False
    return req[1] is None or avl[1] is None or req[1] == avl[1]


@dataclass
class PeerStats:
    """Live scheduling state for one peer.

    ``ewma_ms`` is the moving average of completed-unit durations;
    ``inflight`` counts units handed to this peer and not yet finished.
    Kept as plain data so :func:`pick_peer` stays a pure function of it.
    """

    peer: Peer
    ewma_ms: float
    inflight: int = 0
    down: bool = False


def _seed_ewma(peer: Peer) -> float:
    return peer.latency_ms if peer.latency_ms > 0 else _DEFAULT_EWMA_MS


def _ewma_update(current: float, sample_ms: float) -> float:
    updated = _EWMA_ALPHA * sample_ms + (1.0 - _EWMA_ALPHA) * current
    return max(updated, _MIN_EWMA_MS)


def pick_peer(stats: Iterable[PeerStats]) -> Optional[Peer]:
    """Choose which peer gets the next unit. Pure and deterministic.

    Among peers that are not down, the winner has the lowest predicted
    wait (``inflight x ewma_ms``); ties break toward the lower EWMA,
    then toward the lexicographically smaller base URL, so the same
    stats always produce the same pick. Returns ``None`` when there is
    nothing to pick (empty input, or every peer down).
    """
    candidates = [s for s in stats if not s.down]
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda s: (s.inflight * s.ewma_ms, s.ewma_ms, s.peer.base_url),
    )
    return best.peer


class Distributor:
    def __init__(
        self,
        peers: list[Peer],
        model: str,
        client_factory: Optional[Callable] = None,
    ) -> None:
        if not peers:
            raise SwarmError(
                "no Ollama peer found - start one with: ollama serve, "
                "or set SWARM_PEERS"
            )
        self.model = model
        # Candidate tiers, best first: peers serving the model under its
        # exact name; then peers serving the same model under a spelling
        # models_match recognises (issue #47); then, as before, any peer
        # at all — its client resolves to a model it actually has.
        exact = [p for p in peers if model in (p.models or [])]
        if exact:
            self.peers: list[Peer] = exact
        else:
            matched = [
                p
                for p in peers
                if any(models_match(model, name) for name in (p.models or []))
            ]
            self.peers = matched if matched else list(peers)
        factory = client_factory or (lambda peer: client_for_peer(peer, model))
        self._clients = {p.base_url: factory(p) for p in self.peers}
        self.counts: dict[str, int] = {p.base_url: 0 for p in self.peers}
        self.down: set[str] = set()
        self._stats: dict[str, PeerStats] = {
            p.base_url: PeerStats(peer=p, ewma_ms=_seed_ewma(p))
            for p in self.peers
        }

    # -- introspection ---------------------------------------------------------
    def peer_stats(self) -> dict[str, PeerStats]:
        """A snapshot of the scheduling stats, keyed by base URL."""
        return dict(self._stats)

    # -- single calls --------------------------------------------------------
    def best_peer(self) -> Peer:
        """The fastest peer (discovery sorts by latency)."""
        alive = [p for p in self.peers if p.base_url not in self.down]
        if not alive:
            raise SwarmError("all swarm peers are down")
        return min(alive, key=lambda p: p.latency_ms)

    def client_for(self, peer: Peer):
        return self._clients[peer.base_url]

    def ask(self, messages: list[dict], model: Optional[str] = None) -> str:
        """One chat call on the fastest live peer."""
        peer = self.best_peer()
        return self._clients[peer.base_url].chat(messages, model=model)

    # -- parallel map ----------------------------------------------------------
    def map_units(self, units: list, fn: Callable) -> list:
        """Run ``fn(client, unit)`` for every unit, spread across the swarm.

        One worker thread per peer pulls units from a shared queue, but a
        worker may only take the next unit when :func:`pick_peer` selects
        its peer — so units flow toward the peer with the lowest predicted
        wait, and a fast peer ends up running more of the batch. ``fn``
        receives the client of the peer the unit was assigned to. Returns
        results in input order; a unit that failed everywhere holds the
        exception instance as its result.
        """
        units = list(units)
        results: list = [None] * len(units)
        if not units:
            return results

        total = len(units)
        pending: deque[tuple[int, object]] = deque(enumerate(units))
        attempts: dict[int, int] = {}
        finished = 0
        cond = threading.Condition()
        stats = self._stats

        def worker(peer: Peer) -> None:
            nonlocal finished
            st = stats[peer.base_url]
            client = self._clients[peer.base_url]
            while True:
                with cond:
                    while True:
                        if finished >= total or st.down:
                            return
                        if pending:
                            chosen = pick_peer(stats.values())
                            if chosen is None:
                                return
                            if chosen.base_url == peer.base_url:
                                idx, unit = pending.popleft()
                                st.inflight += 1
                                break
                        # Not our turn (or nothing pending yet): wait for
                        # the next state change. The timeout is a safety
                        # net so a missed notification can never deadlock
                        # the batch; completions notify immediately.
                        cond.wait(timeout=0.05)
                started = time.perf_counter()
                try:
                    value = fn(client, unit)
                except Exception as exc:  # peer is presumed dead
                    with cond:
                        st.inflight -= 1
                        st.down = True
                        self.down.add(peer.base_url)
                        attempts[idx] = attempts.get(idx, 0) + 1
                        if attempts[idx] >= len(self.peers):
                            results[idx] = exc
                            finished += 1
                        else:
                            pending.appendleft((idx, unit))
                        cond.notify_all()
                    return
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                with cond:
                    st.inflight -= 1
                    st.ewma_ms = _ewma_update(st.ewma_ms, elapsed_ms)
                    results[idx] = value
                    self.counts[peer.base_url] += 1
                    finished += 1
                    cond.notify_all()

        threads = [
            threading.Thread(target=worker, args=(peer,), daemon=True)
            for peer in self.peers
            if peer.base_url not in self.down
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Anything still pending has no live peer left to run it.
        for idx, _unit in pending:
            if results[idx] is None:
                results[idx] = SwarmError("all swarm peers are down")
        return results
