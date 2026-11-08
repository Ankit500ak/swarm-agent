"""Distributing work across the swarm.

The distributor owns the peer list and the failure policy:

- Peers that serve the requested model are preferred; if none do, any
  healthy peer is used (its client resolves to a model it actually has).
- :meth:`Distributor.map_units` assigns units round-robin, one worker
  thread per peer.
- A peer whose work raises is marked *down*; the failed unit and everything
  still queued behind it are reassigned, round-robin again, to the
  surviving peers. A unit that has failed on every peer ends up holding its
  exception as its result, so callers can report it instead of crashing.
- Results always come back in input order, and ``counts`` records how many
  units each peer completed.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

from .discovery import Peer
from .ollama import SwarmError, client_for_peer


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
        with_model = [p for p in peers if model in (p.models or [])]
        self.peers: list[Peer] = with_model if with_model else list(peers)
        factory = client_factory or (lambda peer: client_for_peer(peer, model))
        self._clients = {p.base_url: factory(p) for p in self.peers}
        self.counts: dict[str, int] = {p.base_url: 0 for p in self.peers}
        self.down: set[str] = set()

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

        ``fn`` receives the client of the peer the unit was assigned to.
        Returns results in input order; a unit that failed everywhere holds
        the exception instance as its result.
        """
        units = list(units)
        results: list = [None] * len(units)
        if not units:
            return results

        pending: deque[tuple[int, object]] = deque(enumerate(units))
        attempts: dict[int, int] = {}
        active: list[Peer] = list(self.peers)
        lock = threading.Lock()

        while pending and active:
            # Round-robin assignment of the pending units across active peers.
            queues: dict[str, deque] = {p.base_url: deque() for p in active}
            for k, item in enumerate(list(pending)):
                queues[active[k % len(active)].base_url].append(item)
            pending.clear()
            failed: deque[tuple[int, object]] = deque()

            def worker(peer: Peer) -> None:
                client = self._clients[peer.base_url]
                queue = queues[peer.base_url]
                while queue:
                    idx, unit = queue.popleft()
                    try:
                        value = fn(client, unit)
                    except Exception as exc:  # peer is presumed dead
                        with lock:
                            self.down.add(peer.base_url)
                            attempts[idx] = attempts.get(idx, 0) + 1
                            rest = [(idx, unit)] + list(queue)
                            queue.clear()
                            for r_idx, r_unit in rest:
                                if attempts.get(r_idx, 0) >= len(self.peers):
                                    results[r_idx] = exc
                                else:
                                    failed.append((r_idx, r_unit))
                        return
                    results[idx] = value
                    with lock:
                        self.counts[peer.base_url] += 1

            threads = [
                threading.Thread(target=worker, args=(peer,), daemon=True)
                for peer in active
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            active = [p for p in active if p.base_url not in self.down]
            pending = failed

        # Anything still pending has no live peer left to run on.
        for idx, _unit in pending:
            if results[idx] is None:
                results[idx] = SwarmError("all swarm peers are down")
        return results
