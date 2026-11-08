import time

import pytest

from swarm_agent.discovery import Peer
from swarm_agent.distributor import (
    Distributor,
    PeerStats,
    model_key,
    models_match,
    pick_peer,
)
from swarm_agent.ollama import SwarmError

MODEL = "qwen2.5-coder:7b"


class FakeClient:
    def __init__(self, name, fail=False, delay=0.0):
        self.name = name
        self.fail = fail
        self.delay = delay

    def work(self, unit):
        if self.fail:
            raise SwarmError(f"{self.name} is down")
        if self.delay:
            time.sleep(self.delay)
        return f"{self.name}:{unit}"


def _peers():
    return [
        Peer(base_url="http://p1:11434", models=[MODEL], latency_ms=1.0),
        Peer(base_url="http://p2:11434", models=[MODEL], latency_ms=2.0),
    ]


def _distributor(p1_fail=False, p2_fail=False):
    fakes = {
        "http://p1:11434": FakeClient("p1", fail=p1_fail),
        "http://p2:11434": FakeClient("p2", fail=p2_fail),
    }
    dist = Distributor(_peers(), MODEL, client_factory=lambda peer: fakes[peer.base_url])
    return dist


def test_results_in_input_order():
    # Scheduling is latency-weighted now (round-robin is gone), so which
    # peer ran a unit is timing-dependent — but every unit runs exactly
    # once and results come back in input order regardless.
    dist = _distributor()
    results = dist.map_units(list(range(6)), lambda client, unit: unit)
    assert results == list(range(6))
    assert sum(dist.counts.values()) == 6


def test_failover_reassigns_units():
    dist = _distributor(p1_fail=True)
    results = dist.map_units(list(range(6)), lambda client, unit: client.work(unit))
    # every unit eventually completed by the surviving peer, in order
    assert results == [f"p2:{i}" for i in range(6)]
    assert dist.counts["http://p2:11434"] == 6
    assert dist.counts["http://p1:11434"] == 0
    assert "http://p1:11434" in dist.down


def test_all_peers_down_yields_exceptions():
    dist = _distributor(p1_fail=True, p2_fail=True)
    results = dist.map_units([1, 2], lambda client, unit: client.work(unit))
    assert all(isinstance(r, Exception) for r in results)


def test_empty_units():
    dist = _distributor()
    assert dist.map_units([], lambda c, u: u) == []


def test_no_peers_raises():
    with pytest.raises(SwarmError):
        Distributor([], MODEL)


def test_prefers_peers_with_model():
    peers = [
        Peer(base_url="http://other:11434", models=["llama3:8b"], latency_ms=0.5),
        Peer(base_url="http://coder:11434", models=[MODEL], latency_ms=9.0),
    ]
    dist = Distributor(peers, MODEL, client_factory=lambda peer: FakeClient("x"))
    assert [p.base_url for p in dist.peers] == ["http://coder:11434"]
    assert dist.best_peer().base_url == "http://coder:11434"


def test_ask_uses_fastest_peer():
    dist = _distributor()

    class Chatty(FakeClient):
        def chat(self, messages, model=None):
            return f"answer from {self.name}"

    fakes = {"http://p1:11434": Chatty("p1"), "http://p2:11434": Chatty("p2")}
    dist._clients = fakes
    assert dist.ask([{"role": "user", "content": "hi"}]) == "answer from p1"


# -- latency-weighted scheduling ----------------------------------------------


def _timed_distributor(fast_delay, slow_delay, units_latency=(1.0, 5.0)):
    peers = [
        Peer(base_url="http://fast:11434", models=[MODEL], latency_ms=units_latency[0]),
        Peer(base_url="http://slow:11434", models=[MODEL], latency_ms=units_latency[1]),
    ]
    fakes = {
        "http://fast:11434": FakeClient("fast", delay=fast_delay),
        "http://slow:11434": FakeClient("slow", delay=slow_delay),
    }
    return Distributor(peers, MODEL, client_factory=lambda peer: fakes[peer.base_url])


def test_fast_peer_gets_more_units():
    # Scripted speeds: the fast peer's units take ~2 ms, the slow peer's
    # ~60 ms. Over a batch of 24 the fast peer must do strictly more work.
    dist = _timed_distributor(fast_delay=0.002, slow_delay=0.06)
    units = list(range(24))
    results = dist.map_units(units, lambda client, unit: client.work(unit))
    # every unit ran exactly once; order is by input index, not peer
    assert sorted(int(r.split(":")[1]) for r in results) == units
    fast = dist.counts["http://fast:11434"]
    slow = dist.counts["http://slow:11434"]
    assert fast + slow == 24
    assert fast > slow
    assert fast >= 18  # wide margin: this must not be a coin flip


def test_ewma_tracks_observed_speed():
    # Equal discovery seeds: any EWMA gap afterwards came from timings
    # observed during the run, not from the seeds.
    dist = _timed_distributor(
        fast_delay=0.001, slow_delay=0.05, units_latency=(5.0, 5.0)
    )
    dist.map_units(list(range(12)), lambda client, unit: client.work(unit))
    stats = dist.peer_stats()
    assert stats["http://slow:11434"].ewma_ms > 5.0  # moved toward ~50 ms
    assert stats["http://slow:11434"].ewma_ms > stats["http://fast:11434"].ewma_ms


def _stats(url, ewma, inflight=0, down=False):
    return PeerStats(
        peer=Peer(base_url=url, models=[MODEL], latency_ms=ewma),
        ewma_ms=ewma,
        inflight=inflight,
        down=down,
    )


def test_pick_peer_empty_and_all_down():
    assert pick_peer([]) is None
    assert pick_peer([_stats("http://a:1", 1.0, down=True)]) is None


def test_pick_peer_lowest_predicted_wait():
    # A: 1 in flight x 10 ms = 10 ms wait. B: 3 in flight x 2 ms = 6 ms.
    a = _stats("http://a:1", 10.0, inflight=1)
    b = _stats("http://b:1", 2.0, inflight=3)
    assert pick_peer([a, b]).base_url == "http://b:1"
    # An idle peer (wait 0) beats both, however slow its EWMA is.
    c = _stats("http://c:1", 500.0, inflight=0)
    assert pick_peer([a, b, c]).base_url == "http://c:1"


def test_pick_peer_excludes_down():
    down_best = _stats("http://a:1", 0.5, inflight=0, down=True)
    alive = _stats("http://b:1", 50.0, inflight=2)
    assert pick_peer([down_best, alive]).base_url == "http://b:1"


def test_pick_peer_tie_breaks_are_deterministic():
    # Equal predicted wait -> lower EWMA wins.
    a = _stats("http://a:1", 4.0, inflight=1)  # wait 4
    b = _stats("http://b:1", 2.0, inflight=2)  # wait 4, faster peer
    assert pick_peer([a, b]).base_url == "http://b:1"
    # Fully tied -> lexicographically smaller base URL, either input order.
    x = _stats("http://x:1", 3.0, inflight=1)
    y = _stats("http://y:1", 3.0, inflight=1)
    assert pick_peer([y, x]).base_url == "http://x:1"
    assert pick_peer([x, y]).base_url == "http://x:1"


# -- model-name matching (issue #47) ------------------------------------------

MLX_CODER_7B = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


def test_model_key_normalises_ollama_and_mlx_spellings():
    # The two spellings that must agree: Ollama's tag form and the MLX
    # server's registry-prefixed, quant-suffixed form.
    assert model_key(MODEL) == ("qwen2.5-coder", "7b")
    assert model_key(MLX_CODER_7B) == ("qwen2.5-coder", "7b")
    assert model_key("llama3.1:8b") == ("llama3.1", "8b")
    assert model_key("qwen2.5-coder") == ("qwen2.5-coder", None)
    # A bare :latest tag carries no information.
    assert model_key("qwen2.5-coder:latest") == ("qwen2.5-coder", None)
    # Unparseable: empty, or nothing left after the noise tokens.
    assert model_key("") is None
    assert model_key("7b") is None
    assert model_key(None) is None


def test_models_match_rules():
    assert models_match(MODEL, MLX_CODER_7B)
    assert models_match(MODEL, "qwen2.5-coder:7b-instruct-q4_K_M")
    # Size disagreements and family/version disagreements never match.
    assert not models_match(MODEL, "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")
    assert not models_match(MODEL, "mlx-community/Qwen3-Coder-7B-Instruct-4bit")
    assert not models_match(MODEL, "llama3.1:7b")
    # A request with no size accepts any size of the right model.
    assert models_match("qwen2.5-coder", MLX_CODER_7B.replace("7B", "32B"))


def test_mlx_peer_selected_for_default_model():
    # (a) The MLX peer is the only candidate for the default model, even
    # though an unrelated peer is faster — exact filtering used to fall
    # back to *all* peers here.
    peers = [
        Peer(base_url="http://chatty:11434", models=["llama3.1:8b"], latency_ms=0.5),
        Peer(base_url="http://mlx:8080", models=[MLX_CODER_7B], latency_ms=9.0),
    ]
    dist = Distributor(peers, MODEL, client_factory=lambda peer: FakeClient("x"))
    assert [p.base_url for p in dist.peers] == ["http://mlx:8080"]
    assert dist.best_peer().base_url == "http://mlx:8080"


def test_wrong_size_mlx_peer_not_selected():
    # (b) A 14B build of the right model does not satisfy a 7B request:
    # no tier matches, so selection falls back to all peers (as before).
    peers = [
        Peer(
            base_url="http://mlx14:8080",
            models=["mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"],
            latency_ms=1.0,
        ),
        Peer(base_url="http://chatty:11434", models=["llama3.1:8b"], latency_ms=2.0),
    ]
    dist = Distributor(peers, MODEL, client_factory=lambda peer: FakeClient("x"))
    assert {p.base_url for p in dist.peers} == {
        "http://mlx14:8080",
        "http://chatty:11434",
    }


def test_different_family_not_selected():
    # (c) Same size, wrong family/version: not a match either.
    peers = [
        Peer(
            base_url="http://qwen3:8080",
            models=["mlx-community/Qwen3-Coder-7B-Instruct-4bit"],
            latency_ms=1.0,
        ),
        Peer(base_url="http://chatty:11434", models=["llama3.1:8b"], latency_ms=2.0),
    ]
    dist = Distributor(peers, MODEL, client_factory=lambda peer: FakeClient("x"))
    assert {p.base_url for p in dist.peers} == {
        "http://qwen3:8080",
        "http://chatty:11434",
    }


def test_exact_match_beats_normalised_match():
    # (d) When a peer serves the exact requested name, normalised-only
    # peers are not candidates at all — however fast they are.
    peers = [
        Peer(base_url="http://mlx:8080", models=[MLX_CODER_7B], latency_ms=0.5),
        Peer(base_url="http://exact:11434", models=[MODEL], latency_ms=50.0),
    ]
    dist = Distributor(peers, MODEL, client_factory=lambda peer: FakeClient("x"))
    assert [p.base_url for p in dist.peers] == ["http://exact:11434"]


def test_unparseable_names_behave_as_before():
    # (e) Unparseable names match only by exact string equality.
    assert models_match("7b", "7b")
    assert not models_match("7b", MODEL)
    assert not models_match(MODEL, "4bit")
    # A peer whose only "model" is unparseable is reached only through
    # the any-peer fallback, exactly as with exact filtering before.
    peers = [
        Peer(base_url="http://odd:11434", models=["4bit"], latency_ms=1.0),
        Peer(base_url="http://chatty:11434", models=["llama3.1:8b"], latency_ms=2.0),
    ]
    dist = Distributor(peers, MODEL, client_factory=lambda peer: FakeClient("x"))
    assert {p.base_url for p in dist.peers} == {
        "http://odd:11434",
        "http://chatty:11434",
    }
