import pytest

from swarm_agent.discovery import Peer
from swarm_agent.distributor import Distributor
from swarm_agent.ollama import SwarmError

MODEL = "qwen2.5-coder:7b"


class FakeClient:
    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail

    def work(self, unit):
        if self.fail:
            raise SwarmError(f"{self.name} is down")
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


def test_round_robin_and_order():
    dist = _distributor()
    results = dist.map_units(list(range(6)), lambda client, unit: client.work(unit))
    assert results == ["p1:0", "p2:1", "p1:2", "p2:3", "p1:4", "p2:5"]
    assert dist.counts == {"http://p1:11434": 3, "http://p2:11434": 3}


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
