from swarm_agent import discovery
from swarm_agent.config import Config
from swarm_agent.discovery import Peer, discover


def _fake_fetch(latencies):
    def fetch(url, timeout):
        if url in latencies:
            return Peer(base_url=url, models=["qwen2.5-coder:7b"],
                        latency_ms=latencies[url], kind="ollama")
        return None
    return fetch


def test_configured_peers_sorted_by_latency(monkeypatch):
    monkeypatch.setattr(discovery, "_fetch_ollama_peer", _fake_fetch({
        "http://slow:11434": 90.0,
        "http://fast:11434": 5.0,
    }))
    monkeypatch.setattr(discovery, "_tcp_open", lambda *a, **k: False)
    cfg = Config(peers=["http://slow:11434", "http://fast:11434", "http://down:11434"])
    peers = discover(cfg)
    assert [p.base_url for p in peers] == ["http://fast:11434", "http://slow:11434"]


def test_deduplication(monkeypatch):
    monkeypatch.setattr(discovery, "_fetch_ollama_peer", _fake_fetch({
        "http://a:11434": 3.0,
    }))
    monkeypatch.setattr(discovery, "_tcp_open", lambda *a, **k: False)
    cfg = Config(peers=["http://a:11434", "http://a:11434"])
    assert len(discover(cfg)) == 1


def test_subnet_scan(monkeypatch):
    monkeypatch.setattr(discovery, "_local_ip", lambda: "192.168.1.5")
    monkeypatch.setattr(
        discovery, "_tcp_open",
        lambda host, port, timeout: host == "192.168.1.10",
    )
    monkeypatch.setattr(discovery, "_fetch_ollama_peer", _fake_fetch({
        "http://192.168.1.10:11434": 12.0,
    }))
    cfg = Config(peers=[], auto_discover=True)
    peers = discover(cfg)
    assert [p.base_url for p in peers] == ["http://192.168.1.10:11434"]


def test_no_scan_without_auto_discover(monkeypatch):
    calls = []
    monkeypatch.setattr(discovery, "_local_ip", lambda: calls.append(1) or "10.0.0.2")
    monkeypatch.setattr(discovery, "_fetch_ollama_peer", _fake_fetch({}))
    monkeypatch.setattr(discovery, "_tcp_open", lambda *a, **k: False)
    cfg = Config(peers=[], auto_discover=False)
    assert discover(cfg) == []
    assert calls == []  # subnet scan never ran


def test_mlx_peer_from_config(monkeypatch):
    monkeypatch.setattr(discovery, "_fetch_ollama_peer", _fake_fetch({}))
    monkeypatch.setattr(discovery, "_fetch_mlx_peer", lambda url: Peer(
        base_url=url, models=["mlx-community/Qwen2.5-Coder-7B"], latency_ms=1.0, kind="mlx"))
    monkeypatch.setattr(discovery, "_tcp_open", lambda *a, **k: False)
    cfg = Config(peers=[], mlx_url="http://localhost:8080")
    peers = discover(cfg)
    assert len(peers) == 1
    assert peers[0].kind == "mlx"
