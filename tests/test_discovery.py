import socket
import struct

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


# -- mDNS: wire fixtures -------------------------------------------------------

def _enc_name(name):
    out = b""
    for label in name.split("."):
        if label:
            out += bytes([len(label)]) + label.encode()
    return out + b"\x00"


def _ptr(offset):
    return struct.pack("!H", 0xC000 | offset)


def _mdns_response_packet():
    """A realistic mDNS response: PTR + SRV (answers) and A (additional).

    Compression pointers are used the way real responders use them: the
    instance name's suffix points back at the service name, and the SRV/A
    record names are pointers into earlier parts of the packet.
    """
    buf = bytearray(struct.pack("!HHHHHH", 0, 0x8400, 0, 2, 0, 1))
    service_off = len(buf)  # 12: "_ollama._tcp.local" starts here
    buf += _enc_name("_ollama._tcp.local")
    instance_off = len(buf) + 10  # PTR rdata starts after the fixed fields
    ptr_rdata = bytes([len("Living Room")]) + b"Living Room" + _ptr(service_off)
    buf += struct.pack("!HHIH", 12, 1, 120, len(ptr_rdata)) + ptr_rdata
    buf += _ptr(instance_off)  # SRV record name -> the instance name
    target_off = len(buf) + 10 + 6  # past fixed fields + priority/weight/port
    srv_rdata = struct.pack("!HHH", 0, 0, 11434) + _enc_name("macbook.local")
    buf += struct.pack("!HHIH", 33, 1, 120, len(srv_rdata)) + srv_rdata
    buf += _ptr(target_off)  # A record name -> the SRV target
    a_rdata = socket.inet_aton("192.168.1.42")
    buf += struct.pack("!HHIH", 1, 1, 120, len(a_rdata)) + a_rdata
    return bytes(buf)


def test_mdns_parse_response_with_compression():
    records = discovery._parse_dns_message(_mdns_response_packet())
    assert ("_ollama._tcp.local", "PTR", "Living Room._ollama._tcp.local") in records
    assert ("Living Room._ollama._tcp.local", "SRV", ("macbook.local", 11434)) in records
    assert ("macbook.local", "A", "192.168.1.42") in records
    assert discovery._mdns_urls(records) == ["http://192.168.1.42:11434"]


def test_mdns_parse_garbage_is_silence():
    assert discovery._parse_dns_message(b"") == []
    assert discovery._parse_dns_message(b"\xff" * 64) == []
    # Header promises an answer the packet does not contain.
    truncated = struct.pack("!HHHHHH", 0, 0x8400, 0, 1, 0, 0)
    assert discovery._parse_dns_message(truncated) == []


def test_mdns_parse_pointer_loop_terminates():
    # A record name that is a compression pointer to itself must not hang.
    packet = struct.pack("!HHHHHH", 0, 0x8400, 0, 1, 0, 0) + _ptr(12) + b"\x00" * 10
    assert discovery._parse_dns_message(packet) == []


def test_mdns_urls_http_service_requires_ollama_port():
    records = [
        ("_http._tcp.local", "PTR", "Printer._http._tcp.local"),
        ("Printer._http._tcp.local", "SRV", ("printer.local", 8080)),
        ("printer.local", "A", "10.0.0.9"),
        ("_http._tcp.local", "PTR", "Ollama Web._http._tcp.local"),
        ("Ollama Web._http._tcp.local", "SRV", ("tower.local", 11434)),
    ]
    # The printer (port 8080) is not Ollama; tower.local has no A record,
    # so the SRV target hostname itself becomes the candidate host.
    assert discovery._mdns_urls(records) == ["http://tower.local:11434"]


def test_mdns_urls_ollama_service_any_port():
    records = [
        ("_ollama._tcp.local", "PTR", "Box._ollama._tcp.local"),
        ("Box._ollama._tcp.local", "SRV", ("box.local", 9999)),
        ("box.local", "A", "10.0.0.7"),
    ]
    assert discovery._mdns_urls(records) == ["http://10.0.0.7:9999"]


def test_mdns_urls_instance_without_srv_skipped():
    records = [("_ollama._tcp.local", "PTR", "Ghost._ollama._tcp.local")]
    assert discovery._mdns_urls(records) == []


# -- mDNS: merging with the scan ------------------------------------------------

def _scan_setup(monkeypatch, extra_fetches=None):
    latencies = {"http://192.168.1.10:11434": 12.0}
    if extra_fetches:
        latencies.update(extra_fetches)
    monkeypatch.setattr(discovery, "_local_ip", lambda: "192.168.1.5")
    monkeypatch.setattr(
        discovery, "_tcp_open",
        lambda host, port, timeout: host == "192.168.1.10",
    )
    monkeypatch.setattr(discovery, "_fetch_ollama_peer", _fake_fetch(latencies))


def test_mdns_adds_peer_scan_missed(monkeypatch):
    _scan_setup(monkeypatch, {"http://192.168.1.11:11434": 30.0})
    monkeypatch.setattr(discovery, "_browse_mdns",
                        lambda timeout: ["http://192.168.1.11:11434"])
    cfg = Config(peers=[], auto_discover=True)
    peers = discover(cfg)
    assert [p.base_url for p in peers] == [
        "http://192.168.1.10:11434", "http://192.168.1.11:11434"]


def test_mdns_candidate_deduplicated_against_scan(monkeypatch):
    _scan_setup(monkeypatch)
    monkeypatch.setattr(discovery, "_browse_mdns",
                        lambda timeout: ["http://192.168.1.10:11434"])
    cfg = Config(peers=[], auto_discover=True)
    peers = discover(cfg)
    assert [p.base_url for p in peers] == ["http://192.168.1.10:11434"]


def test_mdns_failure_degrades_to_scan_only(monkeypatch):
    _scan_setup(monkeypatch)

    def boom(timeout):
        raise RuntimeError("multicast unavailable")

    monkeypatch.setattr(discovery, "_browse_mdns", boom)
    cfg = Config(peers=[], auto_discover=True)
    peers = discover(cfg)  # must not raise
    assert [p.base_url for p in peers] == ["http://192.168.1.10:11434"]


def test_no_mdns_without_auto_discover(monkeypatch):
    calls = []
    monkeypatch.setattr(discovery, "_browse_mdns",
                        lambda timeout: calls.append(1) or [])
    monkeypatch.setattr(discovery, "_fetch_ollama_peer", _fake_fetch({}))
    monkeypatch.setattr(discovery, "_tcp_open", lambda *a, **k: False)
    cfg = Config(peers=[], auto_discover=False)
    assert discover(cfg) == []
    assert calls == []  # mDNS browse never ran
