"""Finding the swarm: which machines on the LAN can run inference?

Discovery starts from the configured peers. With ``auto_discover`` enabled it
also scans the local /24 subnet for Ollama's port (11434) and browses mDNS
(DNS-SD) for advertised services, and it picks up an MLX server
(``mlx_lm.server``, port 8080) when one is configured or already listening on
localhost. Everything is probed with short timeouts and the port scan runs
concurrently, so discovery stays fast even on a quiet network.

One honest caveat about the mDNS channel: stock Ollama does not advertise
itself over mDNS, so in practice it mainly finds wrapped or third-party
services that do publish ``_ollama._tcp`` (or an ``_http._tcp`` service on
port 11434). The subnet scan remains the workhorse; mDNS is a bonus channel
that costs a second or two and silently yields nothing on most networks.
"""

from __future__ import annotations

import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from .config import Config, normalize_url
from .ollama import MLXClient, OllamaClient

OLLAMA_PORT = 11434
MLX_PORT = 8080

# mDNS / DNS-SD (RFC 6762/6763) — pure-stdlib querier, no zeroconf dependency.
MDNS_GROUP = "224.0.0.251"
MDNS_PORT = 5353
MDNS_TIMEOUT = 1.5  # total browse budget, seconds — discovery must stay fast
MDNS_SERVICES = ("_ollama._tcp.local", "_http._tcp.local")

_TYPE_A = 1
_TYPE_PTR = 12
_TYPE_SRV = 33
_CLASS_IN = 1


@dataclass
class Peer:
    """One inference peer on the local network."""

    base_url: str
    models: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    kind: str = "ollama"  # "ollama" | "mlx"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.base_url} ({self.kind}, {self.latency_ms:.0f} ms)"


# -- probes (module-level so tests can monkeypatch them) ---------------------

def _tcp_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _fetch_ollama_peer(base_url: str, timeout: float) -> Optional[Peer]:
    client = OllamaClient(base_url)
    start = time.perf_counter()
    try:
        models = client.tags(timeout=timeout)
    except Exception:
        return None
    latency = (time.perf_counter() - start) * 1000.0
    return Peer(base_url=base_url, models=models, latency_ms=latency, kind="ollama")


def _fetch_mlx_peer(base_url: str) -> Optional[Peer]:
    host = base_url.split("://", 1)[-1].split(":")[0]
    if not _tcp_open(host, MLX_PORT, timeout=1.0):
        return None
    client = MLXClient(base_url)
    start = time.perf_counter()
    try:
        models = client.tags(timeout=2.0)
    except Exception:
        models = []
    latency = (time.perf_counter() - start) * 1000.0
    return Peer(base_url=base_url, models=models, latency_ms=latency, kind="mlx")


def _local_ip() -> str:
    """Best-effort LAN IP of this machine (no traffic is actually sent)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


def _scan_subnet(config: Config) -> list[Peer]:
    ip = _local_ip()
    parts = ip.split(".")
    if len(parts) != 4:
        return []
    prefix = ".".join(parts[:3])
    hosts = [f"{prefix}.{i}" for i in range(1, 255)]

    def probe(host: str) -> Optional[str]:
        return host if _tcp_open(host, OLLAMA_PORT, config.scan_timeout) else None

    responders: list[str] = []
    with ThreadPoolExecutor(max_workers=64) as pool:
        for hit in pool.map(probe, hosts):
            if hit:
                responders.append(hit)

    peers: list[Peer] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for peer in pool.map(
            lambda h: _fetch_ollama_peer(f"http://{h}:{OLLAMA_PORT}", config.peer_timeout),
            responders,
        ):
            if peer is not None:
                peers.append(peer)
    return peers


# -- mDNS browsing -------------------------------------------------------------
#
# A minimal DNS-SD client over UDP multicast. Only what discovery needs:
# PTR queries for the service types in MDNS_SERVICES, and a wire parser for
# PTR/SRV/A records (with compression-pointer support). Every failure mode —
# no multicast route, blocked UDP, malformed packets — degrades silently to
# "no mDNS peers"; the subnet scan is unaffected.

def _encode_name(name: str) -> bytes:
    out = bytearray()
    for label in name.rstrip(".").split("."):
        encoded = label.encode("utf-8")
        out.append(len(encoded))
        out += encoded
    out.append(0)
    return bytes(out)


def _build_ptr_query(service: str) -> bytes:
    header = struct.pack("!HHHHHH", 0, 0, 1, 0, 0, 0)
    question = _encode_name(service) + struct.pack("!HH", _TYPE_PTR, _CLASS_IN)
    return header + question


def _parse_name(data: bytes, offset: int) -> tuple[str, int]:
    """Decode a possibly-compressed DNS name at ``offset``.

    Returns ``(name, next_offset)`` where ``next_offset`` is where the caller
    continues reading (just past the name, or just past the first compression
    pointer). Raises :class:`ValueError` on malformed input, including
    compression-pointer loops.
    """
    labels: list[str] = []
    next_offset = offset
    jumped = False
    seen_pointers: set[int] = set()
    while True:
        if offset >= len(data):
            raise ValueError("name runs past end of packet")
        length = data[offset]
        if length & 0xC0 == 0xC0:  # compression pointer
            if offset + 1 >= len(data):
                raise ValueError("truncated compression pointer")
            pointer = ((length & 0x3F) << 8) | data[offset + 1]
            if pointer in seen_pointers:
                raise ValueError("compression pointer loop")
            seen_pointers.add(pointer)
            if not jumped:
                next_offset = offset + 2
                jumped = True
            offset = pointer
        elif length == 0:
            if not jumped:
                next_offset = offset + 1
            return ".".join(labels), next_offset
        elif length & 0xC0:  # 0x40 / 0x80 label types are not valid here
            raise ValueError("unknown label type")
        else:
            end = offset + 1 + length
            if end > len(data):
                raise ValueError("label runs past end of packet")
            labels.append(data[offset + 1:end].decode("utf-8", "replace"))
            offset = end


def _parse_dns_message(data: bytes) -> list[tuple[str, str, object]]:
    """Extract PTR/SRV/A records from one DNS message.

    Returns ``(name, kind, value)`` tuples: PTR → instance name (str),
    SRV → ``(target, port)``, A → IPv4 address (str). Records are gathered
    from the answer, authority and additional sections (mDNS puts SRV/A in
    the additionals). A malformed tail stops parsing but keeps whatever was
    already collected — callers treat junk as silence, never as an error.
    """
    records: list[tuple[str, str, object]] = []
    if len(data) < 12:
        return records
    try:
        _id, _flags, qd, an, ns, ar = struct.unpack("!HHHHHH", data[:12])
        offset = 12
        for _ in range(qd):
            _qname, offset = _parse_name(data, offset)
            offset += 4  # qtype + qclass
        for _ in range(an + ns + ar):
            name, offset = _parse_name(data, offset)
            rtype, _rclass, _ttl, rdlen = struct.unpack(
                "!HHIH", data[offset:offset + 10])
            offset += 10
            rdata_end = offset + rdlen
            if rdata_end > len(data):
                raise ValueError("rdata runs past end of packet")
            if rtype == _TYPE_PTR:
                target, _ = _parse_name(data, offset)
                records.append((name, "PTR", target))
            elif rtype == _TYPE_SRV:
                port = struct.unpack("!H", data[offset + 4:offset + 6])[0]
                target, _ = _parse_name(data, offset + 6)
                records.append((name, "SRV", (target, port)))
            elif rtype == _TYPE_A and rdlen == 4:
                records.append((name, "A", socket.inet_ntoa(data[offset:rdata_end])))
            offset = rdata_end
    except (ValueError, IndexError, struct.error):
        pass
    return records


def _mdns_urls(records: list[tuple[str, str, object]]) -> list[str]:
    """Resolve parsed DNS-SD records into candidate peer base URLs.

    A service instance becomes a candidate when it has an SRV record; the
    host is the SRV target's A record when one was seen, else the target
    hostname itself. ``_http._tcp`` is a generic service type, so only
    instances on Ollama's port qualify; ``_ollama._tcp`` instances are
    trusted at whatever port their SRV advertises.
    """
    instances: dict[str, str] = {}  # instance name -> service name
    srvs: dict[str, tuple[str, int]] = {}
    addrs: dict[str, str] = {}
    for name, kind, value in records:
        key = name.lower()
        if kind == "PTR":
            instances[str(value).lower()] = key
        elif kind == "SRV":
            srvs[key] = value  # type: ignore[assignment]
        elif kind == "A":
            addrs[key] = str(value)
    urls: set[str] = set()
    for instance, service in instances.items():
        srv = srvs.get(instance)
        if srv is None:
            continue
        target, port = srv
        if service == "_http._tcp.local" and port != OLLAMA_PORT:
            continue
        host = addrs.get(target.lower(), target)
        urls.add(normalize_url(f"http://{host}:{port}"))
    return sorted(urls)


def _browse_mdns(timeout: float = MDNS_TIMEOUT) -> list[str]:
    """Browse mDNS for candidate peer URLs (module-level for monkeypatching).

    Sends PTR queries for :data:`MDNS_SERVICES` to the multicast group and
    collects responses until ``timeout`` expires. Returns peer base URLs,
    de-duplicated and sorted. Never raises for network problems: a failed
    bind, a blocked send or an empty browse all yield ``[]``.

    Expected yield is honestly low: stock Ollama does not advertise via
    mDNS, so this finds wrapped/third-party services, not plain `ollama
    serve` instances — those are the subnet scan's job.
    """
    records: list[tuple[str, str, object]] = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    except OSError:
        return []
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass  # coexist with a system mDNS daemon where possible
        sock.bind(("", MDNS_PORT))
        membership = socket.inet_aton(MDNS_GROUP) + socket.inet_aton("0.0.0.0")
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        queries = [_build_ptr_query(service) for service in MDNS_SERVICES]
        for query in queries:
            sock.sendto(query, (MDNS_GROUP, MDNS_PORT))
        deadline = time.monotonic() + timeout
        resent = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not resent and remaining < timeout / 2:
                # One repeat, in case the first query was lost on the Wi-Fi.
                for query in queries:
                    try:
                        sock.sendto(query, (MDNS_GROUP, MDNS_PORT))
                    except OSError:
                        pass
                resent = True
            sock.settimeout(remaining)
            try:
                packet, _addr = sock.recvfrom(65536)
            except (socket.timeout, OSError):
                break
            records.extend(_parse_dns_message(packet))
    except OSError:
        return []
    finally:
        sock.close()
    return _mdns_urls(records)


def _mdns_peers(config: Config) -> list[Peer]:
    """Peers found via mDNS, built through the same path as scan results.

    Never raises: any mDNS failure simply means "no mDNS peers", and the
    rest of discovery proceeds exactly as before.
    """
    try:
        urls = _browse_mdns(MDNS_TIMEOUT)
    except Exception:
        return []
    peers: list[Peer] = []
    for url in urls:
        peer = _fetch_ollama_peer(url, config.peer_timeout)
        if peer is not None:
            peers.append(peer)
    return peers


# -- main entry point ----------------------------------------------------------

def discover(config: Config) -> list[Peer]:
    """Return the reachable peers, fastest first, de-duplicated by URL."""
    found: dict[str, Peer] = {}

    for url in config.peers:
        url = normalize_url(url)
        peer = _fetch_ollama_peer(url, config.peer_timeout)
        if peer is not None:
            found[peer.base_url] = peer

    if config.auto_discover:
        for peer in _scan_subnet(config):
            found.setdefault(peer.base_url, peer)
        for peer in _mdns_peers(config):
            found.setdefault(peer.base_url, peer)

    mlx_url = config.mlx_url
    if not mlx_url and _tcp_open("127.0.0.1", MLX_PORT, timeout=0.5):
        mlx_url = f"http://localhost:{MLX_PORT}"
    if mlx_url:
        peer = _fetch_mlx_peer(normalize_url(mlx_url))
        if peer is not None:
            found.setdefault(peer.base_url, peer)

    return sorted(found.values(), key=lambda p: p.latency_ms)
