"""Tests for swarm_agent.backends — including the Apple Silicon path.

Issue #1 (MLX validation) starts here: before this change the only
tests ran on whatever host happened to run pytest, so the whole
Darwin / sysctl / memory-tier code was untested. These tests fake
Darwin with unittest.mock so they run on any machine, including CI
on Linux, with no real Mac and no real sysctl call.

What is covered:
- hardware_report() shape on the real host (unchanged behaviour)
- mocked Darwin: chip name, unified memory, and the empty-chip case
- _memory_gb() parsing: good value, non-numeric reply, missing sysctl
- recommendations() memory tiers: 8 / 16 / 32 / 64 GB on Apple Silicon
- recommendations() with MLX present vs absent (injected, no socket)
- recommendations() on a non-Apple host
"""
from unittest.mock import patch

from swarm_agent import backends


def _darwin_report(chip="Apple M4", mem_bytes="17179869184"):
    """Run hardware_report() as if on an M-series Mac.

    chip: value returned for machdep.cpu.brand_string.
    mem_bytes: value returned for hw.memsize (16 GiB by default).
    """
    def fake_sysctl(name):
        if name == "machdep.cpu.brand_string":
            return chip
        if name == "hw.memsize":
            return mem_bytes
        return None

    with patch("swarm_agent.backends.platform.system", return_value="Darwin"), \
            patch("swarm_agent.backends.platform.machine", return_value="arm64"), \
            patch("swarm_agent.backends._sysctl", side_effect=fake_sysctl):
        return backends.hardware_report()


def _apple_report(mem_gb, chip="Apple M4"):
    return {
        "system": "Darwin",
        "machine": "arm64",
        "platform": "macOS-15-arm64",
        "python": "3.12.0",
        "cpu_count": 10,
        "memory_gb": mem_gb,
        "apple_silicon": True,
        "chip": chip,
        "unified_memory_gb": mem_gb,
    }


# --- existing behaviour, kept ------------------------------------------------

def test_hardware_report_shape():
    report = backends.hardware_report()
    assert isinstance(report, dict)
    for key in ("system", "machine", "python", "cpu_count",
                "memory_gb", "apple_silicon", "chip", "unified_memory_gb"):
        assert key in report
    assert report["cpu_count"] >= 1


def test_not_apple_silicon_on_linux():
    report = backends.hardware_report()
    if report["system"] == "Linux":
        assert report["apple_silicon"] is False
        assert report["unified_memory_gb"] is None
        assert report["memory_gb"] is not None


def test_recommendations_are_text():
    tips = backends.recommendations()
    assert isinstance(tips, list)
    assert len(tips) >= 1
    assert all(isinstance(t, str) and t for t in tips)
    assert any("LAN" in t for t in tips)


# --- Issue #1: mocked Apple Silicon ------------------------------------------

def test_darwin_report_chip_and_unified_memory():
    report = _darwin_report(chip="Apple M4", mem_bytes="17179869184")
    assert report["system"] == "Darwin"
    assert report["apple_silicon"] is True
    assert report["chip"] == "Apple M4"
    assert report["memory_gb"] == 16.0
    assert report["unified_memory_gb"] == 16.0


def test_darwin_report_64gb():
    report = _darwin_report(chip="Apple M4 Max", mem_bytes="68719476736")
    assert report["chip"] == "Apple M4 Max"
    assert report["memory_gb"] == 64.0
    assert report["unified_memory_gb"] == 64.0


def test_darwin_empty_chip_becomes_none():
    # A sysctl reply of "" means "unknown", not a chip named "".
    report = _darwin_report(chip="")
    assert report["chip"] is None
    assert report["apple_silicon"] is True


def test_darwin_bad_memsize_becomes_none():
    report = _darwin_report(mem_bytes="not-a-number")
    assert report["memory_gb"] is None
    assert report["unified_memory_gb"] is None


def test_darwin_missing_memsize_becomes_none():
    report = _darwin_report(mem_bytes=None)
    assert report["memory_gb"] is None


# --- Issue #1: memory tiers ---------------------------------------------------

def test_recommendations_8gb_apple_silicon():
    tips = backends.recommendations(_apple_report(8.0), mlx=False)
    assert any("Under 16GB" in t for t in tips)
    assert any("7b at modest context" in t for t in tips)


def test_recommendations_16gb_apple_silicon():
    tips = backends.recommendations(_apple_report(16.0), mlx=False)
    assert any("16GB unified memory" in t for t in tips)
    assert any("qwen2.5-coder:7b" in t for t in tips)


def test_recommendations_32gb_apple_silicon():
    tips = backends.recommendations(_apple_report(32.0), mlx=False)
    assert any("32GB unified memory" in t for t in tips)
    assert any("14b is the sweet spot" in t for t in tips)


def test_recommendations_64gb_apple_silicon():
    tips = backends.recommendations(_apple_report(64.0), mlx=False)
    assert any("64GB+ unified memory" in t for t in tips)
    assert any("32b fits comfortably" in t for t in tips)


def test_recommendations_no_memory_still_advises():
    report = _apple_report(None)
    tips = backends.recommendations(report, mlx=False)
    # No memory tier line, but still a chip line, an MLX line and the LAN line.
    assert any("Apple M4" in t for t in tips)
    assert any("LAN" in t for t in tips)


# --- Issue #1: MLX present vs absent, without a live socket ------------------

def test_recommendations_mlx_present():
    tips = backends.recommendations(_apple_report(16.0), mlx=True)
    assert any("MLX detected" in t for t in tips)
    assert not any("MLX not detected" in t for t in tips)


def test_recommendations_mlx_absent():
    tips = backends.recommendations(_apple_report(16.0), mlx=False)
    assert any("MLX not detected" in t for t in tips)
    assert any("pip install mlx-lm" in t for t in tips)


def test_recommendations_non_apple_host():
    report = {
        "system": "Linux", "machine": "x86_64", "platform": "Linux",
        "python": "3.12.0", "cpu_count": 4, "memory_gb": 7.7,
        "apple_silicon": False, "chip": None, "unified_memory_gb": None,
    }
    tips = backends.recommendations(report, mlx=False)
    assert any("Not Apple Silicon" in t for t in tips)
    assert any("Limited RAM" in t for t in tips)
