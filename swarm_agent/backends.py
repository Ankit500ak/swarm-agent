"""Hardware awareness: what is this machine good at?

On Apple Silicon the interesting facts are the chip and the *unified*
memory size - one pool shared by CPU and GPU, which is exactly what local
LLM inference wants. Those are read via ``sysctl`` on macOS and guarded so
every other platform still gets a sane report.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _memory_gb() -> float | None:
    if platform.system() == "Darwin":
        raw = _sysctl("hw.memsize")
        if raw and raw.isdigit():
            return round(int(raw) / (1024 ** 3), 1)
        return None
    try:  # Linux
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024 ** 3), 1)
    except (ValueError, OSError, AttributeError):
        return None


def hardware_report() -> dict:
    """A plain dict describing this machine's inference-relevant hardware."""
    system = platform.system()
    machine = platform.machine()
    apple_silicon = system == "Darwin" and machine == "arm64"
    report = {
        "system": system,
        "machine": machine,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "apple_silicon": apple_silicon,
        "chip": None,
        "unified_memory_gb": None,
    }
    if apple_silicon:
        report["chip"] = _sysctl("machdep.cpu.brand_string")
        report["unified_memory_gb"] = report["memory_gb"]
    return report


def mlx_available() -> bool:
    """True if MLX is importable here, or an mlx_lm.server answers locally."""
    try:
        import mlx  # noqa: F401
        return True
    except Exception:
        pass
    try:
        with socket.create_connection(("127.0.0.1", 8080), timeout=0.5):
            return True
    except OSError:
        return False


def recommendations(report: dict | None = None) -> list[str]:
    """Human-readable guidance: which backend and model size fit this box."""
    if report is None:
        report = hardware_report()
    tips: list[str] = []
    mem = report.get("unified_memory_gb") or report.get("memory_gb")

    if report.get("apple_silicon"):
        chip = report.get("chip") or "Apple Silicon"
        tips.append(f"{chip}: use Ollama (Metal) or an MLX peer for the fastest local tokens.")
        if mem is not None:
            if mem >= 64:
                tips.append("64GB+ unified memory: qwen2.5-coder:32b fits comfortably; 7b/14b fly.")
            elif mem >= 32:
                tips.append("32GB unified memory: qwen2.5-coder:14b is the sweet spot; 32b works quantised.")
            elif mem >= 16:
                tips.append("16GB unified memory: stick to qwen2.5-coder:7b (or 14b at low context).")
            else:
                tips.append("Under 16GB unified memory: qwen2.5-coder:7b at modest context, or lean on bigger swarm peers.")
        if mlx_available():
            tips.append("MLX detected: add it as a peer (mlx_lm.server) - it is often the fastest backend on this chip.")
        else:
            tips.append("MLX not detected: `pip install mlx-lm` and run `mlx_lm.server` to add this Mac as an MLX peer.")
    else:
        tips.append("Not Apple Silicon: run Ollama here and let the swarm spread heavy analysis across machines.")
        if mem is not None and mem < 16:
            tips.append("Limited RAM on this box: prefer small models locally and distribute analysis to stronger peers.")

    tips.append("Whatever the hardware: inference stays on your LAN - no code leaves the network.")
    return tips
