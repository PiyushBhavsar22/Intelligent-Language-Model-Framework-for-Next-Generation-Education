"""
memtrack.py
===========
Peak-memory measurement for RQ3 (model size vs accuracy, latency, MEMORY).

Two numbers are reported per generation call:
  * peak_vram_mb : peak GPU memory allocated by torch during the call
                   (0.0 when running on CPU)
  * ram_mb       : current resident set size of the Python process
                   (captures CPU-side model weights + activations)

Usage:
    from src.memtrack import MemoryTracker
    with MemoryTracker() as mt:
        answer = generator.generate_constrained(q, ctx)
    print(mt.peak_vram_mb, mt.ram_mb)

No external dependencies: uses torch.cuda stats and /proc (Linux) or the
stdlib `resource` module as fallback, so it works on Linux/Mac. On Windows,
ram_mb falls back to 0.0 unless psutil is installed (optional).
"""
from __future__ import annotations
import os
import sys

import torch


def _process_ram_mb() -> float:
    """Resident memory of this process in MB, best-effort cross-platform."""
    # 1) psutil if available (works everywhere, optional dependency)
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        pass
    # 2) /proc on Linux
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0  # kB -> MB
    except Exception:
        pass
    # 3) resource module (Unix; ru_maxrss is kB on Linux, bytes on macOS)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024.0 if sys.platform != "darwin" else rss / (1024 ** 2)
    except Exception:
        return 0.0


class MemoryTracker:
    """Context manager capturing peak GPU VRAM and process RAM."""

    def __init__(self) -> None:
        self.peak_vram_mb: float = 0.0
        self.ram_mb: float = 0.0
        self._cuda = torch.cuda.is_available()

    def __enter__(self) -> "MemoryTracker":
        if self._cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> None:
        if self._cuda:
            torch.cuda.synchronize()
            self.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self.ram_mb = _process_ram_mb()


def model_footprint_mb() -> dict:
    """One-off snapshot after model load: how much memory the weights occupy."""
    out = {"ram_mb": round(_process_ram_mb(), 1), "vram_mb": 0.0}
    if torch.cuda.is_available():
        out["vram_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 1)
    return out
