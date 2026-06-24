"""Session baseline measurement and correction.

Scans a zero-shift reference sheet to establish per-channel baseline noise
(mechanical imprecision, lens distortion, etc.).  All subsequent measurements
subtract this baseline.
"""

from __future__ import annotations
import dataclasses
from typing import Optional
import numpy as np


@dataclasses.dataclass
class SessionBaseline:
    """Per-channel baseline shift vectors measured from a zero-shift reference."""
    baselines: dict  # {"C": (dx, dy), "M": (dx, dy), ...}
    n_observations: int = 1
    notes: str = ""

    def correction(self, channel: str) -> tuple[float, float]:
        return self.baselines.get(channel, (0.0, 0.0))


def measure_baseline(
    reference_separations: dict[str, np.ndarray],
    scanned_separations: dict[str, np.ndarray],
    channels: tuple = ("C", "M", "Y"),
    max_shift: int = 20,
    subpixel_method: str = "gaussian",
) -> SessionBaseline:
    """Measure baseline from a zero-shift reference scan.

    Parameters
    ----------
    reference_separations : dict channel → float32 [0,1] digital separation
    scanned_separations   : dict channel → float32 [0,1] scanned separation
                            (should be near-zero shift)
    """
    from misregistration.measurement.ncc import compute_ncc, find_peak_subpixel

    baselines = {}
    for ch in channels:
        ref = reference_separations.get(ch)
        scan = scanned_separations.get(ch)
        if ref is None or scan is None:
            baselines[ch] = (0.0, 0.0)
            continue
        ncc_map = compute_ncc(ref, scan, max_shift=max_shift)
        dx, dy, _ = find_peak_subpixel(ncc_map, method=subpixel_method)
        baselines[ch] = (float(dx), float(dy))

    return SessionBaseline(baselines=baselines, n_observations=1,
                           notes="auto-measured from zero-shift reference")


def apply_baseline_correction(
    dx: float,
    dy: float,
    channel: str,
    baseline: SessionBaseline,
) -> tuple[float, float]:
    """Subtract session baseline from a measured shift."""
    bx, by = baseline.correction(channel)
    return dx - bx, dy - by


def null_baseline(channels: tuple = ("C", "M", "Y", "K")) -> SessionBaseline:
    """Return a zero baseline (no correction)."""
    return SessionBaseline(
        baselines={ch: (0.0, 0.0) for ch in channels},
        notes="null baseline – no correction applied",
    )
