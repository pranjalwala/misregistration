"""Shift estimation from NCC and fiducial centroids; RMS computation (Eq. 2)."""

from __future__ import annotations
import dataclasses
from typing import Optional
import numpy as np

from .ncc import compute_ncc, find_peak_subpixel


@dataclasses.dataclass
class ChannelShift:
    channel: str
    dx_px: float
    dy_px: float
    rms_px: float
    rms_mm: float
    peak_ncc: float
    confidence: float          # 0–1
    n_marks: int = 1
    px_per_mm: Optional[float] = None


def estimate_shift(
    reference: np.ndarray,
    scanned: np.ndarray,
    channel: str,
    max_shift: int = 50,
    subpixel_method: str = "gaussian",
    px_per_mm: Optional[float] = None,
    baseline_shift: Optional[tuple[float, float]] = None,
) -> ChannelShift:
    """Estimate (dx, dy) for a single channel using NCC (Eq. 3–4).

    Parameters
    ----------
    reference      : float32 [0,1] digital separation
    scanned        : float32 [0,1] scanned separation (aligned)
    channel        : "C" / "M" / "Y" / "K"
    px_per_mm      : resolution in px/mm (from scanner metadata or calibration)
    baseline_shift : (dx, dy) baseline to subtract (session correction)
    """
    ncc_map = compute_ncc(reference, scanned, max_shift=max_shift)
    dx, dy, peak = find_peak_subpixel(ncc_map, method=subpixel_method)

    if baseline_shift is not None:
        dx -= baseline_shift[0]
        dy -= baseline_shift[1]

    rms_px = float(np.sqrt(dx**2 + dy**2))
    rms_mm = (rms_px / px_per_mm) if px_per_mm else float("nan")

    # Confidence: normalised peak (spec: min_peak_value = 0.3)
    confidence = float(np.clip((peak - 0.3) / 0.7, 0.0, 1.0))

    return ChannelShift(
        channel=channel,
        dx_px=dx, dy_px=dy,
        rms_px=rms_px, rms_mm=rms_mm,
        peak_ncc=peak,
        confidence=confidence,
        px_per_mm=px_per_mm,
    )


def rms_shift(
    dx_list: list[float],
    dy_list: list[float],
) -> float:
    """Compute RMS shift from multiple mark displacements (Eq. 2).

    RMS_shift = sqrt( (1/N) * sum(dx_i^2 + dy_i^2) )
    """
    n = len(dx_list)
    if n == 0:
        return float("nan")
    sq = [dx**2 + dy**2 for dx, dy in zip(dx_list, dy_list)]
    return float(np.sqrt(np.mean(sq)))


def px_to_mm(pixels: float, px_per_mm: float) -> float:
    return pixels / px_per_mm


def shifts_from_centroids(
    centroids_k: list[tuple[float, float]],
    centroids_ch: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Compute per-mark (dx, dy) from K and channel centroid lists."""
    dx_list, dy_list = [], []
    for (kx, ky), (cx, cy) in zip(centroids_k, centroids_ch):
        dx_list.append(cx - kx)
        dy_list.append(cy - ky)
    return dx_list, dy_list
