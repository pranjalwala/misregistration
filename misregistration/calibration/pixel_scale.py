"""Pixel-to-mm scale determination (hardware-agnostic).

Scale is derived from:
  1. TIFF XResolution/YResolution metadata   (preferred)
  2. Measured distance between known fiducials on a calibration target
  3. Explicit runtime argument

No hardcoded DPI or printer constants.
"""

from __future__ import annotations
import dataclasses
import math
from typing import Optional
import numpy as np


@dataclasses.dataclass
class PixelScale:
    """Encapsulates px/mm (and mm/px) for a scanned image."""
    px_per_mm_x: float
    px_per_mm_y: float
    source: str = "unknown"  # "metadata" | "calibration_target" | "argument"

    @property
    def px_per_mm(self) -> float:
        """Isotropic average."""
        return (self.px_per_mm_x + self.px_per_mm_y) / 2.0

    @property
    def mm_per_px(self) -> float:
        return 1.0 / self.px_per_mm

    def px_to_mm(self, px: float) -> float:
        return px * self.mm_per_px

    def mm_to_px(self, mm: float) -> float:
        return mm * self.px_per_mm


def scale_from_metadata(meta) -> PixelScale:
    """Extract PixelScale from a ScanMetadata object."""
    return PixelScale(
        px_per_mm_x=meta.px_per_mm,
        px_per_mm_y=meta.px_per_mm,
        source="metadata",
    )


def scale_from_target(
    centroid_a_px: tuple[float, float],
    centroid_b_px: tuple[float, float],
    known_distance_mm: float,
) -> PixelScale:
    """Derive scale from two fiducial centroids with a known physical separation.

    Parameters
    ----------
    centroid_a_px / centroid_b_px : (x, y) in pixels
    known_distance_mm             : physical distance in mm
    """
    dx = centroid_b_px[0] - centroid_a_px[0]
    dy = centroid_b_px[1] - centroid_a_px[1]
    dist_px = math.sqrt(dx**2 + dy**2)
    if dist_px < 1e-6:
        raise ValueError("Centroids are too close; cannot determine scale.")
    px_per_mm = dist_px / known_distance_mm
    return PixelScale(
        px_per_mm_x=px_per_mm,
        px_per_mm_y=px_per_mm,
        source="calibration_target",
    )


def scale_from_dpi(dpi: float) -> PixelScale:
    """Convenience: build PixelScale from a known DPI value (e.g. from CLI arg)."""
    return PixelScale(
        px_per_mm_x=dpi / 25.4,
        px_per_mm_y=dpi / 25.4,
        source="argument",
    )
