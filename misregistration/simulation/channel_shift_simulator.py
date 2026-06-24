"""Channel misregistration simulator.

Pipeline:
  RGB image → CMYK separations → per-channel sub-pixel shifts
  → recomposition → reconstructed RGB

Visually reproduces: color fringing, halo artifacts, edge splitting,
text color bleeding, overprint hue shifts, false contours, registration drift.

All shift values are in pixels (float, sub-pixel supported).
No hardcoded printer/scanner constants.
"""

from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Optional, Union
import numpy as np


@dataclasses.dataclass
class SimulationResult:
    """Output of one simulation run."""
    image_path: str
    shifts_px: dict            # {"C": (dx, dy), ...}
    reference_rgb: np.ndarray  # uint8 zero-shift reconstruction
    shifted_rgb:   np.ndarray  # uint8 after misregistration
    separations_ref:  dict     # float32 [0,1] per channel, no shift
    separations_shifted: dict  # float32 [0,1] per channel, shifted
    rms_px: dict               # {"C": float, ...}
    rms_mm: dict               # {"C": float, ...} (NaN if no scale)
    dE_image: float
    dE_map:   np.ndarray       # float32 per-pixel ΔE


class ChannelShiftSimulator:
    """Printer-agnostic CMYK channel misregistration simulator.

    Parameters
    ----------
    shifts_px : dict mapping channel → (dx, dy) in pixels.
                If None, uses small random shifts.
    ucr       : under-color removal fraction (0–1) for RGB→CMYK.
    px_per_mm : pixels per mm for unit conversion (optional; NaN if absent).
    """

    def __init__(
        self,
        shifts_px: Optional[dict] = None,
        ucr: float = 0.85,
        px_per_mm: Optional[float] = None,
        subpixel_method: str = "gaussian",
    ):
        self.shifts_px = shifts_px or {
            "C": (1.5, -0.8),
            "M": (-0.5, 1.2),
            "Y": (2.0, 0.3),
            "K": (0.0, 0.0),
        }
        self.ucr = ucr
        self.px_per_mm = px_per_mm
        self.subpixel_method = subpixel_method

    def simulate(self, rgb: np.ndarray, image_path: str = "") -> SimulationResult:
        """Run the full simulation pipeline on a uint8 RGB image."""
        from misregistration.preprocessing.separation import rgb_to_cmyk_separations, cmyk_to_rgb
        from misregistration.measurement.colorimetric import compute_delta_e_mis
        from misregistration.measurement.ncc import compute_ncc, find_peak_subpixel
        from misregistration.measurement.shift import rms_shift

        seps = rgb_to_cmyk_separations(rgb, ucr=self.ucr)
        ref_rgb = cmyk_to_rgb(seps)

        shifted_seps = {}
        for ch, sep in seps.items():
            dx, dy = self.shifts_px.get(ch, (0.0, 0.0))
            shifted_seps[ch] = _apply_subpixel_shift(sep, dx, dy)

        shifted_rgb = cmyk_to_rgb(shifted_seps)

        # Measure RMS per channel via NCC
        rms_px_dict = {}
        rms_mm_dict = {}
        for ch in ("C", "M", "Y"):
            ref_ch = seps[ch]
            sh_ch = shifted_seps[ch]
            ncc_map = compute_ncc(ref_ch, sh_ch, max_shift=60)
            est_dx, est_dy, _ = find_peak_subpixel(ncc_map, method=self.subpixel_method)
            rms = float(np.sqrt(est_dx**2 + est_dy**2))
            rms_px_dict[ch] = rms
            rms_mm_dict[ch] = (rms / self.px_per_mm) if self.px_per_mm else float("nan")

        dE_result = compute_delta_e_mis(ref_rgb, shifted_rgb)

        return SimulationResult(
            image_path=image_path,
            shifts_px=dict(self.shifts_px),
            reference_rgb=ref_rgb,
            shifted_rgb=shifted_rgb,
            separations_ref=seps,
            separations_shifted=shifted_seps,
            rms_px=rms_px_dict,
            rms_mm=rms_mm_dict,
            dE_image=dE_result["dE_image"],
            dE_map=dE_result["dE_map"],
        )


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def simulate_from_png(
    path: str | Path,
    shifts_px: Optional[dict] = None,
    px_per_mm: Optional[float] = None,
    **kw,
) -> SimulationResult:
    from PIL import Image
    rgb = np.array(Image.open(str(path)).convert("RGB"))
    sim = ChannelShiftSimulator(shifts_px=shifts_px, px_per_mm=px_per_mm, **kw)
    return sim.simulate(rgb, image_path=str(path))


def simulate_from_jpg(
    path: str | Path,
    shifts_px: Optional[dict] = None,
    px_per_mm: Optional[float] = None,
    **kw,
) -> SimulationResult:
    return simulate_from_png(path, shifts_px=shifts_px, px_per_mm=px_per_mm, **kw)


def simulate_from_folder(
    folder: str | Path,
    shifts_px: Optional[dict] = None,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    px_per_mm: Optional[float] = None,
    max_images: Optional[int] = None,
    **kw,
) -> list[SimulationResult]:
    """Run simulation on all images in a folder."""
    folder = Path(folder)
    paths = [p for p in sorted(folder.iterdir())
             if p.suffix.lower() in extensions]
    if max_images:
        paths = paths[:max_images]
    results = []
    for p in paths:
        try:
            r = simulate_from_png(p, shifts_px=shifts_px, px_per_mm=px_per_mm, **kw)
            results.append(r)
        except Exception as e:
            print(f"[warn] skipping {p.name}: {e}")
    return results


# ---------------------------------------------------------------------------
# private
# ---------------------------------------------------------------------------

def _apply_subpixel_shift(
    sep: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Apply a (possibly sub-pixel) 2D translation to a float32 separation."""
    import cv2
    h, w = sep.shape[:2]
    M = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
    shifted = cv2.warpAffine(
        sep.astype(np.float32), M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return shifted
