"""Hardware-agnostic calibration target generation.

All dimensions are specified in millimetres and converted to pixels at runtime
using the supplied dpi parameter (no hardcoded printer/scanner constants).
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# unit helpers
# ---------------------------------------------------------------------------

def _mm2px(mm: float, dpi: float) -> int:
    return max(1, int(round(mm * dpi / 25.4)))


def _canvas(width_mm: float, height_mm: float, dpi: float, n_ch: int = 3) -> np.ndarray:
    w = _mm2px(width_mm, dpi)
    h = _mm2px(height_mm, dpi)
    if n_ch == 1:
        return np.full((h, w), 255, dtype=np.uint8)
    return np.full((h, w, n_ch), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# individual target generators
# ---------------------------------------------------------------------------

def make_corner_fiducials(
    width_mm: float = 100.0,
    height_mm: float = 100.0,
    dpi: float = 300.0,
    mark_size_mm: float = 5.0,
    margin_mm: float = 5.0,
) -> np.ndarray:
    """Four filled-square corner fiducial marks on a white background."""
    img = _canvas(width_mm, height_mm, dpi, n_ch=1)
    ms = _mm2px(mark_size_mm, dpi)
    mg = _mm2px(margin_mm, dpi)
    h, w = img.shape
    corners = [(mg, mg), (w - mg - ms, mg), (mg, h - mg - ms), (w - mg - ms, h - mg - ms)]
    for cx, cy in corners:
        img[cy:cy+ms, cx:cx+ms] = 0
    return img


def make_bullseye(
    width_mm: float = 30.0,
    height_mm: float = 30.0,
    dpi: float = 300.0,
    n_rings: int = 4,
    ring_width_mm: float = 1.5,
) -> np.ndarray:
    """Bullseye registration target with alternating black/white rings."""
    img = _canvas(width_mm, height_mm, dpi, n_ch=1)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    ring_px = _mm2px(ring_width_mm, dpi)
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    for i in range(n_rings):
        r_outer = (i + 1) * ring_px
        r_inner = i * ring_px
        mask = (dist >= r_inner) & (dist < r_outer)
        img[mask] = 0 if i % 2 == 0 else 255
    return img


def make_crosshair(
    width_mm: float = 20.0,
    height_mm: float = 20.0,
    dpi: float = 300.0,
    line_width_mm: float = 0.5,
) -> np.ndarray:
    """Cross-hair registration mark."""
    img = _canvas(width_mm, height_mm, dpi, n_ch=1)
    h, w = img.shape
    lw = max(1, _mm2px(line_width_mm, dpi))
    cy, cx = h // 2, w // 2
    img[cy - lw//2: cy + lw//2 + 1, :] = 0
    img[:, cx - lw//2: cx + lw//2 + 1] = 0
    return img


def make_line_targets(
    width_mm: float = 80.0,
    height_mm: float = 20.0,
    dpi: float = 300.0,
    line_widths_mm: tuple = (0.1, 0.2, 0.5, 1.0),
    gap_mm: float = 2.0,
) -> np.ndarray:
    """Vertical lines of varying widths for fine-line registration testing."""
    img = _canvas(width_mm, height_mm, dpi, n_ch=1)
    x = _mm2px(gap_mm, dpi)
    for lw_mm in line_widths_mm:
        lw = max(1, _mm2px(lw_mm, dpi))
        img[:, x:x+lw] = 0
        x += lw + _mm2px(gap_mm, dpi)
        if x >= img.shape[1]:
            break
    return img


def make_slanted_edge(
    width_mm: float = 40.0,
    height_mm: float = 40.0,
    dpi: float = 300.0,
    angle_deg: float = 15.0,
) -> np.ndarray:
    """Slanted black/white edge for MTF/registration analysis."""
    img = _canvas(width_mm, height_mm, dpi, n_ch=1)
    h, w = img.shape
    angle_rad = math.radians(angle_deg)
    cx = w // 2
    for y in range(h):
        x_edge = int(cx + (y - h//2) * math.tan(angle_rad))
        if 0 <= x_edge < w:
            img[y, :x_edge] = 0
    return img


def make_cmyk_registration_chart(
    width_mm: float = 120.0,
    height_mm: float = 40.0,
    dpi: float = 300.0,
    patch_size_mm: float = 10.0,
    channels: tuple = ("C", "M", "Y", "K"),
) -> dict[str, np.ndarray]:
    """Per-channel registration chart: bullseye marks on colour backgrounds.

    Returns dict mapping channel name → uint8 RGB image.
    """
    CHANNEL_COLORS = {
        "C": (0, 255, 255),
        "M": (255, 0, 255),
        "Y": (255, 255, 0),
        "K": (0, 0, 0),
    }
    charts = {}
    ps = _mm2px(patch_size_mm, dpi)
    for ch in channels:
        img = _canvas(width_mm, height_mm, dpi, n_ch=3)
        color = CHANNEL_COLORS.get(ch, (128, 128, 128))
        # Place a coloured filled circle (registration mark)
        h_img, w_img = img.shape[:2]
        cy, cx = h_img // 2, w_img // 4
        ys, xs = np.ogrid[:h_img, :w_img]
        dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        img[dist < ps // 2] = color
        charts[ch] = img
    return charts


def make_overprint_patches(
    width_mm: float = 120.0,
    height_mm: float = 30.0,
    dpi: float = 300.0,
    patch_size_mm: float = 12.0,
    combinations: tuple = ("CM", "CY", "MY", "CK", "MK", "YK", "CMY", "CMYK"),
) -> dict[str, np.ndarray]:
    """Overprint colour patches for colorimetric misregistration measurement."""
    CHANNEL_RGB = {
        "C": np.array([0, 255, 255]),
        "M": np.array([255, 0, 255]),
        "Y": np.array([255, 255, 0]),
        "K": np.array([0, 0, 0]),
    }
    patches = {}
    ps = _mm2px(patch_size_mm, dpi)
    ph = _mm2px(height_mm, dpi)
    for combo in combinations:
        patch = np.full((ph, ps, 3), 255, dtype=np.uint8)
        color = np.array([255, 255, 255])
        for ch in combo:
            c = CHANNEL_RGB.get(ch, np.array([128, 128, 128]))
            # subtractive blend approximation
            color = np.clip(color.astype(int) - (255 - c.astype(int)), 0, 255)
        patch[:] = color.astype(np.uint8)
        patches[combo] = patch
    return patches


def make_synthetic_misregistration_target(
    width_mm: float = 60.0,
    height_mm: float = 60.0,
    dpi: float = 300.0,
    shifts_px: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Generate a CMYK registration chart with known synthetic shifts applied.

    Returns dict: "reference" (zero shift) and per-channel shifted images.
    shifts_px: dict like {"C": (dx, dy), "M": (dx, dy), ...}
    """
    if shifts_px is None:
        shifts_px = {"C": (2.0, -1.0), "M": (-1.5, 2.0), "Y": (3.0, 0.5), "K": (0.0, 0.0)}

    import cv2

    # Build a composite reference (crosshair + bullseye)
    ref = _canvas(width_mm, height_mm, dpi, n_ch=3)
    h, w = ref.shape[:2]
    # Draw black crosshair on white
    cy, cx = h // 2, w // 2
    lw = max(1, _mm2px(0.5, dpi))
    ref[cy-lw:cy+lw, :] = 0
    ref[:, cx-lw:cx+lw] = 0

    results = {"reference": ref.copy()}
    for ch, (dx, dy) in shifts_px.items():
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(ref, M, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
        results[ch] = shifted
    return results


# ---------------------------------------------------------------------------
# master generator
# ---------------------------------------------------------------------------

def generate_all_targets(
    output_dir: str | Path,
    dpi: float = 300.0,
    save_tiff: bool = True,
    save_png: bool = True,
) -> dict:
    """Generate the full set of calibration targets and save to output_dir.

    Returns a manifest dict with target metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"dpi": dpi, "targets": []}

    def _save(img: np.ndarray, name: str, meta: dict):
        from PIL import Image
        base = output_dir / name
        pil = Image.fromarray(img)
        if save_png:
            pil.save(str(base) + ".png")
            manifest["targets"].append({**meta, "file": name + ".png"})
        if save_tiff:
            pil.save(str(base) + ".tiff", dpi=(dpi, dpi))

    _save(make_corner_fiducials(dpi=dpi), "corner_fiducials",
          {"type": "corner_fiducials", "channels": "all"})
    _save(make_bullseye(dpi=dpi), "bullseye",
          {"type": "bullseye", "channels": "all"})
    _save(make_crosshair(dpi=dpi), "crosshair",
          {"type": "crosshair", "channels": "all"})
    _save(make_line_targets(dpi=dpi), "line_targets",
          {"type": "line_targets", "channels": "all"})
    _save(make_slanted_edge(dpi=dpi), "slanted_edge_15",
          {"type": "slanted_edge", "angle_deg": 15})

    for ch, img in make_cmyk_registration_chart(dpi=dpi).items():
        _save(img, f"cmyk_chart_{ch}", {"type": "cmyk_chart", "channel": ch})

    for combo, patch in make_overprint_patches(dpi=dpi).items():
        _save(patch, f"overprint_{combo}", {"type": "overprint_patch", "combo": combo})

    syn = make_synthetic_misregistration_target(dpi=dpi)
    for key, img in syn.items():
        _save(img, f"synthetic_misreg_{key}",
              {"type": "synthetic_misregistration", "variant": key})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
