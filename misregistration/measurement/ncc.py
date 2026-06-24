"""Normalised Cross-Correlation and sub-pixel peak localisation.

Implements Equations 3, 4 and the Gaussian/parabolic peak refinement
described in the benchmark specification.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve


def compute_ncc(
    reference: np.ndarray,
    template: np.ndarray,
    max_shift: int = 50,
) -> np.ndarray:
    """Compute the normalised cross-correlation surface (Eq. 3).

    Parameters
    ----------
    reference : 2-D float32 array  (digital separation)
    template  : 2-D float32 array  (scanned separation, same size)
    max_shift : search radius in pixels

    Returns
    -------
    ncc_map : 2-D float32, shape (2*max_shift+1, 2*max_shift+1)
              ncc_map[i,j] = NCC at shift (i - max_shift, j - max_shift)
    """
    ref = reference.astype(np.float64)
    tmpl = template.astype(np.float64)

    # Zero-mean
    ref = ref - ref.mean()
    tmpl = tmpl - tmpl.mean()

    ref_std = ref.std()
    tmpl_std = tmpl.std()
    if ref_std < 1e-8 or tmpl_std < 1e-8:
        # Uniform patch – return zero surface
        sz = 2 * max_shift + 1
        return np.zeros((sz, sz), dtype=np.float32)

    ref /= (ref_std * ref.size)
    tmpl /= tmpl_std

    # Full cross-correlation via FFT
    # We only need the central (2*max_shift+1)^2 window
    corr_full = fftconvolve(tmpl, ref[::-1, ::-1], mode="full")
    cy, cx = np.array(corr_full.shape) // 2
    lo = -max_shift
    hi = max_shift + 1
    ncc_map = corr_full[cy + lo: cy + hi, cx + lo: cx + hi]
    return ncc_map.astype(np.float32)


def find_peak_subpixel(
    ncc_map: np.ndarray,
    method: str = "gaussian",
) -> tuple[float, float, float]:
    """Locate the NCC peak with sub-pixel precision (Eq. 4 + refinement).

    Parameters
    ----------
    ncc_map : 2-D NCC surface from compute_ncc
    method  : "gaussian" | "parabolic"

    Returns
    -------
    (dx, dy, peak_value)
      dx, dy in pixels relative to zero-shift (centre of ncc_map)
      peak_value : NCC value at integer peak
    """
    max_shift = ncc_map.shape[0] // 2
    flat_idx = np.argmax(ncc_map)
    py, px = np.unravel_index(flat_idx, ncc_map.shape)
    peak_val = float(ncc_map[py, px])

    # Sub-pixel refinement on the 3x3 neighbourhood
    if py == 0 or py == ncc_map.shape[0] - 1 or px == 0 or px == ncc_map.shape[1] - 1:
        # Peak is on boundary – return integer shift
        return float(px - max_shift), float(py - max_shift), peak_val

    patch = ncc_map[py-1:py+2, px-1:px+2].astype(np.float64)

    if method == "gaussian":
        sub_x, sub_y = _gaussian_subpixel(patch)
    else:
        sub_x, sub_y = _parabolic_subpixel(patch)

    dx = (px - max_shift) + sub_x
    dy = (py - max_shift) + sub_y
    return float(dx), float(dy), peak_val


# --------------------------------------------------------------------------- #
# private: analytic sub-pixel formulas from spec
# --------------------------------------------------------------------------- #

def _parabolic_subpixel(patch: np.ndarray) -> tuple[float, float]:
    """Parabolic (quadratic) sub-pixel fit on 3x3 patch.
    μ_x = 0.5 * (log f(-1) - log f(+1)) / (log f(-1) - 2 log f(0) + log f(+1))
    but without logs (standard parabola formula).
    """
    # x direction: row index 1 (centre row)
    f_xm, f_x0, f_xp = patch[1, 0], patch[1, 1], patch[1, 2]
    # y direction: col index 1 (centre col)
    f_ym, f_y0, f_yp = patch[0, 1], patch[1, 1], patch[2, 1]

    sub_x = _parabola_peak(f_xm, f_x0, f_xp)
    sub_y = _parabola_peak(f_ym, f_y0, f_yp)
    return sub_x, sub_y


def _gaussian_subpixel(patch: np.ndarray) -> tuple[float, float]:
    """Gaussian sub-pixel fit using log of NCC values (spec Eq.)."""
    eps = 1e-12
    patch = np.clip(patch, eps, None)
    log_patch = np.log(patch)

    lxm, lx0, lxp = log_patch[1, 0], log_patch[1, 1], log_patch[1, 2]
    lym, ly0, lyp = log_patch[0, 1], log_patch[1, 1], log_patch[2, 1]

    sub_x = _parabola_peak(lxm, lx0, lxp)
    sub_y = _parabola_peak(lym, ly0, lyp)
    return sub_x, sub_y


def _parabola_peak(fm: float, f0: float, fp: float) -> float:
    denom = fm - 2 * f0 + fp
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (fm - fp) / denom
