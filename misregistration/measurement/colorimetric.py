"""CIELAB conversion and CIEDE2000-based colorimetric misregistration metrics."""

from __future__ import annotations
import numpy as np


def lab_from_rgb(rgb: np.ndarray, illuminant: str = "D50") -> np.ndarray:
    """Convert uint8 RGB image to CIELAB using colour-science library.

    Falls back to a pure-numpy sRGB→XYZ→Lab pipeline if colour is unavailable.

    Returns float32 array shape (..., 3) with L*, a*, b*.
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    try:
        import colour
        xyz = colour.sRGB_to_XYZ(rgb_f)
        lab = colour.XYZ_to_Lab(xyz, illuminant=colour.CCS_ILLUMINANTS[
            "CIE 1931 2 Degree Standard Observer"][illuminant])
        return lab.astype(np.float32)
    except Exception:
        return _numpy_rgb_to_lab(rgb_f, illuminant)


def delta_e00(
    lab1: np.ndarray,
    lab2: np.ndarray,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> np.ndarray:
    """Compute CIEDE2000 per pixel between two LAB arrays (any shape (...,3)).

    Returns float32 array of same spatial shape as input.
    """
    try:
        import colour
        return colour.delta_E(lab1, lab2, method="CIE 2000").astype(np.float32)
    except Exception:
        return _numpy_delta_e00(lab1, lab2, kL, kC, kH)


def compute_delta_e_mis(
    rgb_ref: np.ndarray,
    rgb_shifted: np.ndarray,
    illuminant: str = "D50",
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> dict:
    """Compute ΔE_mis for a patch or full image.

    Parameters
    ----------
    rgb_ref     : uint8 RGB reference (correct registration)
    rgb_shifted : uint8 RGB after applying channel shifts

    Returns dict with:
      dE_image   : float  – mean over all pixels
      dE_map     : ndarray – per-pixel ΔE map
    """
    lab_ref = lab_from_rgb(rgb_ref, illuminant)
    lab_shift = lab_from_rgb(rgb_shifted, illuminant)
    dE_map = delta_e00(lab_ref, lab_shift, kL=kL, kC=kC, kH=kH)
    return {
        "dE_image": float(dE_map.mean()),
        "dE_map": dE_map,
    }


# --------------------------------------------------------------------------- #
# pure-numpy fallback implementations
# --------------------------------------------------------------------------- #

def _numpy_rgb_to_lab(rgb_f: np.ndarray, illuminant: str = "D50") -> np.ndarray:
    """sRGB [0,1] → CIELAB (D50 or D65)."""
    # Linearise sRGB
    mask = rgb_f > 0.04045
    lin = np.where(mask, ((rgb_f + 0.055) / 1.055) ** 2.4, rgb_f / 12.92)

    # sRGB→XYZ (D65 matrix)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T

    # Adapt to illuminant
    if illuminant == "D50":
        # Bradford chromatic adaptation D65→D50
        MA = np.array([
            [1.0478112, 0.0228866, -0.0501270],
            [0.0295424, 0.9904844, -0.0170491],
            [-0.0092345, 0.0150436, 0.7521316],
        ], dtype=np.float32)
        xyz = xyz @ MA.T
        white = np.array([0.96422, 1.00000, 0.82521], dtype=np.float32)
    else:
        white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    xyz_n = xyz / white

    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta**3,
                        t ** (1.0 / 3.0),
                        t / (3 * delta**2) + 4.0 / 29.0)

    fx = f(xyz_n[..., 0])
    fy = f(xyz_n[..., 1])
    fz = f(xyz_n[..., 2])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _numpy_delta_e00(
    lab1: np.ndarray,
    lab2: np.ndarray,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> np.ndarray:
    """CIEDE2000 (pure numpy). Input shape (..., 3)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Step 1: compute C'
    C1ab = np.sqrt(a1**2 + b1**2)
    C2ab = np.sqrt(a2**2 + b2**2)
    Cab_mean7 = ((C1ab + C2ab) / 2.0) ** 7
    G = 0.5 * (1 - np.sqrt(Cab_mean7 / (Cab_mean7 + 25**7)))
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    # Step 2: h'
    def hprime(a, b):
        h = np.degrees(np.arctan2(b, a)) % 360
        return h

    h1p = hprime(a1p, b1)
    h2p = hprime(a2p, b2)

    # Step 3: deltas
    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = np.where(
        np.abs(h2p - h1p) <= 180, h2p - h1p,
        np.where(h2p <= h1p, h2p - h1p + 360, h2p - h1p - 360)
    )
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    # Step 4: means
    Lp_mean = (L1 + L2) / 2
    Cp_mean = (C1p + C2p) / 2
    hp_mean = np.where(
        np.abs(h1p - h2p) <= 180, (h1p + h2p) / 2,
        np.where(h1p + h2p < 360, (h1p + h2p + 360) / 2, (h1p + h2p - 360) / 2)
    )

    # Step 5: weighting
    T = (1
         - 0.17 * np.cos(np.radians(hp_mean - 30))
         + 0.24 * np.cos(np.radians(2 * hp_mean))
         + 0.32 * np.cos(np.radians(3 * hp_mean + 6))
         - 0.20 * np.cos(np.radians(4 * hp_mean - 63)))
    SL = 1 + 0.015 * (Lp_mean - 50)**2 / np.sqrt(20 + (Lp_mean - 50)**2)
    SC = 1 + 0.045 * Cp_mean
    SH = 1 + 0.015 * Cp_mean * T

    Cp_mean7 = Cp_mean ** 7
    RC = 2 * np.sqrt(Cp_mean7 / (Cp_mean7 + 25**7))
    d_theta = 30 * np.exp(-((hp_mean - 275) / 25)**2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    dE = np.sqrt(
        (dLp / (kL * SL))**2 +
        (dCp / (kC * SC))**2 +
        (dHp / (kH * SH))**2 +
        RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return dE.astype(np.float32)
