"""RGB → CMYK separation utilities (device-independent)."""

from __future__ import annotations
import numpy as np


def rgb_to_cmyk_separations(
    rgb: np.ndarray,
    ucr: float = 0.85,
) -> dict[str, np.ndarray]:
    """Convert uint8 RGB image to four float32 [0,1] CMYK separations.

    This is a simple device-independent RGB→CMY inversion plus UCR for K.
    For physical print evaluation the ICC profile path should be used instead.
    UCR (Under Color Removal) controls how much CMY is replaced by K.

    Returns dict with keys "C", "M", "Y", "K", values float32 [0,1].
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    R, G, B = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]

    C_raw = 1.0 - R
    M_raw = 1.0 - G
    Y_raw = 1.0 - B
    K = np.minimum(np.minimum(C_raw, M_raw), Y_raw) * ucr

    C = np.clip(C_raw - K, 0, 1)
    M = np.clip(M_raw - K, 0, 1)
    Y = np.clip(Y_raw - K, 0, 1)
    K = np.clip(K, 0, 1)

    return {"C": C, "M": M, "Y": Y, "K": K}


def extract_channel(
    rgb_scan: np.ndarray,
    channel: str,
) -> np.ndarray:
    """Extract a single-channel proxy from an RGB scanned image.

    Scanner convention (as in spec pseudocode):
      C → Red channel (scanner R absorbs cyan ink)
      M → Green channel
      Y → Blue channel
      K → luminance complement

    Returns float32 [0,1].
    """
    rgb_f = rgb_scan.astype(np.float32) / 255.0
    ch = channel.upper()
    if ch == "C":
        return rgb_f[..., 0]
    elif ch == "M":
        return rgb_f[..., 1]
    elif ch == "Y":
        return rgb_f[..., 2]
    elif ch == "K":
        # K proxy: 1 - mean(R,G,B)
        return 1.0 - rgb_f.mean(axis=2)
    else:
        raise ValueError(f"Unknown channel '{channel}'. Expected C/M/Y/K.")


def cmyk_to_rgb(separations: dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct uint8 RGB from float32 [0,1] CMYK separations."""
    C = separations["C"]
    M = separations["M"]
    Y = separations["Y"]
    K = separations["K"]
    R = (1 - C) * (1 - K)
    G = (1 - M) * (1 - K)
    B = (1 - Y) * (1 - K)
    rgb = np.stack([R, G, B], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
