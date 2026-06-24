"""Geometric alignment of scanned images to digital reference."""

from __future__ import annotations
import numpy as np


def align_to_reference(
    scan: np.ndarray,
    reference: np.ndarray,
    method: str = "ecc",
) -> tuple[np.ndarray, np.ndarray]:
    """Align *scan* to *reference* using ECC or feature-based matching.

    Parameters
    ----------
    scan      : uint8 RGB or grayscale scanned image
    reference : uint8 RGB or grayscale digital reference (same size approx.)
    method    : "ecc" (default) | "feature"

    Returns
    -------
    aligned : np.ndarray  – scan warped onto reference grid
    M       : np.ndarray  – 2×3 affine warp matrix
    """
    import cv2

    ref_gray = _to_gray(reference)
    scan_gray = _to_gray(scan)

    # Resize scan to match reference if needed
    if ref_gray.shape != scan_gray.shape:
        scan_gray = cv2.resize(scan_gray, (ref_gray.shape[1], ref_gray.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

    if method == "ecc":
        M = _ecc_align(ref_gray, scan_gray)
    else:
        M = _feature_align(ref_gray, scan_gray)

    h, w = reference.shape[:2]
    if scan.ndim == 3:
        aligned = cv2.warpAffine(scan, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    else:
        aligned = cv2.warpAffine(scan_gray, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    return aligned, M


def crop_to_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop image to region of interest (x, y, w, h)."""
    x, y, w, h = roi
    return image[y:y+h, x:x+w]


# --------------------------------------------------------------------------- #
# private
# --------------------------------------------------------------------------- #

def _to_gray(img: np.ndarray) -> np.ndarray:
    import cv2
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _ecc_align(ref: np.ndarray, src: np.ndarray) -> np.ndarray:
    import cv2
    M = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, M = cv2.findTransformECC(
            ref.astype(np.float32), src.astype(np.float32),
            M, cv2.MOTION_EUCLIDEAN, criteria,
        )
    except cv2.error:
        # ECC failed – return identity
        pass
    return M


def _feature_align(ref: np.ndarray, src: np.ndarray) -> np.ndarray:
    import cv2
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(src, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.eye(2, 3, dtype=np.float32)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    if len(matches) < 4:
        return np.eye(2, 3, dtype=np.float32)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    M, _ = cv2.estimateAffinePartial2D(pts2, pts1,
                                        method=cv2.RANSAC,
                                        ransacReprojThreshold=3.0)
    if M is None:
        return np.eye(2, 3, dtype=np.float32)
    return M
