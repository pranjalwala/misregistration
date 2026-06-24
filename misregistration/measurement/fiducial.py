"""Fiducial mark detection, connected-component extraction, centroid estimation."""

from __future__ import annotations
import numpy as np
from typing import Optional


def detect_fiducials(
    channel_image: np.ndarray,
    threshold_method: str = "otsu",
    threshold_fixed: int = 128,
    min_area: int = 20,
    max_area: int = 50000,
) -> list[dict]:
    """Detect registration mark blobs in a single-channel image.

    Parameters
    ----------
    channel_image    : uint8 or float32 [0,1] single-channel image
    threshold_method : "otsu" | "adaptive" | "fixed"
    min_area / max_area : blob area filter in pixels

    Returns
    -------
    list of dicts with keys: centroid_x, centroid_y, area, bbox
    """
    import cv2
    from skimage.filters import threshold_otsu, threshold_local
    from skimage.measure import label, regionprops

    img = _to_uint8(channel_image)

    if threshold_method == "otsu":
        thresh_val = threshold_otsu(img)
        binary = (img < thresh_val).astype(np.uint8)  # dark marks on light background
    elif threshold_method == "adaptive":
        thresh_map = threshold_local(img, block_size=35, offset=10)
        binary = (img < thresh_map).astype(np.uint8)
    else:
        binary = (img < threshold_fixed).astype(np.uint8)

    labeled = label(binary)
    marks = []
    for prop in regionprops(labeled):
        if prop.area < min_area or prop.area > max_area:
            continue
        cy, cx = prop.centroid
        marks.append({
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "area": int(prop.area),
            "bbox": prop.bbox,  # (min_row, min_col, max_row, max_col)
            "label": int(prop.label),
        })
    return marks


def estimate_centroid(
    region: np.ndarray,
    offset: tuple[int, int] = (0, 0),
) -> tuple[float, float]:
    """Sub-pixel centroid of a binary or grayscale region.

    offset : (x0, y0) top-left corner of the region in the full image.
    """
    arr = region.astype(np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    # Invert so ink=1
    arr = 1.0 - arr if arr.mean() > 0.5 else arr
    total = arr.sum()
    if total < 1e-8:
        h, w = region.shape[:2]
        return float(offset[0] + w / 2), float(offset[1] + h / 2)
    ys, xs = np.mgrid[0:region.shape[0], 0:region.shape[1]]
    cx = float((xs * arr).sum() / total) + offset[0]
    cy = float((ys * arr).sum() / total) + offset[1]
    return cx, cy


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return (img * 255).astype(np.uint8)
    return img.astype(np.uint8)
