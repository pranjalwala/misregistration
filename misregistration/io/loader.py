"""Hardware-agnostic image and TIFF loader with metadata extraction."""

from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np


@dataclasses.dataclass
class ScanMetadata:
    """Device-independent metadata extracted from TIFF tags or fallback."""
    dpi_x: float
    dpi_y: float
    bit_depth: int
    width: int
    height: int
    n_channels: int
    color_space: Optional[str] = None
    source_path: Optional[str] = None
    extra_tags: dict = dataclasses.field(default_factory=dict)

    @property
    def px_per_mm(self) -> float:
        """Pixels per millimetre in x direction."""
        return self.dpi_x / 25.4

    @property
    def mm_per_px(self) -> float:
        return 25.4 / self.dpi_x


def load_tiff(path: str | Path, fallback_dpi: float = 300.0) -> tuple[np.ndarray, ScanMetadata]:
    """Load a TIFF file; extract resolution from tags if present.

    Returns
    -------
    image : ndarray  uint8 or uint16, shape (H, W) or (H, W, C)
    meta  : ScanMetadata
    """
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            image = tf.asarray()
            page = tf.pages[0]
            dpi_x = _read_tiff_dpi(page, axis=0, fallback=fallback_dpi)
            dpi_y = _read_tiff_dpi(page, axis=1, fallback=fallback_dpi)
            bit_depth = page.bitspersample
            extra = {}
            for tag in page.tags.values():
                extra[tag.name] = tag.value
    except ImportError:
        # tifffile not available: fall back to PIL
        image, dpi_x, dpi_y, bit_depth, extra = _load_via_pil(path, fallback_dpi)

    image = _normalise_image(image)
    h, w = image.shape[:2]
    nc = 1 if image.ndim == 2 else image.shape[2]
    meta = ScanMetadata(
        dpi_x=dpi_x, dpi_y=dpi_y, bit_depth=bit_depth,
        width=w, height=h, n_channels=nc,
        source_path=str(path), extra_tags=extra
    )
    return image, meta


def load_image(path: str | Path, fallback_dpi: float = 300.0) -> tuple[np.ndarray, ScanMetadata]:
    """Load any supported image format (TIFF, PNG, JPEG, etc.)."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        return load_tiff(path, fallback_dpi)
    return _load_via_pil(path, fallback_dpi, return_tuple=True)


def extract_metadata(path: str | Path, fallback_dpi: float = 300.0) -> ScanMetadata:
    """Extract only metadata without keeping the full image array in memory."""
    _, meta = load_image(path, fallback_dpi)
    return meta


# --------------------------------------------------------------------------- #
# private helpers
# --------------------------------------------------------------------------- #

def _read_tiff_dpi(page, axis: int, fallback: float) -> float:
    """Read XResolution (axis=0) or YResolution (axis=1) from a tifffile page."""
    try:
        import tifffile
        tag_id = 282 if axis == 0 else 283   # XResolution / YResolution
        tag = page.tags.get(tag_id)
        if tag is None:
            return fallback
        val = tag.value
        if isinstance(val, tuple):
            num, den = val
            return float(num) / float(den) if den != 0 else fallback
        return float(val)
    except Exception:
        return fallback


def _load_via_pil(path: str | Path, fallback_dpi: float, return_tuple: bool = False):
    from PIL import Image
    img = Image.open(str(path))
    info = img.info
    dpi_x = dpi_y = fallback_dpi
    if "dpi" in info:
        try:
            dpi_x, dpi_y = float(info["dpi"][0]), float(info["dpi"][1])
        except Exception:
            pass
    arr = np.array(img)
    bit_depth = 8 if arr.dtype == np.uint8 else 16
    extra = dict(info)
    if return_tuple:
        arr = _normalise_image(arr)
        h, w = arr.shape[:2]
        nc = 1 if arr.ndim == 2 else arr.shape[2]
        meta = ScanMetadata(
            dpi_x=dpi_x, dpi_y=dpi_y, bit_depth=bit_depth,
            width=w, height=h, n_channels=nc,
            source_path=str(path), extra_tags=extra
        )
        return arr, meta
    return arr, dpi_x, dpi_y, bit_depth, extra


def _normalise_image(arr: np.ndarray) -> np.ndarray:
    """Ensure array is uint8 (scale 16-bit down) and drop alpha if present."""
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    # drop alpha
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr
