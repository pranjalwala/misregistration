"""Preprocessing: geometric alignment, channel separation, normalisation."""
from .alignment import align_to_reference, crop_to_roi
from .separation import rgb_to_cmyk_separations, extract_channel

__all__ = [
    "align_to_reference", "crop_to_roi",
    "rgb_to_cmyk_separations", "extract_channel",
]
