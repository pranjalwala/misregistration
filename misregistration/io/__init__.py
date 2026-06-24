"""TIFF/image loading and metadata extraction."""
from .loader import load_image, load_tiff, extract_metadata, ScanMetadata

__all__ = ["load_image", "load_tiff", "extract_metadata", "ScanMetadata"]
