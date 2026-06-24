"""Unit tests for RGB→CMYK separation."""
import numpy as np
import pytest
from misregistration.preprocessing.separation import (
    rgb_to_cmyk_separations, extract_channel, cmyk_to_rgb
)


def test_white_gives_zero_ink():
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    seps = rgb_to_cmyk_separations(white)
    for ch in ("C", "M", "Y", "K"):
        assert seps[ch].max() < 0.05, f"{ch} should be near 0 for white"


def test_black_gives_k_ink():
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    seps = rgb_to_cmyk_separations(black)
    assert seps["K"].mean() > 0.5


def test_roundtrip_close():
    rng = np.random.default_rng(42)
    rgb = rng.integers(50, 200, (32, 32, 3), dtype=np.uint8)
    seps = rgb_to_cmyk_separations(rgb)
    recon = cmyk_to_rgb(seps)
    diff = np.abs(rgb.astype(float) - recon.astype(float))
    assert diff.mean() < 20.0  # loose tolerance due to UCR


def test_extract_channel_shapes():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    for ch in ("C", "M", "Y", "K"):
        out = extract_channel(rgb, ch)
        assert out.shape == (8, 8)
        assert out.dtype == np.float32
