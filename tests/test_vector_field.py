"""Tests for vector field visualization."""
import numpy as np
import tempfile
from pathlib import Path
from misregistration.visualization import (
    plot_displacement_overlay, plot_shift_heatmap,
    plot_multichannel_shifts, build_local_shift_grid,
)


def _rng_rgb(size=64):
    rng = np.random.default_rng(1)
    return rng.integers(40, 220, (size, size, 3), dtype=np.uint8)


def test_displacement_overlay():
    ref = _rng_rgb()
    shifts = {"C": (2.0, -1.0), "M": (-1.0, 1.5)}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "overlay.png"
        fig = plot_displacement_overlay(ref, shifts, output_path=p)
        assert p.exists()


def test_shift_heatmap():
    rms_map = np.random.default_rng(2).random((16, 16)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "heatmap.png"
        fig = plot_shift_heatmap(rms_map, channel="C", output_path=p)
        assert p.exists()


def test_multichannel_shifts():
    shifts_list = [{"C": (1.0+i*0.1, -0.5), "M": (-0.5, 1.0+i*0.05), "Y": (2.0, 0.3)}
                   for i in range(5)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "multi.png"
        fig = plot_multichannel_shifts(shifts_list, output_path=p)
        assert p.exists()


def test_build_local_shift_grid():
    import cv2
    ref = np.random.default_rng(3).random((64, 64)).astype(np.float32)
    M = np.float32([[1, 0, 2.0], [0, 1, -1.0]])
    shifted = cv2.warpAffine(ref, M, (64, 64))
    grid = build_local_shift_grid(ref, shifted, grid_rows=2, grid_cols=2, max_shift=10)
    assert grid.shape == (2, 2, 2)
