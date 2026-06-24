"""Unit tests for NCC and sub-pixel peak localisation."""
import numpy as np
import pytest
from misregistration.measurement.ncc import compute_ncc, find_peak_subpixel


def _gaussian_blob(size=64, cx=32, cy=32, sigma=4.0):
    ys, xs = np.mgrid[:size, :size]
    return np.exp(-((xs-cx)**2 + (ys-cy)**2) / (2*sigma**2)).astype(np.float32)


def test_ncc_identical_images():
    img = _gaussian_blob()
    ncc = compute_ncc(img, img, max_shift=10)
    assert ncc.shape == (21, 21)
    peak_idx = np.argmax(ncc)
    py, px = np.unravel_index(peak_idx, ncc.shape)
    assert py == 10 and px == 10  # centre = zero shift


def test_ncc_known_shift():
    ref = _gaussian_blob(size=64, cx=32, cy=32)
    shifted = np.zeros_like(ref)
    dx_true, dy_true = 3, -2
    shifted = np.roll(np.roll(ref, dy_true, axis=0), dx_true, axis=1)
    ncc = compute_ncc(ref, shifted, max_shift=10)
    dx_est, dy_est, peak = find_peak_subpixel(ncc)
    assert abs(dx_est - dx_true) < 1.0, f"dx error too large: {dx_est}"
    assert abs(dy_est - dy_true) < 1.0, f"dy error too large: {dy_est}"
    assert peak > 0.5


def test_ncc_zero_input():
    zero = np.zeros((32, 32), dtype=np.float32)
    ncc = compute_ncc(zero, zero, max_shift=5)
    assert ncc.shape == (11, 11)
    assert np.all(ncc == 0)


def test_subpixel_peak_centre():
    ncc = np.zeros((11, 11), dtype=np.float32)
    ncc[5, 5] = 1.0
    ncc[4, 5] = ncc[6, 5] = ncc[5, 4] = ncc[5, 6] = 0.5
    ncc[4, 4] = ncc[4, 6] = ncc[6, 4] = ncc[6, 6] = 0.25
    dx, dy, peak = find_peak_subpixel(ncc)
    assert abs(dx) < 0.1
    assert abs(dy) < 0.1
    assert peak == pytest.approx(1.0)
