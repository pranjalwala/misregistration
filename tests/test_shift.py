"""Unit tests for shift estimation and RMS computation."""
import numpy as np
import pytest
from misregistration.measurement.shift import rms_shift, shifts_from_centroids


def test_rms_shift_simple():
    dx = [3.0, 4.0]
    dy = [4.0, 3.0]
    rms = rms_shift(dx, dy)
    assert abs(rms - 5.0) < 1e-6


def test_rms_shift_zero():
    assert rms_shift([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)


def test_rms_shift_empty():
    assert np.isnan(rms_shift([], []))


def test_shifts_from_centroids():
    k = [(10.0, 20.0), (50.0, 60.0)]
    c = [(11.5, 19.0), (51.5, 61.0)]
    dx, dy = shifts_from_centroids(k, c)
    assert dx == pytest.approx([1.5, 1.5])
    assert dy == pytest.approx([-1.0, 1.0])
