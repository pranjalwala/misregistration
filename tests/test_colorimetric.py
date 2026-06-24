"""Unit tests for colorimetric metrics."""
import numpy as np
import pytest
from misregistration.measurement.colorimetric import (
    lab_from_rgb, delta_e00, compute_delta_e_mis
)


def test_lab_white():
    white = np.array([[[255, 255, 255]]], dtype=np.uint8)
    lab = lab_from_rgb(white)
    assert abs(lab[0, 0, 0] - 100.0) < 3.0  # L* ≈ 100


def test_lab_black():
    black = np.array([[[0, 0, 0]]], dtype=np.uint8)
    lab = lab_from_rgb(black)
    assert abs(lab[0, 0, 0]) < 3.0  # L* ≈ 0


def test_delta_e_identical():
    lab = np.array([[[50.0, 10.0, -10.0]]], dtype=np.float32)
    dE = delta_e00(lab, lab)
    assert dE.item() == pytest.approx(0.0, abs=1e-4)


def test_delta_e_large():
    lab1 = np.array([[[50.0, 0.0, 0.0]]], dtype=np.float32)
    lab2 = np.array([[[50.0, 50.0, 0.0]]], dtype=np.float32)
    dE = delta_e00(lab1, lab2)
    assert dE.item() > 5.0


def test_compute_delta_e_mis():
    rgb1 = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb2 = np.full((10, 10, 3), 128, dtype=np.uint8)
    result = compute_delta_e_mis(rgb1, rgb2)
    assert "dE_image" in result
    assert "dE_map" in result
    assert result["dE_image"] > 0
    assert result["dE_map"].shape == (10, 10)
