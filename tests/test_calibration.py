"""Tests for calibration modules."""
import numpy as np
import pytest
from misregistration.calibration import (
    PixelScale, scale_from_dpi, scale_from_target,
    SessionBaseline, null_baseline, apply_baseline_correction
)


def test_scale_from_dpi():
    sc = scale_from_dpi(300.0)
    assert abs(sc.px_per_mm - 300/25.4) < 0.001
    assert sc.source == "argument"


def test_scale_from_target():
    a = (0.0, 0.0)
    b = (100.0, 0.0)
    sc = scale_from_target(a, b, known_distance_mm=10.0)
    assert abs(sc.px_per_mm - 10.0) < 0.001


def test_px_to_mm_roundtrip():
    sc = scale_from_dpi(600.0)
    mm = sc.px_to_mm(sc.mm_to_px(5.0))
    assert abs(mm - 5.0) < 1e-6


def test_null_baseline():
    bl = null_baseline()
    assert bl.correction("C") == (0.0, 0.0)


def test_apply_baseline_correction():
    bl = SessionBaseline(baselines={"C": (1.5, -0.5)})
    dx, dy = apply_baseline_correction(3.0, 2.0, "C", bl)
    assert abs(dx - 1.5) < 1e-9
    assert abs(dy - 2.5) < 1e-9
