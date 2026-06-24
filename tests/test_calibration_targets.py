"""Tests for calibration target generation."""
import numpy as np
import tempfile
from pathlib import Path
from misregistration.calibration_targets import (
    make_corner_fiducials, make_bullseye, make_crosshair,
    make_line_targets, make_slanted_edge,
    make_cmyk_registration_chart, make_overprint_patches,
    make_synthetic_misregistration_target, generate_all_targets,
)


def test_corner_fiducials_shape():
    img = make_corner_fiducials(dpi=72)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    # should have black pixels
    assert img.min() == 0


def test_bullseye_has_black():
    img = make_bullseye(dpi=72)
    assert img.min() == 0


def test_crosshair_has_black():
    img = make_crosshair(dpi=72)
    assert img.min() == 0


def test_cmyk_chart_returns_dict():
    charts = make_cmyk_registration_chart(dpi=72)
    for ch in ("C", "M", "Y", "K"):
        assert ch in charts
        assert charts[ch].ndim == 3


def test_overprint_patches():
    patches = make_overprint_patches(dpi=72)
    assert "CM" in patches
    assert "CMYK" in patches


def test_synthetic_misregistration():
    targets = make_synthetic_misregistration_target(dpi=72)
    assert "reference" in targets
    assert "C" in targets


def test_generate_all_targets():
    with tempfile.TemporaryDirectory() as td:
        manifest = generate_all_targets(td, dpi=72, save_tiff=False, save_png=True)
        assert "targets" in manifest
        assert len(manifest["targets"]) > 5
        assert (Path(td) / "manifest.json").exists()
