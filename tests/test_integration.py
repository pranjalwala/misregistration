"""End-to-end integration test: full pipeline on a synthetic image."""
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image


def _make_test_image(path: Path, size: int = 128) -> None:
    rng = np.random.default_rng(7)
    rgb = rng.integers(30, 230, (size, size, 3), dtype=np.uint8)
    # Add some structure
    rgb[20:60, 20:60, 0] = 220
    rgb[70:110, 70:110, 2] = 200
    Image.fromarray(rgb).save(str(path))


def test_full_pipeline():
    """Load image → simulate → measure → aggregate → export CSV."""
    from misregistration.simulation import simulate_from_png
    from misregistration.aggregation import (
        aggregate_replicates, build_per_channel_table, export_csv
    )
    from misregistration.visualization import save_summary_figure

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_path = td / "test.png"
        _make_test_image(img_path)

        shifts = {"C": (2.0, -1.0), "M": (-1.0, 1.5), "Y": (1.5, 0.5), "K": (0.0, 0.0)}
        result = simulate_from_png(img_path, shifts_px=shifts)

        assert result.dE_image >= 0
        assert result.dE_map.shape == (128, 128)
        for ch in ("C", "M", "Y"):
            assert ch in result.rms_px

        all_agg = []
        for ch in ("C", "M", "Y"):
            agg = aggregate_replicates(
                [result.rms_px[ch]], [float("nan")],
                [shifts[ch][0]], [shifts[ch][1]],
                [result.dE_image], ch, "integration_test"
            )
            all_agg.append(agg)

        rows = build_per_channel_table(all_agg)
        csv_path = td / "integration.csv"
        export_csv(rows, csv_path)
        assert csv_path.exists()

        figs = save_summary_figure(
            {ch: result.rms_px[ch] for ch in ("C","M","Y")},
            {ch: shifts[ch] for ch in ("C","M","Y")},
            result.dE_map,
            td / "plots",
        )
        assert len(figs) >= 2


def test_imports_all_public():
    """Verify all public imports work."""
    from misregistration.measurement import (
        compute_ncc, find_peak_subpixel,
        estimate_shift, rms_shift,
        compute_delta_e_mis, lab_from_rgb, delta_e00,
        detect_fiducials, estimate_centroid,
    )
    from misregistration.calibration import (
        PixelScale, scale_from_metadata, scale_from_target,
        SessionBaseline, measure_baseline, apply_baseline_correction,
    )
    from misregistration.aggregation import (
        aggregate_replicates, compute_ci,
        build_per_channel_table, build_per_method_table,
        export_csv, AggregatedResult,
    )
    from misregistration.visualization import (
        plot_rms_bar, plot_shift_vectors, plot_dE_map,
        plot_ci_bars, plot_channel_overlay, plot_ncc_surface,
        plot_histogram, save_summary_figure,
    )
    from misregistration.simulation import (
        ChannelShiftSimulator, simulate_from_png,
        simulate_from_jpg, simulate_from_folder, SimulationResult,
    )
    from misregistration.calibration_targets import (
        generate_all_targets, make_corner_fiducials, make_bullseye,
        make_crosshair, make_line_targets, make_slanted_edge,
        make_cmyk_registration_chart, make_overprint_patches,
        make_synthetic_misregistration_target,
    )
