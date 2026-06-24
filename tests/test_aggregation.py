"""Unit tests for aggregation statistics."""
import math
import pytest
from misregistration.aggregation.statistics import (
    compute_ci, aggregate_replicates, build_per_channel_table,
    build_per_method_table, export_csv
)
import tempfile, csv
from pathlib import Path


def test_compute_ci_single():
    mean, ci = compute_ci([5.0])
    assert mean == 5.0
    assert math.isnan(ci)


def test_compute_ci_multiple():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, ci = compute_ci(vals)
    assert abs(mean - 3.0) < 1e-9
    assert ci > 0


def test_compute_ci_empty():
    mean, ci = compute_ci([])
    assert math.isnan(mean)


def test_aggregate_replicates():
    agg = aggregate_replicates(
        rms_px_list=[2.0, 2.2, 1.9],
        rms_mm_list=[0.17, 0.18, 0.16],
        dx_list=[1.5, 1.6, 1.4],
        dy_list=[-0.8, -0.9, -0.7],
        dE_list=[3.0, 3.2, 2.9],
        channel="C", method="DBS",
    )
    assert agg.channel == "C"
    assert agg.method == "DBS"
    assert abs(agg.mean_rms_px - 2.033) < 0.1
    assert agg.n_obs == 3


def test_per_channel_table():
    agg = aggregate_replicates(
        [1.0], [0.08], [1.0], [0.0], [2.5], "M", "ErrorDiffusion"
    )
    rows = build_per_channel_table([agg])
    assert len(rows) == 1
    assert rows[0]["channel"] == "M"


def test_export_csv():
    agg = aggregate_replicates([1.0], [0.08], [1.0], [0.0], [2.5], "Y", "test")
    rows = build_per_channel_table([agg])
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "out.csv"
        export_csv(rows, path)
        assert path.exists()
        content = list(csv.DictReader(open(path)))
        assert len(content) == 1
        assert content[0]["channel"] == "Y"
