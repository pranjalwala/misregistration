"""Tests for Pareto frontier analysis."""
import tempfile
from pathlib import Path
import pytest
from misregistration.aggregation import (
    compute_pareto_frontier, export_pareto_csv, ParetoPoint
)


def test_pareto_simple():
    methods = ["A", "B", "C", "D"]
    rms    = [0.1, 0.2, 0.1, 0.3]
    dE     = [3.0, 2.0, 2.5, 1.5]
    pts = compute_pareto_frontier(methods, rms, dE)
    assert len(pts) == 4
    pareto = [p for p in pts if p.is_pareto]
    # A: (0.1, 3.0) — dominated by C (0.1, 2.5)
    # B: (0.2, 2.0) — dominated by C (0.1, 2.5)? No: C has lower dE only if 2.5<2.0, false.
    # C: (0.1, 2.5) — not dominated
    # D: (0.3, 1.5) — not dominated by C (rms worse, dE better)
    pareto_names = {p.method for p in pareto}
    assert "C" in pareto_names
    assert "D" in pareto_names


def test_pareto_all_dominated_except_one():
    methods = ["Best", "Worse1", "Worse2"]
    rms    = [0.05, 0.10, 0.20]
    dE     = [1.0,  2.0,  3.0]
    pts = compute_pareto_frontier(methods, rms, dE)
    pareto = [p for p in pts if p.is_pareto]
    assert len(pareto) == 1
    assert pareto[0].method == "Best"


def test_export_pareto_csv():
    pts = [ParetoPoint("A", 0.1, 2.5, True), ParetoPoint("B", 0.2, 1.5, True)]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pareto.csv"
        export_pareto_csv(pts, path)
        assert path.exists()
        content = path.read_text()
        assert "method" in content
        assert "A" in content
