"""Pareto frontier analysis for misregistration benchmark results.

Identifies methods that are non-dominated across RMS shift and ΔE_mis.
A method is Pareto-optimal if no other method is simultaneously better
on ALL metrics.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import NamedTuple
import numpy as np


class ParetoPoint(NamedTuple):
    method: str
    rms_mm: float
    dE_image: float
    is_pareto: bool


def compute_pareto_frontier(
    methods: list[str],
    rms_mm_values: list[float],
    dE_values: list[float],
) -> list[ParetoPoint]:
    """Identify the Pareto-optimal set.

    Lower RMS and lower ΔE are both better (minimisation problem).

    Returns list of ParetoPoint sorted by rms_mm ascending.
    """
    n = len(methods)
    points = list(zip(methods, rms_mm_values, dE_values))
    is_pareto = [True] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is <= on all objectives and < on at least one
            if (points[j][1] <= points[i][1] and points[j][2] <= points[i][2] and
                    (points[j][1] < points[i][1] or points[j][2] < points[i][2])):
                is_pareto[i] = False
                break

    results = [
        ParetoPoint(m, r, d, p)
        for (m, r, d), p in zip(points, is_pareto)
    ]
    return sorted(results, key=lambda x: x.rms_mm)


def export_pareto_csv(
    points: list[ParetoPoint],
    path: str | Path,
) -> None:
    """Export Pareto analysis to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "rms_mm", "dE_image", "is_pareto"])
        writer.writeheader()
        for p in points:
            writer.writerow({
                "method": p.method,
                "rms_mm": f"{p.rms_mm:.4f}",
                "dE_image": f"{p.dE_image:.4f}",
                "is_pareto": str(p.is_pareto),
            })


def plot_pareto_frontier(
    points: list[ParetoPoint],
    title: str = "Pareto Frontier: RMS Shift vs ΔE",
    output_path: str | Path | None = None,
):
    """Plot Pareto frontier scatter with dominated/non-dominated markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pareto = [p for p in points if p.is_pareto]
    dominated = [p for p in points if not p.is_pareto]

    fig, ax = plt.subplots(figsize=(7, 5))

    if dominated:
        ax.scatter([p.rms_mm for p in dominated],
                   [p.dE_image for p in dominated],
                   color="lightgray", s=80, zorder=3, label="Dominated")

    if pareto:
        ax.scatter([p.rms_mm for p in pareto],
                   [p.dE_image for p in pareto],
                   color="steelblue", s=120, zorder=4, marker="*",
                   label="Pareto-optimal")
        # Draw staircase frontier line
        pareto_sorted = sorted(pareto, key=lambda x: x.rms_mm)
        xs = [p.rms_mm for p in pareto_sorted]
        ys = [p.dE_image for p in pareto_sorted]
        ax.step(xs, ys, where="post", color="steelblue",
                linewidth=1.2, linestyle="--", alpha=0.6)

    for p in points:
        ax.annotate(p.method, (p.rms_mm, p.dE_image),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

    ax.set_xlabel("RMS Shift (mm)  ↓ better")
    ax.set_ylabel("ΔE_mis (image)  ↓ better")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        # Also save SVG
        svg_path = path.with_suffix(".svg")
        fig.savefig(str(svg_path), bbox_inches="tight")
        plt.close(fig)
    return fig
