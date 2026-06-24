"""Visualization utilities for channel misregistration results."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np


# Always use non-interactive backend so it works headlessly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


CHANNEL_COLORS = {"C": "cyan", "M": "magenta", "Y": "gold", "K": "black"}


def _savefig(fig, path: str | Path, fmt: str = "png") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rms_bar(
    channel_rms: dict[str, float],
    title: str = "RMS Channel Shift",
    ylabel: str = "RMS shift (mm)",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart of per-channel RMS shift."""
    fig, ax = plt.subplots(figsize=(6, 4))
    channels = list(channel_rms.keys())
    values = [channel_rms[c] for c in channels]
    colors = [CHANNEL_COLORS.get(c, "steelblue") for c in channels]
    bars = ax.bar(channels, values, color=colors, edgecolor="gray", linewidth=0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Channel")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    if output_path:
        _savefig(fig, output_path)
    return fig


def plot_shift_vectors(
    shifts: dict[str, tuple[float, float]],
    title: str = "Channel Shift Vectors",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Vector field showing per-channel (dx, dy) displacement."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    for ch, (dx, dy) in shifts.items():
        color = CHANNEL_COLORS.get(ch, "steelblue")
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.text(dx, dy, f"  {ch}", color=color, fontsize=11, fontweight="bold")
    all_vals = [v for dxy in shifts.values() for v in dxy] + [0]
    lim = max(abs(v) for v in all_vals) * 1.4 + 0.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("dx (px)")
    ax.set_ylabel("dy (px)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    if output_path:
        _savefig(fig, output_path)
    return fig


def plot_dE_map(
    dE_map: np.ndarray,
    title: str = "ΔE Misregistration Map",
    output_path: Optional[str | Path] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """Heatmap of per-pixel CIEDE2000 misregistration error."""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(dE_map, cmap="hot", vmin=0, vmax=vmax or np.percentile(dE_map, 99))
    plt.colorbar(im, ax=ax, label="ΔE₀₀")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if output_path:
        _savefig(fig, output_path)
    return fig


def plot_ci_bars(
    results,
    metric: str = "mean_rms_mm",
    ci_field: str = "ci_rms_mm",
    title: str = "RMS Shift ± 95% CI",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart with 95% confidence interval error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, values, cis = [], [], []
    for r in results:
        labels.append(f"{r.method}\n{r.channel}")
        values.append(getattr(r, metric, 0.0))
        cis.append(getattr(r, ci_field, 0.0))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=[ci if not (ci != ci) else 0 for ci in cis],
           capsize=4, color="steelblue", edgecolor="gray", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(metric)
    fig.tight_layout()
    if output_path:
        _savefig(fig, output_path)
    return fig


def plot_channel_overlay(
    separations: dict[str, np.ndarray],
    title: str = "Channel Overlay",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Side-by-side display of CMYK channel separations."""
    channels = list(separations.keys())
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        img = separations[ch]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(ch)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if output_path:
        _savefig(fig, output_path)
    return fig


def plot_ncc_surface(
    ncc_map: np.ndarray,
    peak: tuple[float, float] = (0.0, 0.0),
    title: str = "NCC Surface",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """3D or 2D plot of NCC correlation surface."""
    fig, ax = plt.subplots(figsize=(6, 5))
    max_shift = ncc_map.shape[0] // 2
    extent = [-max_shift, max_shift, -max_shift, max_shift]
    im = ax.imshow(ncc_map, cmap="viridis", extent=extent, origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax, label="NCC")
    ax.scatter([peak[0]], [peak[1]], color="red", s=50, zorder=5, label="peak")
    ax.legend(fontsize=8)
    ax.set_xlabel("dx (px)")
    ax.set_ylabel("dy (px)")
    ax.set_title(title)
    fig.tight_layout()
    if output_path:
        _savefig(fig, output_path)
    return fig


def plot_histogram(
    values: list[float],
    label: str = "RMS shift (mm)",
    title: str = "Distribution",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Histogram of measurement distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=max(5, len(values)//3 + 1), color="steelblue",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    if output_path:
        _savefig(fig, output_path)
    return fig


def save_summary_figure(
    channel_rms: dict[str, float],
    shifts: dict[str, tuple[float, float]],
    dE_map: Optional[np.ndarray],
    output_dir: str | Path,
    prefix: str = "summary",
) -> list[Path]:
    """Generate and save the standard set of summary plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    p = out / f"{prefix}_rms_bar.png"
    plot_rms_bar(channel_rms, output_path=p)
    saved.append(p)

    p = out / f"{prefix}_shift_vectors.png"
    plot_shift_vectors(shifts, output_path=p)
    saved.append(p)

    if dE_map is not None:
        p = out / f"{prefix}_dE_map.png"
        plot_dE_map(dE_map, output_path=p)
        saved.append(p)

    return saved
