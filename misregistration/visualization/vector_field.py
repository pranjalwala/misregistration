"""Vector field visualizations for channel misregistration.

Generates:
  • registration vector maps (quiver plots)
  • channel displacement overlays
  • spatial heatmaps of shift magnitude
  • multi-channel comparison grids

All exports: PNG + SVG.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


CHANNEL_COLORS = {"C": "cyan", "M": "magenta", "Y": "gold", "K": "dimgray"}


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    svg = path.with_suffix(".svg")
    fig.savefig(str(svg), bbox_inches="tight")
    plt.close(fig)


def plot_registration_vector_map(
    shift_grid: dict[str, np.ndarray],
    title: str = "Registration Vector Map",
    output_path: Optional[str | Path] = None,
    scale: float = 1.0,
) -> plt.Figure:
    """Quiver plot of shift vectors at each grid point.

    Parameters
    ----------
    shift_grid : dict channel → (H, W, 2) array of (dx, dy) per grid cell
    scale      : arrow scale factor
    """
    channels = list(shift_grid.keys())
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        grid = shift_grid[ch]  # (H, W, 2)
        H, W = grid.shape[:2]
        ys, xs = np.mgrid[0:H, 0:W]
        dx = grid[..., 0]
        dy = grid[..., 1]
        color = CHANNEL_COLORS.get(ch, "steelblue")
        q = ax.quiver(xs, ys, dx, dy, scale=scale,
                      color=color, alpha=0.8, width=0.005)
        ax.quiverkey(q, 0.9, 1.02, 1.0, "1 px", labelpos="E", fontproperties={"size": 7})
        ax.set_title(f"Ch {ch}")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("x (grid cells)")
        ax.set_ylabel("y (grid cells)")

    fig.suptitle(title)
    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_displacement_overlay(
    reference: np.ndarray,
    shifts: dict[str, tuple[float, float]],
    title: str = "Channel Displacement Overlay",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Show reference image with arrows indicating each channel's shift."""
    fig, ax = plt.subplots(figsize=(6, 6))
    if reference.ndim == 2:
        ax.imshow(reference, cmap="gray", alpha=0.6)
    else:
        ax.imshow(reference, alpha=0.6)

    h, w = reference.shape[:2]
    cx, cy = w / 2, h / 2
    scale = max(w, h) / 10.0

    for ch, (dx, dy) in shifts.items():
        color = CHANNEL_COLORS.get(ch, "steelblue")
        ax.annotate(
            "", xy=(cx + dx * scale, cy + dy * scale), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
        )
        ax.text(cx + dx * scale + 3, cy + dy * scale + 3,
                ch, color=color, fontsize=12, fontweight="bold")

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_shift_heatmap(
    rms_map: np.ndarray,
    channel: str = "?",
    title: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """Spatial heatmap of local RMS shift magnitude."""
    title = title or f"RMS Shift Heatmap — Channel {channel}"
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rms_map, cmap="plasma",
                   vmin=0, vmax=vmax or float(np.percentile(rms_map, 99)))
    plt.colorbar(im, ax=ax, label="RMS shift (px)")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_multichannel_shifts(
    shifts_per_image: list[dict[str, tuple[float, float]]],
    image_names: Optional[list[str]] = None,
    output_path: Optional[str | Path] = None,
    title: str = "Per-Image Channel Shifts",
) -> plt.Figure:
    """Line plot of dx, dy per channel across multiple images."""
    channels = ["C", "M", "Y"]
    n = len(shifts_per_image)
    image_names = image_names or [str(i) for i in range(n)]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, n * 0.4 + 2), 6), sharex=True)
    x = np.arange(n)

    for ch in channels:
        color = CHANNEL_COLORS.get(ch, "steelblue")
        dx_vals = [s.get(ch, (0, 0))[0] for s in shifts_per_image]
        dy_vals = [s.get(ch, (0, 0))[1] for s in shifts_per_image]
        axes[0].plot(x, dx_vals, label=f"{ch} dx", color=color, linewidth=1.2)
        axes[1].plot(x, dy_vals, label=f"{ch} dy", color=color,
                     linewidth=1.2, linestyle="--")

    axes[0].set_ylabel("dx (px)")
    axes[0].legend(fontsize=8)
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_ylabel("dy (px)")
    axes[1].legend(fontsize=8)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_xticks(x[::max(1, n//20)])
    axes[1].set_xticklabels(image_names[::max(1, n//20)],
                             rotation=45, ha="right", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def build_local_shift_grid(
    reference: np.ndarray,
    shifted: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 4,
    max_shift: int = 20,
) -> np.ndarray:
    """Compute local NCC shift at each grid cell.

    Returns (grid_rows, grid_cols, 2) array of (dx, dy).
    """
    from misregistration.measurement.ncc import compute_ncc, find_peak_subpixel

    h, w = reference.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    grid = np.zeros((grid_rows, grid_cols, 2), dtype=np.float32)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            ref_cell = reference[y0:y1, x0:x1]
            sh_cell = shifted[y0:y1, x0:x1]
            if ref_cell.size < 16:
                continue
            ncc = compute_ncc(ref_cell.astype(np.float32),
                              sh_cell.astype(np.float32),
                              max_shift=max_shift)
            dx, dy, _ = find_peak_subpixel(ncc)
            grid[r, c] = [dx, dy]

    return grid
