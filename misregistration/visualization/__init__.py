"""Visualization: plots, heatmaps, vector fields, ΔE maps."""
from .misregistration_plots import (
    plot_rms_bar,
    plot_shift_vectors,
    plot_dE_map,
    plot_ci_bars,
    plot_channel_overlay,
    plot_ncc_surface,
    plot_histogram,
    save_summary_figure,
)
from .vector_field import (
    plot_registration_vector_map,
    plot_displacement_overlay,
    plot_shift_heatmap,
    plot_multichannel_shifts,
    build_local_shift_grid,
)

__all__ = [
    "plot_rms_bar", "plot_shift_vectors", "plot_dE_map",
    "plot_ci_bars", "plot_channel_overlay", "plot_ncc_surface",
    "plot_histogram", "save_summary_figure",
    "plot_registration_vector_map", "plot_displacement_overlay",
    "plot_shift_heatmap", "plot_multichannel_shifts",
    "build_local_shift_grid",
]
