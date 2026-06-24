"""Aggregation: statistics over replicate sheets/scans, CI, CSV export, Pareto."""
from .statistics import (
    aggregate_replicates,
    compute_ci,
    build_per_channel_table,
    build_per_method_table,
    export_csv,
    AggregatedResult,
    build_overprint_patch_table,
    build_benchmark_table3,
)
from .pareto import (
    ParetoPoint,
    compute_pareto_frontier,
    export_pareto_csv,
    plot_pareto_frontier,
)

__all__ = [
    "aggregate_replicates", "compute_ci",
    "build_per_channel_table", "build_per_method_table",
    "export_csv", "AggregatedResult",
    "build_overprint_patch_table", "build_benchmark_table3",
    "ParetoPoint", "compute_pareto_frontier",
    "export_pareto_csv", "plot_pareto_frontier",
]
