"""Aggregation statistics: mean, 95% CI, per-channel/per-method tables.

Supports 3-sheets × 3-scans protocol from the benchmark specification.
"""

from __future__ import annotations
import dataclasses
import csv
import math
from pathlib import Path
from typing import Optional
import numpy as np

# t-distribution critical values (two-tailed 95 %) for small n
# index = degrees of freedom (1..30), then use normal approx
_T_CRIT = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042,
}

def _t_crit(df: int, alpha: float = 0.05) -> float:
    """Two-tailed t critical value for given df."""
    if df <= 0:
        return float("nan")
    if df in _T_CRIT:
        return _T_CRIT[df]
    # Linear interpolation or normal approx for large df
    if df > 30:
        return 1.96
    keys = sorted(_T_CRIT.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i+1]:
            lo, hi = keys[i], keys[i+1]
            frac = (df - lo) / (hi - lo)
            return _T_CRIT[lo] + frac * (_T_CRIT[hi] - _T_CRIT[lo])
    return 1.96


@dataclasses.dataclass
class AggregatedResult:
    channel: str
    method: str
    mean_dx_px: float
    mean_dy_px: float
    mean_rms_px: float
    ci_rms_px: float
    mean_rms_mm: float
    ci_rms_mm: float
    mean_dE: float
    ci_dE: float
    n_obs: int


def compute_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return (mean, half-width CI) for a list of scalar observations."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if n == 1:
        return mean, float("nan")
    std = float(np.std(values, ddof=1))
    df = n - 1
    tc = _t_crit(df)
    ci = tc * std / math.sqrt(n)
    return mean, ci


def aggregate_replicates(
    rms_px_list: list[float],
    rms_mm_list: list[float],
    dx_list: list[float],
    dy_list: list[float],
    dE_list: list[float],
    channel: str = "?",
    method: str = "?",
) -> AggregatedResult:
    """Aggregate over n_sheets × n_scans replicates."""
    mean_dx, _ = compute_ci(dx_list)
    mean_dy, _ = compute_ci(dy_list)
    mean_rms_px, ci_rms_px = compute_ci(rms_px_list)
    mean_rms_mm, ci_rms_mm = compute_ci(rms_mm_list)
    mean_dE, ci_dE = compute_ci(dE_list)
    return AggregatedResult(
        channel=channel, method=method,
        mean_dx_px=mean_dx, mean_dy_px=mean_dy,
        mean_rms_px=mean_rms_px, ci_rms_px=ci_rms_px,
        mean_rms_mm=mean_rms_mm, ci_rms_mm=ci_rms_mm,
        mean_dE=mean_dE, ci_dE=ci_dE,
        n_obs=len(rms_px_list),
    )


def build_per_channel_table(results: list[AggregatedResult]) -> list[dict]:
    """Build a list-of-dicts table suitable for CSV export (per channel)."""
    rows = []
    for r in results:
        rows.append({
            "method": r.method,
            "channel": r.channel,
            "mean_dx_px": f"{r.mean_dx_px:.4f}",
            "mean_dy_px": f"{r.mean_dy_px:.4f}",
            "mean_rms_px": f"{r.mean_rms_px:.4f}",
            "ci_rms_px":   f"{r.ci_rms_px:.4f}",
            "mean_rms_mm": f"{r.mean_rms_mm:.4f}",
            "ci_rms_mm":   f"{r.ci_rms_mm:.4f}",
            "mean_dE":     f"{r.mean_dE:.4f}",
            "ci_dE":       f"{r.ci_dE:.4f}",
            "n_obs":        r.n_obs,
        })
    return rows


def build_per_method_table(results: list[AggregatedResult]) -> list[dict]:
    """Collapse channels, report worst-case RMS per method (benchmark Table 3)."""
    from collections import defaultdict
    by_method: dict[str, list] = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)
    rows = []
    for method, items in by_method.items():
        max_rms_mm = max(i.mean_rms_mm for i in items)
        max_dE = max(i.mean_dE for i in items)
        rows.append({
            "method": method,
            "max_rms_mm": f"{max_rms_mm:.4f}",
            "max_dE": f"{max_dE:.4f}",
            "channels": ",".join(i.channel for i in items),
            "n_obs": items[0].n_obs,
        })
    return rows


def export_csv(rows: list[dict], path: str | Path) -> None:
    """Write a list-of-dicts to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Per-patch overprint dE table (spec requirement)
# ---------------------------------------------------------------------------

def build_overprint_patch_table(
    patch_results: list[dict],
) -> list[dict]:
    """Build per-patch ΔE_mis table for benchmark reporting.

    Parameters
    ----------
    patch_results : list of dicts with keys:
        method, combo (e.g. "CM"), dE_patch, image_name (optional)

    Returns list of row dicts.
    """
    rows = []
    for p in patch_results:
        rows.append({
            "method":     p.get("method", "?"),
            "patch_combo": p.get("combo", "?"),
            "image":      p.get("image_name", ""),
            "dE_patch":   f"{p.get('dE_patch', float('nan')):.4f}",
        })
    return rows


def build_benchmark_table3(
    per_method_rows: list[dict],
) -> list[dict]:
    """Format Table 3 from the benchmark spec.

    Columns: Method | Misreg.(mm) ↓ | ΔE_mis ↓ | n_obs
    """
    rows = []
    for r in per_method_rows:
        rows.append({
            "Method":       r["method"],
            "Misreg_mm":    r["max_rms_mm"],
            "DeltaE_mis":   r["max_dE"],
            "n_obs":        r["n_obs"],
        })
    return rows
