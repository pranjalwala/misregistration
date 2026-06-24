#!/usr/bin/env python3
"""Full dataset benchmark: process up to 450 images and produce all outputs.

Usage
-----
# Flat folder of images (single method called "dataset"):
    python scripts/run_dataset_benchmark.py dataset/

# One sub-folder per method:
    python scripts/run_dataset_benchmark.py dataset/ --by-method

# Limit images per method (useful for quick tests):
    python scripts/run_dataset_benchmark.py dataset/ --max-images 10

Outputs
-------
results/
  per_image.csv          one row per (image, channel)
  per_method.csv         one row per (method, channel) with CI
  summary.csv            worst-case per method
  benchmark_table.csv    Table 3 from spec
  pareto.csv             Pareto-optimal methods
  pareto_plot.png/.svg
  plots/                 per-image summary figures (optional, --plots)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from misregistration.simulation import simulate_from_folder, simulate_from_png
from misregistration.aggregation import (
    aggregate_replicates,
    build_per_channel_table,
    build_per_method_table,
    build_benchmark_table3,
    export_csv,
    compute_pareto_frontier,
    export_pareto_csv,
    plot_pareto_frontier,
    AggregatedResult,
)
from misregistration.visualization import save_summary_figure

CHANNELS = ("C", "M", "Y")


def _process_folder(
    folder: Path,
    method_name: str,
    shifts_px: dict | None,
    px_per_mm: float | None,
    max_images: int | None,
    make_plots: bool,
    plots_dir: Path,
) -> tuple[list[dict], list[AggregatedResult]]:
    """Process one method folder. Returns (per_image_rows, aggregated_list)."""
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    image_paths = [p for p in sorted(folder.iterdir())
                   if p.is_file() and p.suffix.lower() in exts]
    if max_images:
        image_paths = image_paths[:max_images]

    if not image_paths:
        print(f"  [warn] no images found in {folder}")
        return [], []

    print(f"  Processing {len(image_paths)} image(s) for method '{method_name}'")

    per_image_rows = []
    # Accumulate per-channel lists for CI computation
    ch_data: dict[str, dict] = {ch: {"rms_px": [], "rms_mm": [],
                                      "dx": [], "dy": [], "dE": []}
                                 for ch in CHANNELS}

    for i, img_path in enumerate(image_paths):
        try:
            result = simulate_from_png(img_path, shifts_px=shifts_px,
                                       px_per_mm=px_per_mm)
        except Exception as e:
            print(f"    [skip] {img_path.name}: {e}")
            continue

        # Per-image rows
        for ch in CHANNELS:
            per_image_rows.append({
                "method":     method_name,
                "image":      img_path.name,
                "channel":    ch,
                "dx_px":      f"{result.shifts_px.get(ch, (0,0))[0]:.4f}",
                "dy_px":      f"{result.shifts_px.get(ch, (0,0))[1]:.4f}",
                "rms_px":     f"{result.rms_px.get(ch, 0):.4f}",
                "rms_mm":     f"{result.rms_mm.get(ch, float('nan')):.4f}",
                "dE_image":   f"{result.dE_image:.4f}",
            })
            ch_data[ch]["rms_px"].append(result.rms_px.get(ch, 0))
            ch_data[ch]["rms_mm"].append(result.rms_mm.get(ch, float("nan")))
            ch_data[ch]["dx"].append(result.shifts_px.get(ch, (0, 0))[0])
            ch_data[ch]["dy"].append(result.shifts_px.get(ch, (0, 0))[1])
            ch_data[ch]["dE"].append(result.dE_image)

        # Optional plots (every 10th image to keep it fast)
        if make_plots and i % max(1, len(image_paths) // 10) == 0:
            try:
                save_summary_figure(
                    result.rms_px,
                    {ch: result.shifts_px[ch] for ch in CHANNELS},
                    result.dE_map,
                    plots_dir / method_name / img_path.stem,
                )
            except Exception:
                pass

    # Aggregate per channel
    aggregated = []
    for ch in CHANNELS:
        d = ch_data[ch]
        if not d["rms_px"]:
            continue
        agg = aggregate_replicates(
            rms_px_list=d["rms_px"],
            rms_mm_list=d["rms_mm"],
            dx_list=d["dx"],
            dy_list=d["dy"],
            dE_list=d["dE"],
            channel=ch,
            method=method_name,
        )
        aggregated.append(agg)

    return per_image_rows, aggregated


def main():
    p = argparse.ArgumentParser(
        description="Full misregistration dataset benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("dataset", help="Dataset folder (flat or sub-folders per method)")
    p.add_argument("--output-dir", default="results", help="Output directory")
    p.add_argument("--by-method", action="store_true",
                   help="Each sub-folder is a separate halftoning method")
    p.add_argument("--max-images", type=int, default=None,
                   help="Max images per method (useful for quick tests)")
    p.add_argument("--dpi", type=float, default=None,
                   help="Scanner DPI for mm conversion (read from TIFF metadata if absent)")
    p.add_argument("--shifts", default=None,
                   help='JSON: e.g. \'{"C":[1.5,-0.8],"M":[-0.5,1.2],"Y":[2,0.3],"K":[0,0]}\'')
    p.add_argument("--plots", action="store_true",
                   help="Generate per-image summary plots (slower)")
    args = p.parse_args()

    import json
    shifts_px = None
    if args.shifts:
        raw = json.loads(args.shifts)
        shifts_px = {k: tuple(v) for k, v in raw.items()}

    px_per_mm = (args.dpi / 25.4) if args.dpi else None
    dataset = Path(args.dataset)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"

    # Discover method folders or use flat dataset
    if args.by_method:
        method_dirs = [(d.name, d) for d in sorted(dataset.iterdir()) if d.is_dir()]
        if not method_dirs:
            print("[warn] --by-method set but no sub-folders found; treating as flat")
            method_dirs = [("dataset", dataset)]
    else:
        method_dirs = [(dataset.name, dataset)]

    all_per_image: list[dict] = []
    all_aggregated: list[AggregatedResult] = []

    for method_name, folder in method_dirs:
        print(f"\n[{method_name}]")
        rows, aggs = _process_folder(
            folder, method_name, shifts_px, px_per_mm,
            args.max_images, args.plots, plots_dir,
        )
        all_per_image.extend(rows)
        all_aggregated.extend(aggs)

    if not all_per_image:
        print("\n[ERROR] No images were processed. Check dataset path.")
        sys.exit(1)

    # --- Export outputs ---
    export_csv(all_per_image,                          out / "per_image.csv")
    export_csv(build_per_channel_table(all_aggregated), out / "per_method.csv")

    per_method_summary = build_per_method_table(all_aggregated)
    export_csv(per_method_summary,                     out / "summary.csv")

    table3 = [{"Method": r["method"], "Misreg_mm": r["max_rms_mm"],
                "DeltaE_mis": r["max_dE"], "n_obs": r["n_obs"]}
              for r in per_method_summary]
    export_csv(table3,                                  out / "benchmark_table.csv")

    # --- Pareto analysis ---
    methods   = [r["method"]    for r in per_method_summary]
    rms_vals  = [float(r["max_rms_mm"]) for r in per_method_summary]
    dE_vals   = [float(r["max_dE"])     for r in per_method_summary]

    if len(methods) >= 2:
        pareto_pts = compute_pareto_frontier(methods, rms_vals, dE_vals)
        export_pareto_csv(pareto_pts, out / "pareto.csv")
        try:
            plot_pareto_frontier(pareto_pts, output_path=out / "pareto_plot.png")
        except Exception as e:
            print(f"  [warn] Pareto plot failed: {e}")
    else:
        # Single method — write trivial pareto file
        from misregistration.aggregation import ParetoPoint
        pts = [ParetoPoint(methods[0], rms_vals[0], dE_vals[0], True)]
        export_pareto_csv(pts, out / "pareto.csv")

    # --- Summary printout ---
    print(f"\n{'='*56}")
    print(f"{'Method':<22} {'Misreg(mm)':>12} {'ΔE_mis':>10} {'n':>5}")
    print(f"{'-'*56}")
    for r in per_method_summary:
        print(f"  {r['method']:<20} {r['max_rms_mm']:>12} {r['max_dE']:>10} {r['n_obs']:>5}")
    print(f"{'='*56}")
    print(f"\nOutputs saved to {out}/")
    print(f"  per_image.csv       ({len(all_per_image)} rows)")
    print(f"  per_method.csv")
    print(f"  summary.csv")
    print(f"  benchmark_table.csv")
    print(f"  pareto.csv")


if __name__ == "__main__":
    main()
