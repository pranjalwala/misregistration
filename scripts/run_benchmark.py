#!/usr/bin/env python3
"""Run the full misregistration benchmark over a dataset folder.

Produces per-channel tables, per-method summary, and benchmark Table 3 CSV.
"""
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from misregistration.simulation import simulate_from_folder
from misregistration.aggregation import (
    aggregate_replicates, build_per_channel_table,
    build_per_method_table, export_csv
)
from misregistration.visualization import save_summary_figure

METHODS = ["DBS", "ErrorDiffusion", "OrderedDithering", "DeepLearning"]

def main():
    p = argparse.ArgumentParser(description="Full benchmark runner")
    p.add_argument("dataset", help="Dataset root folder (sub-folders = methods)")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--dpi", type=float, default=300.0)
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.output_dir)
    px_per_mm = args.dpi / 25.4

    all_agg = []
    # If sub-folders exist treat them as methods; otherwise run flat
    subdirs = [d for d in dataset.iterdir() if d.is_dir()]
    if not subdirs:
        subdirs = [dataset]

    for subdir in subdirs:
        method = subdir.name
        results = simulate_from_folder(subdir, px_per_mm=px_per_mm,
                                       max_images=args.max_images)
        for ch in ("C", "M", "Y"):
            rms_px_list = [r.rms_px.get(ch, 0) for r in results]
            rms_mm_list = [r.rms_mm.get(ch, float("nan")) for r in results]
            dx_list = [r.shifts_px.get(ch, (0,0))[0] for r in results]
            dy_list = [r.shifts_px.get(ch, (0,0))[1] for r in results]
            dE_list  = [r.dE_image for r in results]
            if not rms_px_list:
                continue
            agg = aggregate_replicates(rms_px_list, rms_mm_list, dx_list, dy_list,
                                       dE_list, channel=ch, method=method)
            all_agg.append(agg)

    per_ch_rows = build_per_channel_table(all_agg)
    per_method_rows = build_per_method_table(all_agg)
    export_csv(per_ch_rows,     out / "table_per_channel.csv")
    export_csv(per_method_rows, out / "table3_per_method.csv")
    print(f"Benchmark complete. Results in {out}/")
    print("Per-method summary:")
    for row in per_method_rows:
        print(f"  {row['method']:20s}  misreg={row['max_rms_mm']}mm  dE={row['max_dE']}")

if __name__ == "__main__":
    main()
