#!/usr/bin/env python3
"""Batch measure misregistration over a folder of images."""
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from misregistration.simulation import simulate_from_folder
from misregistration.aggregation import (
    aggregate_replicates, build_per_channel_table, export_csv
)
from misregistration.visualization import save_summary_figure

def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Folder of images")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--dpi", type=float, default=None)
    p.add_argument("--shifts", default=None)
    args = p.parse_args()

    shifts = json.loads(args.shifts) if args.shifts else None
    if shifts:
        shifts = {k: tuple(v) for k, v in shifts.items()}
    px_per_mm = (args.dpi / 25.4) if args.dpi else None

    results = simulate_from_folder(args.folder, shifts_px=shifts,
                                   px_per_mm=px_per_mm,
                                   max_images=args.max_images)
    print(f"Processed {len(results)} images")

    all_rows = []
    for r in results:
        for ch in ("C", "M", "Y"):
            agg = aggregate_replicates(
                rms_px_list=[r.rms_px.get(ch, 0)],
                rms_mm_list=[r.rms_mm.get(ch, float("nan"))],
                dx_list=[r.shifts_px.get(ch, (0,0))[0]],
                dy_list=[r.shifts_px.get(ch, (0,0))[1]],
                dE_list=[r.dE_image],
                channel=ch, method=Path(r.image_path).stem,
            )
            all_rows.append(agg)

    out = Path(args.output_dir)
    rows = build_per_channel_table(all_rows)
    export_csv(rows, out / "batch_misregistration.csv")
    print(f"CSV saved to {out / 'batch_misregistration.csv'}")

if __name__ == "__main__":
    main()
