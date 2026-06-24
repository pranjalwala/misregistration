#!/usr/bin/env python3
"""Measure channel misregistration from a scanned image (or synthetic demo)."""
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from misregistration.simulation import simulate_from_png
from misregistration.visualization import save_summary_figure
from misregistration.aggregation import export_csv, build_per_channel_table, aggregate_replicates

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Input image (RGB PNG/JPG/TIFF)")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--shifts", default=None,
                   help='JSON shift dict e.g. \'{"C":[1.5,-0.8],"M":[-0.5,1.2],"Y":[2.0,0.3],"K":[0,0]}\'')
    p.add_argument("--dpi", type=float, default=None)
    args = p.parse_args()

    shifts = json.loads(args.shifts) if args.shifts else None
    if shifts:
        shifts = {k: tuple(v) for k, v in shifts.items()}

    px_per_mm = (args.dpi / 25.4) if args.dpi else None
    result = simulate_from_png(args.image, shifts_px=shifts, px_per_mm=px_per_mm)

    channel_rms = {ch: result.rms_px[ch] for ch in result.rms_px}
    shifts_out  = {ch: result.shifts_px[ch] for ch in result.shifts_px
                   if ch in ("C", "M", "Y", "K")}

    out = Path(args.output_dir)
    save_summary_figure(channel_rms, shifts_out, result.dE_map, out)

    agg = aggregate_replicates(
        rms_px_list=[result.rms_px.get("C", 0)],
        rms_mm_list=[result.rms_mm.get("C", float("nan"))],
        dx_list=[result.shifts_px.get("C", (0,0))[0]],
        dy_list=[result.shifts_px.get("C", (0,0))[1]],
        dE_list=[result.dE_image],
        channel="C", method="simulation",
    )
    rows = build_per_channel_table([agg])
    export_csv(rows, out / "misregistration.csv")
    print(f"dE_image = {result.dE_image:.4f}")
    print(f"Results saved to {out}/")

if __name__ == "__main__":
    main()
