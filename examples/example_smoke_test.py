#!/usr/bin/env python3
"""Example: run smoke test on demo images (no printer required)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from misregistration.simulation import simulate_from_png
from misregistration.visualization import save_summary_figure
from misregistration.aggregation import aggregate_replicates, build_per_channel_table, export_csv

DEMO_DIR = Path(__file__).parent.parent / "demo_data"
OUT_DIR  = Path(__file__).parent.parent / "results" / "example"

SHIFTS = {"C": (1.5, -0.8), "M": (-0.5, 1.2), "Y": (2.0, 0.3), "K": (0.0, 0.0)}

def main():
    images = sorted(DEMO_DIR.glob("*.png"))
    if not images:
        print("No demo images found. Run: python scripts/smoke_test.py --demo")
        return

    all_agg = []
    for img_path in images:
        print(f"Processing {img_path.name}...")
        result = simulate_from_png(img_path, shifts_px=SHIFTS)
        print(f"  ΔE_mis = {result.dE_image:.3f}")
        for ch in ("C","M","Y"):
            agg = aggregate_replicates(
                [result.rms_px[ch]], [result.rms_mm[ch]],
                [SHIFTS[ch][0]], [SHIFTS[ch][1]],
                [result.dE_image], ch, "example"
            )
            all_agg.append(agg)
        save_summary_figure(
            {ch: result.rms_px[ch] for ch in ("C","M","Y")},
            {ch: SHIFTS[ch] for ch in ("C","M","Y")},
            result.dE_map,
            OUT_DIR / img_path.stem,
        )

    rows = build_per_channel_table(all_agg)
    export_csv(rows, OUT_DIR / "example_results.csv")
    print(f"\nDone. Results in {OUT_DIR}/")

if __name__ == "__main__":
    main()
