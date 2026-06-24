#!/usr/bin/env python3
"""Smoke test: validates entire pipeline using ordinary PNG/JPG images.

Usage:
    python scripts/smoke_test.py --dataset dataset/
    python scripts/smoke_test.py --demo   # uses synthetic demo images

Pipeline per image:
  1. load image
  2. generate synthetic CMYK separations
  3. apply known channel shifts
  4. reconstruct image
  5. estimate dx, dy, RMS shift
  6. estimate ΔE_mis
  7. generate plots
  8. export CSV
"""
import argparse, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

KNOWN_SHIFTS = {"C": (1.5, -0.8), "M": (-0.5, 1.2), "Y": (2.0, 0.3), "K": (0.0, 0.0)}
TOLERANCE_PX = 1.5   # acceptable estimation error in pixels

def _make_demo_images(folder: Path, n: int = 3) -> None:
    """Create simple synthetic RGB images for demo mode."""
    from PIL import Image, ImageDraw, ImageFont
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (256, 256), color=(200, 180, 160))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 100, 100], fill=(255, 50, 50))
        draw.rectangle([130, 20, 230, 100], fill=(50, 200, 50))
        draw.ellipse([80, 130, 200, 220], fill=(50, 50, 220))
        draw.text((10, 230), f"demo_{i}", fill=(0, 0, 0))
        img.save(str(folder / f"demo_{i:03d}.png"))


def run_smoke_test(dataset_folder: Path, output_dir: Path, demo: bool) -> bool:
    from misregistration.simulation import simulate_from_folder, simulate_from_png
    from misregistration.visualization import save_summary_figure
    from misregistration.aggregation import (
        aggregate_replicates, build_per_channel_table, export_csv
    )

    if demo:
        demo_dir = output_dir / "demo_images"
        _make_demo_images(demo_dir)
        dataset_folder = demo_dir

    images = [p for p in sorted(dataset_folder.iterdir())
              if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]
    if not images:
        print(f"[ERROR] No images found in {dataset_folder}")
        return False

    print(f"Found {len(images)} image(s) in {dataset_folder}")
    output_dir.mkdir(parents=True, exist_ok=True)

    passed = 0
    failed = 0
    all_agg = []

    for img_path in images:
        print(f"\n--- {img_path.name} ---")
        try:
            # Step 1-4: simulate
            result = simulate_from_png(img_path, shifts_px=KNOWN_SHIFTS)

            # Step 5: check estimation accuracy
            for ch in ("C", "M", "Y"):
                est_rms = result.rms_px[ch]
                true_dx, true_dy = KNOWN_SHIFTS[ch]
                true_rms = float(np.sqrt(true_dx**2 + true_dy**2))
                err = abs(est_rms - true_rms)
                status = "OK" if err < TOLERANCE_PX else "WARN"
                print(f"  {ch}: true_rms={true_rms:.3f}px  est_rms={est_rms:.3f}px  err={err:.3f}px [{status}]")

            # Step 6: ΔE
            print(f"  ΔE_mis(image) = {result.dE_image:.4f}")

            # Step 7: plots
            ch_rms = result.rms_px
            shifts_plot = {ch: result.shifts_px[ch] for ch in ("C", "M", "Y")}
            save_summary_figure(ch_rms, shifts_plot, result.dE_map,
                                output_dir / img_path.stem)

            # Step 8: CSV
            for ch in ("C", "M", "Y"):
                agg = aggregate_replicates(
                    rms_px_list=[result.rms_px[ch]],
                    rms_mm_list=[result.rms_mm[ch]],
                    dx_list=[result.shifts_px[ch][0]],
                    dy_list=[result.shifts_px[ch][1]],
                    dE_list=[result.dE_image],
                    channel=ch, method="smoke_test",
                )
                all_agg.append(agg)

            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    rows = build_per_channel_table(all_agg)
    csv_path = output_dir / "smoke_test_results.csv"
    export_csv(rows, csv_path)
    print(f"\n{'='*50}")
    print(f"Smoke test: {passed} passed, {failed} failed")
    print(f"CSV: {csv_path}")
    print(f"Plots: {output_dir}/")
    return failed == 0


def main():
    p = argparse.ArgumentParser(description="Smoke test for misregistration pipeline")
    p.add_argument("--dataset", default="dataset", help="Folder containing images")
    p.add_argument("--output-dir", default="results/smoke_test")
    p.add_argument("--demo", action="store_true", help="Use synthetic demo images")
    args = p.parse_args()

    ok = run_smoke_test(Path(args.dataset), Path(args.output_dir), demo=args.demo)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
