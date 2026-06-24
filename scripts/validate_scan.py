#!/usr/bin/env python3
"""Validate a scanned TIFF: check metadata, resolution, channel count."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from misregistration.io import load_image

def main():
    p = argparse.ArgumentParser(description="Validate a scan file")
    p.add_argument("scan", help="Path to scanned TIFF/PNG")
    p.add_argument("--fallback-dpi", type=float, default=300.0)
    args = p.parse_args()
    img, meta = load_image(args.scan, fallback_dpi=args.fallback_dpi)
    print(f"File      : {meta.source_path}")
    print(f"Size      : {meta.width} x {meta.height} px")
    print(f"Channels  : {meta.n_channels}")
    print(f"Bit depth : {meta.bit_depth}")
    print(f"DPI       : {meta.dpi_x:.1f} x {meta.dpi_y:.1f}")
    print(f"px/mm     : {meta.px_per_mm:.4f}")
    if meta.px_per_mm < 5:
        print("[WARN] Very low resolution – measurements may be inaccurate")
    print("PASS")

if __name__ == "__main__":
    main()
