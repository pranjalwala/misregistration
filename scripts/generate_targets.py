#!/usr/bin/env python3
"""Generate calibration targets and save to generated_targets/."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from misregistration.calibration_targets import generate_all_targets

def main():
    p = argparse.ArgumentParser(description="Generate calibration targets")
    p.add_argument("--output-dir", default="generated_targets", help="Output directory")
    p.add_argument("--dpi", type=float, default=300.0, help="Target DPI (runtime, not hardcoded)")
    p.add_argument("--no-tiff", action="store_true")
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args()
    manifest = generate_all_targets(args.output_dir, dpi=args.dpi,
                                    save_tiff=not args.no_tiff,
                                    save_png=not args.no_png)
    print(f"Generated {len(manifest['targets'])} targets in '{args.output_dir}'")

if __name__ == "__main__":
    main()
