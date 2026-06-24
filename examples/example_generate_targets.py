#!/usr/bin/env python3
"""Example: generate calibration targets at 300 DPI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from misregistration.calibration_targets import generate_all_targets

manifest = generate_all_targets("generated_targets", dpi=300.0, save_png=True, save_tiff=False)
print(f"Generated {len(manifest['targets'])} targets")
for t in manifest["targets"][:5]:
    print(f"  {t['file']}  type={t['type']}")
