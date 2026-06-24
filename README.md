# misregistration-benchmark

**Hardware-agnostic CMYK channel misregistration benchmark** for halftone print-and-scan evaluation.

Implements the channel misregistration metrics from the *Halftoning Benchmark Specification*:

| Metric | Definition |
|--------|-----------|
| `RMS_shift_k` (mm) | Per-channel geometric displacement (C, M, Y relative to K reference) via NCC |
| `ΔE_mis(patch)` | CIEDE2000 error on individual overprint patches |
| `ΔE_mis(image)` | Mean CIEDE2000 error over full image after shift |

---

## Quick Start

```bash
pip install -e .

# Smoke test (no printer required — auto-generates demo images):
python scripts/smoke_test.py --demo

# Or test on your own images:
python scripts/smoke_test.py --dataset dataset/
```

---

## Smoke Test (No Hardware Required)

Place any RGB PNG/JPG images in `dataset/` and run:

```bash
python scripts/smoke_test.py --dataset dataset/
```

Pipeline per image:
1. Load image (PNG / JPG / TIFF)
2. Generate synthetic CMYK separations (device-independent RGB→CMY+K with UCR)
3. Apply known channel shifts (configurable, default: C=1.5px, M=−0.5px, Y=2.0px)
4. Reconstruct shifted composite image
5. Estimate `dx`, `dy`, `RMS shift` per channel via Normalised Cross-Correlation
6. Estimate `ΔE_mis` via CIEDE2000
7. Generate summary plots (bar chart, shift vectors, ΔE heatmap)
8. Export CSV

Expected output:
```
C: true_rms=1.700px  est_rms=1.700px  err=0.000px [OK]
M: true_rms=1.300px  est_rms=1.300px  err=0.000px [OK]
Y: true_rms=2.022px  est_rms=2.022px  err=0.000px [OK]
ΔE_mis(image) = 14.84
Smoke test: 3 passed, 0 failed
```

---

## Dataset Benchmark (450 Images)

```bash
# Flat folder (all images treated as one method):
python scripts/run_dataset_benchmark.py dataset/ --dpi 300

# One sub-folder per halftoning method:
python scripts/run_dataset_benchmark.py dataset/ --by-method --dpi 300

# Quick test (10 images per method):
python scripts/run_dataset_benchmark.py dataset/ --by-method --max-images 10 --dpi 300

# With custom shifts:
python scripts/run_dataset_benchmark.py dataset/ --dpi 300 \
  --shifts '{"C":[1.5,-0.8],"M":[-0.5,1.2],"Y":[2.0,0.3],"K":[0,0]}'
```

Outputs in `results/`:

| File | Description |
|------|-------------|
| `per_image.csv` | One row per (image, channel): dx, dy, rms_px, rms_mm, dE |
| `per_method.csv` | Per-channel mean ± 95% CI per method |
| `summary.csv` | Worst-case RMS and ΔE per method |
| `benchmark_table.csv` | **Table 3** from the benchmark spec |
| `pareto.csv` | Pareto-optimal methods |
| `pareto_plot.png/.svg` | Pareto frontier scatter plot |

---

## Print-and-Scan Workflow (Physical Hardware)

1. **Generate targets:**
   ```bash
   python scripts/generate_targets.py --dpi 4800
   ```
   Outputs: bullseye, crosshair, corner fiducials, line targets, overprint patches.

2. **Print the targets** on your device at the configured DPI.

3. **Scan the prints** at ≥300 dpi (optical), save as TIFF.

4. **Validate the scan:**
   ```bash
   python scripts/validate_scan.py scan.tiff
   ```

5. **Measure misregistration:**
   ```bash
   python scripts/measure_misregistration_from_scan.py scan.tiff --output-dir results/
   ```

6. **Batch process:**
   ```bash
   python scripts/batch_measure_misregistration.py scans/ --output-dir results/
   ```

DPI is read from TIFF `XResolution`/`YResolution` tags automatically.
Pass `--dpi` only if the TIFF lacks metadata.

---

## Expected Outputs

```
results/
  smoke_test/
    demo_000/
      summary_rms_bar.png
      summary_shift_vectors.png
      summary_dE_map.png
    smoke_test_results.csv
  dataset_benchmark/
    per_image.csv
    per_method.csv
    benchmark_table.csv
    pareto.csv
    pareto_plot.png
    pareto_plot.svg
```

---

## Running Tests

```bash
pytest -v
# Expected: 49 passed
```

---

## Windows PowerShell

All commands work identically on Windows PowerShell:

```powershell
pip install -e .
python scripts/smoke_test.py --demo
python scripts/run_dataset_benchmark.py dataset\ --by-method --dpi 300
pytest -v
```

Path separators are handled by `pathlib.Path` throughout — no platform-specific code.

---

## Architecture

```
misregistration/
  io/                  TIFF/PNG/JPG loading; metadata from TIFF tags
  preprocessing/       ECC geometric alignment; RGB→CMYK separation
  measurement/
    ncc.py             Normalised Cross-Correlation (Eq. 3–4)
    shift.py           Shift estimation; RMS computation (Eq. 2)
    fiducial.py        Blob detection; centroid estimation
    colorimetric.py    CIELAB; CIEDE2000 (pure-numpy + colour-science)
  calibration/
    pixel_scale.py     px/mm from TIFF metadata or calibration targets
    session_baseline.py  Zero-shift baseline correction
  calibration_targets/ Bullseye, crosshair, corner fiducials, overprint patches
  aggregation/
    statistics.py      Mean, 95% CI, Table 3 generation
    pareto.py          Pareto frontier analysis
  visualization/
    misregistration_plots.py  Bar charts, dE heatmaps, NCC surfaces
    vector_field.py    Quiver maps, displacement overlays
  simulation/
    channel_shift_simulator.py  Full CMYK shift pipeline
```

## Hardware Independence

All device parameters come from runtime sources only:

| Parameter | Source |
|-----------|--------|
| Scanner DPI | TIFF `XResolution`/`YResolution` tags |
| Pixel scale (mm/px) | Fiducial-based calibration or `--dpi` argument |
| Bit depth | TIFF `BitsPerSample` tag |
| Shift values | Config file or `--shifts` JSON argument |

No printer model, scanner model, paper type, or ICC profile is hardcoded.

---

## Troubleshooting

**`No images found`** — Check that images have extensions `.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff`.

**`dpi=nan` in CSV** — Pass `--dpi 300` (or your scanner's actual DPI) to enable mm conversion.

**Low NCC peak (<0.3)** — Image may lack sufficient spatial structure. Try a more detailed image.

**`colour` import warning** — The package falls back to a pure-numpy CIEDE2000 implementation if `colour-science` is unavailable. Results are equivalent.
