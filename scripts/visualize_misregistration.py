from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from misregistration.simulation.channel_shift_simulator import simulate_from_png

# --------------------------------------------------
# Choose image
# --------------------------------------------------

IMAGE = r"data\Copy of Copy of kodim20.png"

# --------------------------------------------------
# Simulate misregistration
# --------------------------------------------------

result = simulate_from_png(
    IMAGE,
    shifts_px={
        "C": (8, -4),
        "M": (-6, 5),
        "Y": (10, 2),
        "K": (0, 0),
    }
)

original = result.reference_rgb
misreg = result.shifted_rgb
de_map = result.dE_map

# --------------------------------------------------
# Difference image
# --------------------------------------------------

diff = np.abs(
    misreg.astype(np.float32)
    - original.astype(np.float32)
)

# grayscale difference magnitude
diff_gray = diff.mean(axis=2)

# contrast stretch
p99 = np.percentile(diff_gray, 99)

if p99 > 0:
    diff_vis = np.clip(diff_gray / p99, 0, 1)
else:
    diff_vis = diff_gray

# uint8 version for saving
diff_uint8 = (diff_vis * 255).astype(np.uint8)

# --------------------------------------------------
# Output folder
# --------------------------------------------------

outdir = Path("results/visual_demo")
outdir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Save images
# --------------------------------------------------

Image.fromarray(original).save(
    outdir / "original.png"
)

Image.fromarray(misreg).save(
    outdir / "misregistered.png"
)

Image.fromarray(diff_uint8).save(
    outdir / "difference.png"
)

# --------------------------------------------------
# Save ΔE map
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(
    de_map,
    cmap="hot"
)

plt.colorbar(label="ΔE00")

plt.title(
    "ΔE Misregistration Map"
)

plt.tight_layout()

plt.savefig(
    outdir / "deltaE_map.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Combined figure
# --------------------------------------------------

fig, ax = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

# Original
ax[0, 0].imshow(original)
ax[0, 0].set_title("Original")

# Misregistered
ax[0, 1].imshow(misreg)
ax[0, 1].set_title("Misregistered")

# Difference heatmap
ax[1, 0].imshow(
    diff_vis,
    cmap="hot"
)

ax[1, 0].set_title(
    "Difference Magnitude"
)

# ΔE map
im = ax[1, 1].imshow(
    de_map,
    cmap="hot"
)

ax[1, 1].set_title(
    f"ΔE Map (mean={result.dE_image:.2f})"
)

for a in ax.ravel():
    a.axis("off")

fig.colorbar(
    im,
    ax=ax[1, 1]
)

plt.tight_layout()

plt.savefig(
    outdir / "comparison_figure.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Statistics
# --------------------------------------------------

print("\nSimulation Statistics")
print("-" * 40)

print(
    f"Mean ΔE = {result.dE_image:.3f}"
)

for ch, rms in result.rms_px.items():
    print(
        f"{ch}: RMS shift = {rms:.3f} px"
    )

print(
    f"Difference image max = {diff_gray.max():.3f}"
)

print(
    f"Difference image mean = {diff_gray.mean():.3f}"
)

print(
    f"Difference image p99 = {p99:.3f}"
)

print(
    "\nSaved to:",
    outdir
)