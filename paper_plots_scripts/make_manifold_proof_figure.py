import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
    }
)

# File paths for the 2 input images
IMAGE_PATHS = [
    "outputs/intreventions/sdxl-cat-random/pca_dirs_sdxl_1.png",
    "outputs/intreventions/sdxl-cat-sdxl-dmd/pca_dirs_sdxl_1.png",
]

# Labels for the top of the images
TITLES = ["Random direction", "Discovered direction"]

# Text for the side label
SIDE_LABEL_TEXT = "Intervention Strength"

# Output filename
OUTPUT_FILE = "intervention_plot.pdf"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================


def load_and_crop(path):
    """
    Loads an image and crops it:
    - Width: 1/2 (Left half)
    - Height: 5/6 (Top portion)
    """
    if os.path.exists(path):
        img = Image.open(path)
    else:
        # Create a dummy image if file is missing (Gradient for visibility)
        print(f"Warning: {path} not found. Using placeholder.")
        w, h = 400, 600
        arr = np.linspace(0, 255, w * h).reshape((h, w))
        img = Image.fromarray(np.stack([arr] * 3, axis=-1).astype("uint8"))

    # Get dimensions
    width, height = img.size

    # Calculate crop box: (left, top, right, bottom)
    # 5/6 height, 1/2 width
    crop_box = (0, 0, int(width * 0.5), int(height * (4 / 5)))

    return img.crop(crop_box)


# ==========================================
# 3. PLOTTING LOGIC
# ==========================================


def create_intervention_plot():
    # Setup Figure
    # Adjust figsize to match your desired aspect ratio
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Ensure axes is iterable if only 1 image (though we strictly set 2)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, ax in enumerate(axes):
        # 1. Load and Crop
        img_path = IMAGE_PATHS[i] if i < len(IMAGE_PATHS) else "missing.png"
        img = load_and_crop(img_path)

        # 2. Display Image
        ax.imshow(img)
        ax.axis("off")  # Hide default pixel axes

        # 3. Add Top Label
        title = TITLES[i] if i < len(TITLES) else "Label"
        ax.set_title(title, fontsize=24, pad=20, fontweight="bold")

        # 4. Add "Intervention Strength" Arrow and Label
        # We draw relative to the Axes coordinates (0,0 is bottom-left, 1,1 is top-right)

        # ARROW:
        # xy = Tip of arrow (Bottom-Left: x=-0.05, y=0)
        # xytext = Start of arrow (Top-Left: x=-0.05, y=1)
        # 'coordsA' and 'coordsB' set to "axes fraction" ensures it stays relative to the image box
        ax.annotate(
            "",
            xy=(-0.03, 0),
            xycoords="axes fraction",  # Tip (Bottom)
            xytext=(-0.03, 1),
            textcoords="axes fraction",  # Tail (Top)
            arrowprops=dict(arrowstyle="->, head_width=1, head_length=1.7", color="black", lw=4.5),
        )

        # SIDE LABEL:
        # Placed slightly to the left of the arrow (-0.12), centered vertically (0.5)
        ax.text(-0.07, 0.5, SIDE_LABEL_TEXT, rotation=90, va="center", ha="center", transform=ax.transAxes, fontsize=24)

    # Save output
    plt.tight_layout()
    # Extra pad ensures the left-side text isn't cut off
    plt.savefig(OUTPUT_FILE, bbox_inches="tight", dpi=300)
    print(f"Plot saved to {OUTPUT_FILE}")
    plt.show()


if __name__ == "__main__":
    create_intervention_plot()
