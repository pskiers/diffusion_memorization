import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 23,
        "axes.labelsize": 29,
        "axes.titlesize": 27,
        "xtick.labelsize": 27,
        "ytick.labelsize": 27,
        "legend.fontsize": 23,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.figsize": (24, 5),  # Wide format for side-by-side
    }
)

# ================= CONFIGURATION =================
# Path to your images folder
IMAGE_FOLDER = "./outputs/failures_sdxl_dmd"

# The 3 classes (Rows)
OBJECTS = ["car", "person"]

# The 4 main columns (Original + i1, i2, i3)
# 'og' maps to filename flux_{obj}_og{m}.jpg
# '1','2','3' map to filename flux_{obj}_i{n}_{m}.jpg
COLUMN_KEYS = ["og", "1", "2", "3"]

# Custom Labels for the ROWS (3 items)
ROW_LABELS = ["Car", "Person"]

# Custom Labels for the GRIDS (3 rows x 4 columns)
# Edit these to change the title above each 2x2 grid
GRID_LABELS = [
    # Row 1: Dog
    # ["Original", "Lying", "Long fur", "German shepherd"],
    # Row 2: Car
    ["Original", "Direction 1", "Drection 2", "Direction 3"],
    # Row 3: Person
    ["Original", "Direction 1", "Drection 2", "Direction 3"],
]

# Image settings
IMG_SIZE = (256, 256)  # Resize all images to this size for consistent grids
# =================================================


def get_image_path(folder, obj, col_key, m):
    """Generates the filename based on the naming convention."""
    if col_key == "og":
        filename = f"failure_{obj}_og{m}.jpg"
    else:
        filename = f"failure_{obj}_i{col_key}_{m}.jpg"
    return os.path.join(folder, filename)


def create_stitched_grid(folder, obj, col_key):
    """
    Loads 4 images (m=1,2,3,4) and stitches them into a single 2x2 numpy array.
    Returns the stitched image.
    """
    sub_images = []

    # We expect m = 1, 2, 3, 4
    for m in range(1, 5):
        path = get_image_path(folder, obj, col_key, m)

        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(IMG_SIZE)
                sub_images.append(np.array(img))
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Placeholder for corrupted image
                sub_images.append(np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8))
        else:
            print(f"Warning: Missing file {path}")
            # Placeholder black square for missing file
            sub_images.append(np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8))

    # Stitch: [1, 2]
    #         [3, 4]
    top_row = np.hstack((sub_images[0], sub_images[1]))
    bot_row = np.hstack((sub_images[2], sub_images[3]))
    full_grid = np.vstack((top_row, bot_row))

    return full_grid


def main():
    # Create the main figure with 3 rows and 4 columns
    # Adjust figsize to control the aspect ratio (width, height)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 12))

    # Adjust spacing
    plt.subplots_adjust(wspace=-5.1, hspace=0.1)

    for r, obj in enumerate(OBJECTS):
        for c, col_key in enumerate(COLUMN_KEYS):
            ax = axes[r, c]

            # 1. Create and display the 2x2 image grid
            stitched_img = create_stitched_grid(IMAGE_FOLDER, obj, col_key)
            ax.imshow(stitched_img)

            # 2. Set the Custom Grid Label (Top)
            try:
                label = GRID_LABELS[r][c]
            except IndexError:
                label = "Label Missing"
            ax.set_title(label, fontsize=24, fontweight="bold", pad=10)

            # 3. Handle Row Labels (Left of the first column)
            if c == 0:
                # We use the ylabel of the first column's axis
                ax.set_ylabel(ROW_LABELS[r], rotation=90, size="large", fontweight="bold", labelpad=20)

            # 4. Clean up axes
            # We remove ticks but keep the frame for a clean look,
            # or turn axis completely off if you prefer no border.
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.axis('off') # Uncomment this line to remove borders entirely

    print("Plot generated. Saving to 'output_grid.png'...")
    plt.tight_layout()
    plt.savefig("output_grid.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
