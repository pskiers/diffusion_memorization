import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================

# OUTPUT SETTINGS
OUTPUT_FILENAME = "grid_figure.pdf"  # Saves as vector graphic for LaTeX
FIG_SIZE = (24, 9)  # Wide aspect ratio (approx 3:1)
FONT_FAMILY = "serif"  # 'serif' = Times New Roman style
FONT_SIZE_LABEL = 24  # Large font for main labels

# DIRECTORIES (Placeholders)
# Map your dataset keys to real paths
DIRS = {
    "TopLeft": "/home/pskiers/LID-project/data/sdxl_dmd_person",
    "TopRight": "/data/pskiers/concept_dimention/sdxl-dmd-person-samples-with-directions-from-sdxl-sae",
    "BotLeft": "/home/pskiers/LID-project/data/sdxl_dmd_dog",
    "BotRight": "/data/pskiers/concept_dimention/sdxl-dmd-dog-samples-with-directions-sae",
}

# LABELS
# The labels for the main 2x2 grid
COL_LABELS = ["SDXL-DMD", "SDXL-DMD Enhanced (ELROND)"]  # Top Labels
ROW_LABELS = ["Person", "Dog"]  # Side Labels

# IMAGE LISTS (Placeholders)
# Replace these lists with your actual filenames.
# Each list must have 12 filenames (for the 2 rows x 6 columns grid)
# Order: Top-Left(0) -> Top-Right(5), then Bot-Left(6) -> Bot-Right(11)
IMG_LISTS = {
    "TopLeft": [f"sample_0000_{i:02d}.png" for i in range(12)],
    "TopRight": [f"{i}.png" for i in range(12)],
    "BotLeft": [f"sample_0000_{i:02d}.png" for i in range(12)],
    "BotRight": [f"{i}.png" for i in range(6)] + [f"{i}.png" for i in range(18, 24)],
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================


def load_image(folder, filename):
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        img = Image.open(path)
        rgb_im = img.convert("RGB")
        rgb_im.save(folder + filename[:-3] + "jpg")
        return Image.open(folder + filename[:-3] + "jpg")
    else:
        # Return a stylish placeholder if file is missing
        # Dark gray background with a lighter border to see grid structure
        arr = np.ones((256, 256, 3), dtype=np.uint8) * 240
        arr[0:5, :, :] = 100  # Border
        arr[-5:, :, :] = 100
        arr[:, 0:5, :] = 100
        arr[:, -5:, :] = 100
        return Image.fromarray(arr)


# ==========================================
# 3. PLOTTING LOGIC
# ==========================================


def create_complex_grid():
    # Use LaTeX font family
    plt.rcParams.update({"font.family": FONT_FAMILY})

    fig = plt.figure(figsize=FIG_SIZE)

    # OUTER GRID: 2 Rows x 2 Columns
    # wspace/hspace creates the separation between the 4 main blocks
    outer_grid = gridspec.GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.15)

    # Keys to access the configuration dictionaries
    keys = [["TopLeft", "TopRight"], ["BotLeft", "BotRight"]]

    for row in range(2):
        for col in range(2):
            key = keys[row][col]
            folder = DIRS[key]
            filenames = IMG_LISTS[key]

            # INNER GRID: 2 Rows x 6 Columns (Total 12 images)
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                2, 6, subplot_spec=outer_grid[row, col], wspace=0.02, hspace=0.02  # Tight packing for images
            )

            # Plot the 12 images for this block
            for idx, filename in enumerate(filenames):
                if idx >= 12:
                    break  # Safety break

                ax = fig.add_subplot(inner_grid[idx])
                img = load_image(folder, filename)
                ax.imshow(img)
                ax.axis("off")

            # --- LABELS ---

            # COLUMN LABELS (Only on the very top row of the outer grid)
            if row == 0:
                ax_col = fig.add_subplot(outer_grid[row, col], frameon=False)
                ax_col.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
                ax_col.set_title(COL_LABELS[col], fontsize=FONT_SIZE_LABEL, pad=20, fontweight="bold")

            # ROW LABELS (Only on the very left column of the outer grid)
            if col == 0:
                ax_row = fig.add_subplot(outer_grid[row, col], frameon=False)
                ax_row.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
                # Position coordinates (-0.05, 0.5) place it to the left, centered vertically
                ax_row.text(
                    -0.05,
                    0.5,
                    ROW_LABELS[row],
                    va="center",
                    ha="right",
                    rotation=90,
                    fontsize=FONT_SIZE_LABEL,
                    fontweight="bold",
                    transform=ax_row.transAxes,
                )

    # Save to PDF for LaTeX
    plt.savefig(OUTPUT_FILENAME, bbox_inches="tight", dpi=300)
    print(f"Successfully generated {OUTPUT_FILENAME}")
    plt.show()


if __name__ == "__main__":
    create_complex_grid()
