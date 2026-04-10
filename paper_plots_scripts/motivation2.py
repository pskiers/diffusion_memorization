import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

# --- Configuration for Single-Column Publication ---
# Width is set to 3.5 inches (standard single column).
# Height is adjusted to keep images roughly square-ish based on 5 rows.
FIG_SIZE = (3.5, 3.5)

# The exact same color palette used in the trajectory plot
# Sequence 1 (Blue), Seq 2 (Red), Seq 3 (Green), Seq 4 (Purple)
COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B3"]

# These settings ensure high readability in papers
plt.rcParams.update(
    {
        "font.family": "serif",  # Matches LaTeX/Paper font
        "font.size": 14,  # Base font size
        "axes.labelsize": 16,  # Axis label size
        "axes.titlesize": 0,  # No title displayed
        "xtick.labelsize": 14,  # X-tick size
        "ytick.labelsize": 14,  # Y-tick size
        "axes.linewidth": 1.5,  # Thicker axis frame lines
        "lines.linewidth": 0.5,  # Thicker data lines
        "xtick.major.width": 1.5,  # Thicker tick marks
        "ytick.major.width": 1.5,
    }
)

# --- 1. Single-Column Publication Config ---
# Targeted for a width of 3.3 to 3.5 inches (standard 2-column paper width)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,  # Standard caption font size
        "axes.labelsize": 10,  # Slightly larger for axes
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,  # Compact legend
        "axes.linewidth": 1.0,  # Thinner spines for small plots
        "lines.linewidth": 1.5,  # Balanced line width
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.color": "#bdc3c7",
        "grid.linestyle": "--",
        "figure.dpi": 300,  # High resolution for screen check
    }
)


def plot_image_grid(sequences, intervention_strengths, save_path="sequence_grid.pdf"):
    """
    Generates a publication-ready grid of image sequences.
    """
    batch_size = len(sequences)
    if batch_size == 0 or len(sequences[0]) == 0:
        print("No data to plot.")
        return

    # --- 1. Subsampling Steps ---
    # We cannot show 17 rows in a single column.
    # We select 5 key representative steps: Start, and 4 evenly spaced intervals.
    # Indices corresponds to strengths: 0.0, 0.4, 0.8, 1.2, 1.6
    indices_to_plot = [
        [0, 2, 5, 7, 10],
        [0, 2, 5, 7, 10],
        [0, 2, 5, 7, 10],
        [0, 2, 5, 7, 10],
    ]

    num_rows = len(indices_to_plot[0])
    num_cols = batch_size

    # --- 2. Setup Grid ---
    # gridspec_kw={'wspace': 0.05, 'hspace': 0.05} ensures very tight spacing
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=FIG_SIZE,
        gridspec_kw={"wspace": 0.05, "hspace": 0.0, "top": 0.93, "bottom": 0.05, "left": 0.15, "right": 0.98},
    )

    for col_idx in range(num_cols):
        # Use corresponding color for the sequence
        seq_color = COLORS[col_idx % len(COLORS)]

        # --- Column Header (Visual Link) ---
        # We label the top axes of each column with colored text
        header_ax = axes[0, col_idx]
        header_ax.set_title(f"Seq {col_idx+1}", color=seq_color, fontweight="bold", pad=8)

        for row_idx, step_idx in enumerate(indices_to_plot[col_idx]):
            ax = axes[row_idx, col_idx]

            # Safety check if sequence didn't load completely
            if step_idx < len(sequences[col_idx]):
                img = sequences[col_idx][step_idx]
                ax.imshow(img)
            else:
                # Placeholder if missing data
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                ax.set_facecolor("#f0f0f0")

            # --- Clean up Axes ---
            ax.set_xticks([])
            ax.set_yticks([])
            # Remove spines for a cleaner look (images floating)
            for spine in ax.spines.values():
                spine.set_visible(False)

            # --- Row Labels (Left Side Only) ---
            if col_idx == 0:
                strength_val = intervention_strengths[step_idx]
                # Rotate label 90 degrees for compactness on the left
                ax.set_ylabel(
                    f"$t={strength_val:.1f}$",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=5,
                    fontweight="bold",
                    color="#555555",
                )

    # No plt.tight_layout() here because we manually tuned gridspec_kw
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Publication-ready image grid saved to: {save_path}")


# =========================================
# Main Data Loading Block (From your snippet)
# =========================================
if __name__ == "__main__":
    intervention_strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    batch_size = 4

    sequences = [[] for _ in range(batch_size)]

    data_found = False
    for i in range(batch_size):
        for j in range(len(intervention_strengths)):
            try:
                img_path = f"outputs/motivation/seq{i}_step{j}.png"
                image = Image.open(img_path)

                rgb_im = image.convert("RGB")
                rgb_im.save(f"outputs/motivation/seq{i}_step{j}.jpg")

                image = Image.open(f"outputs/motivation/seq{i}_step{j}.jpg")

                sequences[i].append(image)
                data_found = True
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    if data_found and len(sequences[0]) > 0:
        plot_image_grid(sequences, intervention_strengths, save_path="sequence_visuals.pdf")
    else:
        print("Could not load enough data to plot.")
