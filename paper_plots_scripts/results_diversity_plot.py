import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Configuration & Data
# ==========================================

# -- Style Configuration for Publication --
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 23,
        "axes.labelsize": 25,
        "axes.titlesize": 27,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "legend.fontsize": 23,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.figsize": (24, 5),  # Wide format for side-by-side
    }
)

# -- Data Hardcoding --
# Structure: data[concept][metric][model_key]
raw_data = {
    "Monster": {
        "clip": {"sdxl-dmd": 0.2556, "sae": 0.2564, "pca": 0.2518, "sdxl": 0.2207, "sliders": 0.2563},
        "dreamsim": {"sdxl-dmd": 0.3194, "sae": 0.5602, "pca": 0.5325, "sdxl": 0.2803, "sliders": 0.5257},
    },
    "Person": {
        "clip": {"sdxl-dmd": 0.2478, "sae": 0.2399, "pca": 0.2404, "sdxl": 0.1925, "sliders": 0.2498},
        "dreamsim": {"sdxl-dmd": 0.4081, "sae": 0.6300, "pca": 0.6260, "sdxl": 0.5779, "sliders": 0.5631},
    },
    "Dog": {
        "clip": {"sdxl-dmd": 0.2701, "sae": 0.2703, "pca": 0.2609, "sdxl": 0.2290, "sliders": 0.2687},
        "dreamsim": {"sdxl-dmd": 0.2326, "sae": 0.5282, "pca": 0.5460, "sdxl": 0.4146, "sliders": 0.5118},
    },
    "Cat": {
        "clip": {"sdxl-dmd": 0.2754, "sae": 0.2747, "pca": 0.2709, "sdxl": 0.2436, "sliders": 0.2756},
        "dreamsim": {"sdxl-dmd": 0.1678, "sae": 0.3631, "pca": 0.3897, "sdxl": 0.2933, "sliders": 0.2926},
    },
    "Car": {
        "clip": {"sdxl-dmd": 0.2545, "sae": 0.2516, "pca": 0.2517, "sdxl": 0.2212, "sliders": 0.2571},
        "dreamsim": {"sdxl-dmd": 0.3452, "sae": 0.5105, "pca": 0.4705, "sdxl": 0.4528, "sliders": 0.5273},
    },
}

# -- Control Panel --
# Add or remove keys from this list to change the plot
MODELS_TO_INCLUDE = [
    "sdxl-dmd",
    "sliders",
    "pca",
    "sae",
    # "sdxl",
]

# Display names for the legend (map keys to nice names)
MODEL_LABELS = {
    "sdxl-dmd": "SDXL-DMD",
    "sae": "ELROND (SAE)",
    "pca": "ELROND (PCA)",
    "sdxl": "SDXL Base",
    "sliders": "SliderSpace",
}

# Colors for the bars (Publication-safe palette)
COLORS = ["#3448ad", "#c3ddde", "#4cba76", "#f26c6c", "#e4e0cf"]


# ==========================================
# 2. Plotting Logic
# ==========================================
def create_comparison_plots():
    concepts = list(raw_data.keys())
    metrics = ["dreamsim", "clip"]
    metric_titles = ["Diversity", "Text Alignment"]

    n_groups = len(concepts)
    n_models = len(MODELS_TO_INCLUDE)

    # Calculate bar positions
    # We create an index for each concept [0, 1, 2, 3, 4]
    index = np.arange(n_groups)

    # Total width allocated for one group of bars
    total_bar_width = 0.8
    single_bar_width = total_bar_width / n_models

    fig, axes = plt.subplots(1, 2, figsize=(20, 4.2))  # 1 row, 2 columns

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Loop through each model to plot its bar for all concepts
        for j, model_key in enumerate(MODELS_TO_INCLUDE):

            # Extract scores for this specific model across all concepts
            scores = []
            for concept in concepts:
                val = raw_data[concept][metric].get(model_key, 0)
                scores.append(val)

            # Calculate offset for this model's bar
            # Center the group of bars on the tick mark
            offset = (j - n_models / 2) * single_bar_width + (single_bar_width / 2)

            ax.bar(
                index + offset,
                scores,
                width=single_bar_width,
                label=MODEL_LABELS.get(model_key, model_key),
                color=COLORS[j % len(COLORS)],
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
                alpha=0.75,
            )  # zorder=3 ensures bars are above grid lines

        # Formatting the specific subplot
        ax.set_ylabel(metric_titles[i], fontweight="bold")
        ax.set_xticks(index)
        ax.set_xticklabels(concepts, rotation=30, ha="right", rotation_mode="anchor")

        # Add a subtle grid behind the bars
        ax.grid(axis="y", zorder=0)

        # Determine logical Y limits (add a little headroom)
        # Find max value in this metric to set ylim
        all_vals = []
        for c in concepts:
            for m in MODELS_TO_INCLUDE:
                all_vals.append(raw_data[c][metric][m])
        max_val = max(all_vals)
        ax.set_ylim(0, max_val * 1.15)  # 15% headroom for legend/clarity

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Add a single legend for the whole figure (since models are the same)
    # We place it at the top center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_models, frameon=False, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()

    # Save nicely
    plt.savefig("metric_comparison_plot.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("metric_comparison_plot.png", bbox_inches="tight", dpi=300)
    print("Plots saved as 'metric_comparison_plot.pdf' and '.png'")
    plt.show()


if __name__ == "__main__":
    create_comparison_plots()
