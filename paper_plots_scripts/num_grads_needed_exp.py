import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- PLOTTING CONFIGURATION (Professional/Camera-Ready) ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset": "cm",
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "lines.linewidth": 2.5,
        "lines.markersize": 7,
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


class GPUPCA:
    """
    A minimal PCA implementation using PyTorch for GPU acceleration.
    """

    def __init__(self):
        self.components_ = None
        self.explained_variance_ = None  # Raw eigenvalues
        self.explained_variance_ratio_ = None  # Normalized (sums to 1)

    def fit(self, X):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if X.device != device:
            X = X.to(device)

        n_samples, n_features = X.shape

        # Center Data
        mean_ = torch.mean(X, dim=0)
        X_centered = X - mean_

        # SVD
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        # Variances
        eigenvalues = (S**2) / (n_samples - 1)
        self.explained_variance_ = eigenvalues

        # Normalize
        total_variance = torch.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance

        # Components
        self.components_ = Vh
        return self


def load_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.npy"))
    data_list = []
    print(f"Found {len(files)} files in {folder_path}...")
    for f in files:
        try:
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            data_list.append(arr)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not data_list:
        raise ValueError("No data found.")

    return np.concatenate(data_list, axis=0)


def get_significant_subspace_gpu(data_tensor, var_threshold_factor=2.5):
    """
    Returns basis vectors where Explained Variance Ratio > 2.5 / dim.
    """
    n_samples, dim = data_tensor.shape

    pca = GPUPCA()
    pca.fit(data_tensor)

    # Threshold based on normalized ratio
    threshold = var_threshold_factor / dim
    significant_mask = pca.explained_variance_ratio_ > threshold

    if not torch.any(significant_mask):
        return None, 0.0

    basis = pca.components_[significant_mask].T
    total_var_covered = pca.explained_variance_ratio_[significant_mask].sum().item()

    return basis, total_var_covered


def subspace_overlap_gpu(basis_target, basis_source):
    if basis_target is None or basis_source is None:
        return 0.0

    # Project Target onto Source
    coeffs = torch.matmul(basis_source.T, basis_target)
    projection_energies = torch.sum(coeffs**2, dim=0)
    return torch.mean(projection_energies).item()


def main():
    # --- CONFIG ---
    FOLDER_PATH = "/data/pskiers/concept_dimention/grads/sdxl-monster/grads/t50/shards/"
    MAX_VECTORS = 50000

    # LINEAR SCALE with higher density
    # Combines small steps (for the initial curve) and large steps (for the plateau)
    small_steps = [100, 500, 1000, 2500]
    large_steps = list(range(5000, MAX_VECTORS + 1, 2500))  # 5k, 7.5k, ... 30k
    SUBSET_SIZES = sorted(list(set(small_steps + large_steps)))

    THRESHOLD_FACTOR = 2.5
    OUTPUT_FILENAME = "subspace_analysis_linear"

    # 1. Load Data
    try:
        X_numpy = load_data(FOLDER_PATH)
    except ValueError as e:
        print(e)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X = torch.tensor(X_numpy, dtype=torch.float32).to(device)

    # 2. Reference Subspace
    if len(X) > MAX_VECTORS:
        indices = torch.randperm(len(X))[:MAX_VECTORS]
        X_ref = X[indices]
    else:
        X_ref = X
        print(f"Note: Using all {len(X)} vectors for reference.")

    print(f"Computing Ref Subspace (N={len(X_ref)})...")
    basis_ref, var_ref = get_significant_subspace_gpu(X_ref, THRESHOLD_FACTOR)

    if basis_ref is None:
        print("Error: Reference space is empty.")
        return

    print(f"Reference Dim: {basis_ref.shape[1]}")

    # 3. Analyze Subsets
    results_coverage = []
    results_leakage = []
    valid_sizes = []

    print("\nProcessing subsets...")
    for size in SUBSET_SIZES:
        if size > len(X):
            continue
        valid_sizes.append(size)

        # Sample
        indices = torch.randperm(len(X))[:size]
        X_sub = X[indices]

        # PCA
        basis_sub, _ = get_significant_subspace_gpu(X_sub, THRESHOLD_FACTOR)

        if basis_sub is None:
            results_coverage.append(0.0)
            results_leakage.append(0.0)
            continue

        # Overlap Logic
        cov = subspace_overlap_gpu(basis_target=basis_ref, basis_source=basis_sub)
        reverse_cov = subspace_overlap_gpu(basis_target=basis_sub, basis_source=basis_ref)
        leak = 1.0 - reverse_cov

        results_coverage.append(cov)
        results_leakage.append(leak)

        print(f"N={size:5d} | Cov: {cov:.3f} | Noise: {leak:.3f}")

    # --- PLOTTING (LINEAR SCALE) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    color_cov = "#004c6d"
    color_noise = "#a70000"

    # Plot 1: Coverage
    ax1.plot(valid_sizes, results_coverage, marker="o", color=color_cov, label="Subspace Span")
    ax1.set_xlabel("Number of Vectors ($N$)", fontweight="bold")
    ax1.set_ylabel("Fraction of space spanned", fontweight="bold", color=color_cov)
    ax1.set_title("Subspace Reconstruction", fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)

    # Linear Scale formatting
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k" if x >= 1000 else int(x)))
    ax1.tick_params(axis="y", labelcolor=color_cov)
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.7)

    # Plot 2: Noise
    ax2.plot(valid_sizes, results_leakage, marker="s", color=color_noise, label="Artifacts")
    ax2.set_xlabel("Number of Vectors ($N$)", fontweight="bold")
    ax2.set_ylabel("Fraction of noise", fontweight="bold", color=color_noise)
    ax2.set_title('Subspace "Hallucinations"', fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)

    # Linear Scale formatting
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k" if x >= 1000 else int(x)))
    ax2.tick_params(axis="y", labelcolor=color_noise)

    # Despine
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FILENAME}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{OUTPUT_FILENAME}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"\nPlots saved to {OUTPUT_FILENAME}.pdf")


if __name__ == "__main__":
    main()
