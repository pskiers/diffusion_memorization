"""
Compare subspaces built from two disjoint image subsets.

For each gradient in the dataset, the "first image" of the pair that
produced it determines which subset that gradient belongs to.
We split all unique first-images into two equal halves, collect the
corresponding gradients, run PCA on each half, and report subspace
coverage and hallucination between the two halves.
"""

import os
import json
import glob
import random
import argparse

import numpy as np
import torch


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class GPUPCA:
    def __init__(self):
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = X.to(device)

        n_samples, _ = X.shape
        X_centered = X - X.mean(dim=0)
        _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        eigenvalues = (S ** 2) / (n_samples - 1)
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues / total_variance
        self.components_ = Vh
        return self


def get_significant_subspace(data_tensor, var_threshold_factor=2.5):
    """Returns column basis where explained variance ratio > 2.5 / dim."""
    _, dim = data_tensor.shape
    pca = GPUPCA()
    pca.fit(data_tensor)

    threshold = var_threshold_factor / dim
    mask = pca.explained_variance_ratio_ > threshold
    if not torch.any(mask):
        return None, 0.0

    basis = pca.components_[mask].T          # shape (dim, k)
    total_var_covered = pca.explained_variance_ratio_[mask].sum().item()
    return basis, total_var_covered


def subspace_overlap(basis_target, basis_source):
    """Mean squared cosine between each target vector and the source subspace."""
    if basis_target is None or basis_source is None:
        return 0.0
    coeffs = torch.matmul(basis_source.T, basis_target)   # (k_src, k_tgt)
    projection_energies = torch.sum(coeffs ** 2, dim=0)   # (k_tgt,)
    return torch.mean(projection_energies).item()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset_info(grads_dir):
    """
    Returns a flat list of (shard_index_0based, row_within_shard, first_img, second_img).
    shard_index_0based=0 → shard_1.npy
    """
    info_path = os.path.join(grads_dir, "dataset_info.json")
    with open(info_path) as f:
        info = json.load(f)

    records = []
    for shard_idx, shard_pairs in enumerate(info["shards"]):
        for row_idx, (img_a, img_b) in enumerate(shard_pairs):
            records.append((shard_idx, row_idx, img_a, img_b))
    return records


def load_grads_for_indices(grads_dir, indices_by_shard):
    """
    indices_by_shard: dict mapping shard_idx (0-based) -> list of row indices.
    Returns a (N, D) float32 numpy array.
    """
    chunks = []
    for shard_idx in sorted(indices_by_shard.keys()):
        shard_path = os.path.join(grads_dir, "shards", f"shard_{shard_idx + 1}.npy")
        shard_data = np.load(shard_path).astype(np.float32)
        rows = indices_by_shard[shard_idx]
        chunks.append(shard_data[rows])
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(grads_dir, seed=42, var_threshold_factor=0):
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Read dataset info
    records = load_dataset_info(grads_dir)
    print(f"Total gradient records: {len(records)}")

    # 2. Collect unique first images and split into two equal halves
    first_images = sorted(set(r[2] for r in records))
    print(f"Unique first-images: {len(first_images)}")

    random.shuffle(first_images)
    half = len(first_images) // 2
    subset_a = set(first_images[:half])
    subset_b = set(first_images[half:2 * half])   # equal size even if odd total
    print(f"Subset A images: {len(subset_a)}, Subset B images: {len(subset_b)}")

    # 3. Partition gradient indices by subset
    indices_a, indices_b = {}, {}
    for shard_idx, row_idx, img_a, _ in records:
        if img_a in subset_a:
            indices_a.setdefault(shard_idx, []).append(row_idx)
        elif img_a in subset_b:
            indices_b.setdefault(shard_idx, []).append(row_idx)

    n_a = sum(len(v) for v in indices_a.values())
    n_b = sum(len(v) for v in indices_b.values())
    print(f"Gradients in subset A: {n_a}, subset B: {n_b}")

    # 4. Load gradients
    print("Loading gradients for subset A...")
    grads_a = torch.tensor(load_grads_for_indices(grads_dir, indices_a), dtype=torch.float32).to(device)
    print("Loading gradients for subset B...")
    grads_b = torch.tensor(load_grads_for_indices(grads_dir, indices_b), dtype=torch.float32).to(device)

    # 5. Compute subspaces
    print(f"\nComputing subspace for subset A (N={len(grads_a)}, D={grads_a.shape[1]})...")
    basis_a, var_a = get_significant_subspace(grads_a, var_threshold_factor)
    dim_a = basis_a.shape[1] if basis_a is not None else 0
    print(f"  Significant directions: {dim_a}  (covers {var_a:.3f} of total variance)")

    print(f"Computing subspace for subset B (N={len(grads_b)}, D={grads_b.shape[1]})...")
    basis_b, var_b = get_significant_subspace(grads_b, var_threshold_factor)
    dim_b = basis_b.shape[1] if basis_b is not None else 0
    print(f"  Significant directions: {dim_b}  (covers {var_b:.3f} of total variance)")

    if basis_a is None or basis_b is None:
        print("One of the subspaces is empty — cannot compare.")
        return

    # 6. Subspace overlap metrics
    # "How much of A is reconstructed by B?" and vice-versa
    cov_a_from_b = subspace_overlap(basis_target=basis_a, basis_source=basis_b)
    cov_b_from_a = subspace_overlap(basis_target=basis_b, basis_source=basis_a)

    hallucination_a = 1.0 - subspace_overlap(basis_target=basis_b, basis_source=basis_a)
    hallucination_b = 1.0 - subspace_overlap(basis_target=basis_a, basis_source=basis_b)

    print("\n" + "=" * 55)
    print("SUBSPACE COMPARISON RESULTS")
    print("=" * 55)
    print(f"  Subspace A dim: {dim_a}  |  Subspace B dim: {dim_b}")
    print()
    print(f"  Subspace Reconstruction (how much of A is in B): {cov_a_from_b:.4f}")
    print(f"  Subspace Reconstruction (how much of B is in A): {cov_b_from_a:.4f}")
    print()
    print(f"  Subspace Hallucination  (fraction of A not in B): {hallucination_b:.4f}")
    print(f"  Subspace Hallucination  (fraction of B not in A): {hallucination_a:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare subspaces from two disjoint image subsets.")
    parser.add_argument(
        "--grads_dir",
        type=str,
        default="/data/pskiers/concept_dimention/grads/sdxl-person/grads/t50",
        help="Path to timestep grads dir containing dataset_info.json and shards/",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold_factor", type=float, default=2.5,
                        help="Significance factor: threshold = factor / dim")
    args = parser.parse_args()

    main(args.grads_dir, seed=args.seed, var_threshold_factor=args.threshold_factor)
