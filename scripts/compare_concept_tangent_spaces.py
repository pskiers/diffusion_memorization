"""
Compare subspaces built from two separate gradient datasets.

Usage:
  python compare_concept_tangent_spaces.py \
      --grads_a path/to/dataset_a/shards \
      --grads_b path/to/dataset_b/shards \
      [--threshold_factor 2.5]
"""

import argparse
import torch
from compare_subspaces import load_grads
from image_subset_subspace_exp import get_significant_subspace, subspace_overlap


def main(grads_dir_a, grads_dir_b, var_threshold_factor=2.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset A...")
    grads_a = load_grads(grads_dir_a, 10000).to(device)
    print(f"  {grads_a.shape[0]} vectors, dim {grads_a.shape[1]}")

    print("Loading dataset B...")
    grads_b = load_grads(grads_dir_b, 10000).to(device)
    print(f"  {grads_b.shape[0]} vectors, dim {grads_b.shape[1]}")

    print(f"\nComputing subspace A...")
    basis_a, var_a = get_significant_subspace(grads_a, var_threshold_factor)
    dim_a = basis_a.shape[1] if basis_a is not None else 0
    print(f"  Significant directions: {dim_a}  (covers {var_a:.3f} of total variance)")

    print(f"Computing subspace B...")
    basis_b, var_b = get_significant_subspace(grads_b, var_threshold_factor)
    dim_b = basis_b.shape[1] if basis_b is not None else 0
    print(f"  Significant directions: {dim_b}  (covers {var_b:.3f} of total variance)")

    if basis_a is None or basis_b is None:
        print("One of the subspaces is empty — cannot compare.")
        return

    cov_a_from_b = subspace_overlap(basis_target=basis_a, basis_source=basis_b)
    cov_b_from_a = subspace_overlap(basis_target=basis_b, basis_source=basis_a)

    print("\n" + "=" * 55)
    print("SUBSPACE COMPARISON RESULTS")
    print("=" * 55)
    print(f"  Subspace A dim: {dim_a}  |  Subspace B dim: {dim_b}")
    print()
    print(f"  Reconstruction (how much of A is in B): {cov_a_from_b:.4f}")
    print(f"  Reconstruction (how much of B is in A): {cov_b_from_a:.4f}")
    print()
    print(f"  Hallucination  (fraction of A not in B): {1 - cov_a_from_b:.4f}")
    print(f"  Hallucination  (fraction of B not in A): {1 - cov_b_from_a:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grads_a", type=str, required=True)
    parser.add_argument("--grads_b", type=str, required=True)
    parser.add_argument(
        "--threshold_factor", type=float, default=2.5, help="Significance factor: threshold = factor / dim"
    )
    args = parser.parse_args()

    main(args.grads_a, args.grads_b, args.threshold_factor)
