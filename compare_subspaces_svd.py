import os
import numpy as np

from scipy.linalg import orth
from numpy.linalg import svd



def compare_subspaces_svd():
    dir1 = "/net/scratch/hscra/plgrid/plgpawel269/LID-project/diffusion_memorization/outputs/sdxl-frog/grads/t13"
    dir2 = "/net/scratch/hscra/plgrid/plgpawel269/LID-project/diffusion_memorization/outputs/sdxl-turbo-frog/grads/t1"
    threshold = 1e-3

    def load_grads(grads_dir):
        grad_files = [f for f in os.listdir(grads_dir) if f.endswith(".npy")]
        grads = []
        filenames = []
        for fname in grad_files:
            grad = np.load(os.path.join(grads_dir, fname))
            grads.append(grad.flatten())
            filenames.append(fname)
        grad_matrix = np.stack(grads, axis=0).astype(np.float32)
        return grad_matrix, filenames

    # Load grads
    grads1, filenames1 = load_grads(dir1)
    grads2, filenames2 = load_grads(dir2)

    # SVD
    u1, s1, vh1 = np.linalg.svd(grads1, full_matrices=False)
    u2, s2, vh2 = np.linalg.svd(grads2, full_matrices=False)

    # Significant directions
    sig_idx1 = np.where(s1 > threshold)[0]
    sig_idx2 = np.where(s2 > threshold)[0]
    subspace1 = vh1[sig_idx1]
    subspace2 = vh2[sig_idx2]

    # Orthonormalize
    subspace1_orth = orth(subspace1.T)
    subspace2_orth = orth(subspace2.T)

    # Compute intersection

    M = np.dot(subspace1_orth.T, subspace2_orth)
    _, s, _ = svd(M)
    common_size = np.sum(s > 1e-2)

    print(f"Common subspace size: {common_size}")
    print(f"Subspace1 size: {subspace1_orth.shape[1]}")
    print(f"Subspace2 size: {subspace2_orth.shape[1]}")
    print(f"Exclusive to subspace1: {subspace1_orth.shape[1] - common_size}")
    print(f"Exclusive to subspace2: {subspace2_orth.shape[1] - common_size}")

    # Find exclusive directions
    # Project subspace1_orth onto subspace2_orth and get residuals
    proj1_on_2 = subspace2_orth @ (subspace2_orth.T @ subspace1_orth)
    exclusive1 = subspace1_orth - proj1_on_2
    exclusive1 = orth(exclusive1)
    np.save("exclusive_to_subspace1.npy", exclusive1)

    proj2_on_1 = subspace1_orth @ (subspace1_orth.T @ subspace2_orth)
    exclusive2 = subspace2_orth - proj2_on_1
    exclusive2 = orth(exclusive2)
    np.save("exclusive_to_subspace2.npy", exclusive2)

    # Project each original vector in grads1 onto the exclusive subspace
    projections = grads1 @ exclusive1
    projection_magnitudes = np.linalg.norm(projections, axis=1)

    n = 5
    top_n_idx = np.argpartition(-projection_magnitudes, n-1)[:n]
    top_n_idx = top_n_idx[np.argsort(-projection_magnitudes[top_n_idx])]
    # print filenames and magnitudes for the top-n
    for idx in top_n_idx:
        print(f"{filenames1[idx]}: {projection_magnitudes[idx]}")
    most_exclusive_idx = np.argmax(projection_magnitudes)
    most_exclusive_vector = grads1[most_exclusive_idx]
    np.save("most_exclusive_vector.npy", most_exclusive_vector)

    print(f"{filenames1[most_exclusive_idx]}, {most_exclusive_vector}")


if __name__ == "__main__":
    compare_subspaces_svd()
