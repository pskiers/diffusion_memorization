import numpy as np
import os
import argparse
import glob


def load_gradients(directory, max_samples=None):
    """
    Loads .npy files from the specified directory up to a maximum number of samples.
    Expects files to contain arrays of shape (bs, d).
    """
    print(f"Loading gradients from {directory}...")
    file_paths = glob.glob(os.path.join(directory, "*.npy"))

    if not file_paths:
        raise FileNotFoundError(f"No .npy files found in {directory}")

    grad_list = []
    total_loaded = 0

    for path in file_paths:
        if max_samples is not None and total_loaded >= max_samples:
            break

        try:
            batch = np.load(path)
            if batch.ndim == 1:
                batch = batch.reshape(1, -1)

            grad_list.append(batch)
            total_loaded += batch.shape[0]

        except Exception as e:
            print(f"Error loading {path}: {e}")

    if not grad_list:
        raise ValueError("Could not load any valid gradient files.")

    all_grads = np.concatenate(grad_list, axis=0)

    if max_samples is not None and all_grads.shape[0] > max_samples:
        print(f"Limiting samples to {max_samples} (loaded {all_grads.shape[0]})...")
        all_grads = all_grads[:max_samples]

    print(f"Successfully loaded {all_grads.shape[0]} gradients of dimension {all_grads.shape[1]}.")
    return all_grads


def analyze_spectrum(gradients, threshold=0.95):
    """
    Performs SVD and calculates dimensionality using both:
    1. Explained Variance Threshold (Effective Dimension)
    2. Spectral Gap (Stanczuk et al. method)
    """
    print("Centering gradients...")
    gradients_centered = gradients - np.mean(gradients, axis=0)

    print("Performing SVD...")
    # Singular values 's' are sorted in descending order
    _, s, _ = np.linalg.svd(gradients_centered, full_matrices=False)

    # --- Method 1: Explained Variance ---
    eigenvalues = s**2
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    if threshold > 1.0:
        threshold = 1.0

    if cumulative_variance[-1] < threshold:
        lid_variance = len(cumulative_variance)
    else:
        lid_variance = np.searchsorted(cumulative_variance, threshold) + 1

    # --- Method 2: Spectral Gap (Stanczuk et al.) ---
    # We find i that maximizes (s_i - s_{i+1})
    # Note: s is 0-indexed, so s[i] corresponds to the (i+1)-th singular value
    # We look for the gap between s[i] and s[i+1]. The dimension is i+1.
    if len(s) > 1:
        # Calculate gaps between consecutive singular values
        gaps = s[:-1] - s[1:]
        # Find index of largest gap
        max_gap_idx = np.argmax(gaps)
        # The dimension is the number of components *before* the drop
        lid_gap = max_gap_idx + 1
    else:
        lid_gap = 1

    return {
        "lid_variance": lid_variance,
        "lid_gap": lid_gap,
        "singular_values": s,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate Concept Complexity from Gradients")
    parser.add_argument("--dir", type=str, required=True, help="Path to directory containing .npy gradient files")
    parser.add_argument("--threshold", type=float, default=0.95, help="Explained variance threshold (default: 0.95)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of gradient samples to load")

    args = parser.parse_args()

    try:
        # 1. Load Data
        gradients = load_gradients(args.dir, args.max_samples)

        # 2. Analyze
        results = analyze_spectrum(gradients, args.threshold)

        # 3. Report Results
        print("\n" + "=" * 40)
        print(f"ANALYSIS RESULTS: {os.path.basename(args.dir)}")
        print("=" * 40)
        print(f"Samples: {gradients.shape[0]} | Dimension: {gradients.shape[1]}")
        print("-" * 40)
        print(f"Est. Complexity (Variance {args.threshold*100}%): {results['lid_variance']}")
        print(f"Est. Dimension (Spectral Gap)      : {results['lid_gap']}")
        print("-" * 40)

        print("\nTop 5 Components:")
        print(f"{'Comp':<5} | {'Singular Val':<12} | {'Expl. Var.':<12} | {'Cumulative'}")
        print("-" * 50)
        for i in range(min(5, len(results["singular_values"]))):
            sv = results["singular_values"][i]
            var = results["explained_variance_ratio"][i]
            cum = results["cumulative_variance"][i]
            print(f"{i+1:<5} | {sv:<12.4f} | {var:<12.4f} | {cum:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
