"""
Subspace-stability sweep over gradient-pool size (ELROND Appendix B / Figure 9).

Reads a timestep-stratified gradient pool written by token_subspace.py:

    <grads_root>/t{T}/shards/shard_*.npy      each (shard_size, 2048)

and reproduces the two curves of Figure 9:

    "Subspace Reconstruction"  -- fraction of the reference subspace spanned by
                                  the subspace recovered from a subset of size N
    "Subspace Hallucinations"  -- fraction of the subset's own directions that are
                                  absent from the reference subspace

NOTE ON "EXACTLY". Appendix B defines neither metric formally: it says only
"fraction of subspace spanned by the smaller subset compared to the reference
space" and "directions present in the smaller subset but absent in the
reference". This file implements the standard reading of both -- squared
Frobenius overlap of the two orthonormal bases, normalised by the reference
dimension and by the subset dimension respectively. That is a defensible
interpretation, not a verified match to the authors' code. Report the formulas
alongside the curves.

Usage
-----
    python subspace_sweep.py \
        --grads-root /net/scratch/.../outputs/sdxl-monster/grads \
        --out-dir    ./sweep_monster \
        --normalize  none \
        --tau-mode   fixed

    # same pool, unit-norm gradients, N-dependent threshold
    python subspace_sweep.py ... --normalize l2 --tau-mode n_dependent
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_pool(grads_root, embed_slice=None, verbose=True):
    """Load every shard under <grads_root>/t*/shards/*.npy.

    Returns
    -------
    G : torch.Tensor, (M, D) float32
    t_labels : np.ndarray, (M,) int   timestep index each row was collected at
    """
    pattern = os.path.join(grads_root, "t*", "shards", "*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"no shards matched {pattern}\n"
            "check the path: token_subspace.py writes to grads/t{T}/shards/, and "
            "os.listdir on the t{T} directory alone will not see them."
        )

    blocks, labels = [], []
    for f in files:
        m = re.search(r"[/\\]t(\d+)[/\\]", f)
        if m is None:
            raise ValueError(f"cannot parse timestep from path: {f}")
        t = int(m.group(1))
        a = np.load(f)
        if a.ndim != 2:
            raise ValueError(f"{f}: expected 2-D array, got shape {a.shape}")
        blocks.append(a.astype(np.float32, copy=False))
        labels.append(np.full(a.shape[0], t, dtype=np.int64))

    G = np.concatenate(blocks, axis=0)
    t_labels = np.concatenate(labels, axis=0)

    # multi_token_subspace's load_grads slices [:, 2048:] because its rows are a
    # concatenation of two embeddings. Single-token shards are already (N, 2048),
    # so slicing there returns an empty array. Only slice if asked.
    if embed_slice is not None:
        lo, hi = embed_slice
        G = G[:, lo:hi]

    if verbose:
        uniq, cnt = np.unique(t_labels, return_counts=True)
        print(f"loaded {G.shape[0]:,} gradients of dim {G.shape[1]} "
              f"from {len(files)} shards across {len(uniq)} timesteps")
        print(f"  per-timestep counts: min={cnt.min()}, max={cnt.max()}, "
              f"mean={cnt.mean():.1f}")
        if cnt.min() != cnt.max():
            print("  WARNING: strata are unbalanced -- proportional sampling will "
                  "inherit the imbalance at every N.")

    return torch.from_numpy(G), t_labels


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #

def pca_subspace(X, tau):
    """Mean-centred PCA, returning components with explained-variance ratio > tau.

    Implemented with SVD rather than an eigendecomposition of the covariance:
    it is numerically better conditioned and handles N < D and N >= D through
    the same code path.

    (The N < D branch of SubspaceGetter.find_significant_directions indexes
    `sorted_eigenvalues` with `valid_indices`, which are positions in the
    unsorted eigenvalue array -- so eigenvalues and eigenvectors come back
    misaligned whenever any eigenvalue falls below its 1e-9 cutoff. That branch
    is exactly the one every N < 2048 point of this sweep would take, which is
    why this file does its own PCA. Worth flagging upstream.)

    Returns
    -------
    V : (k, D) orthonormal rows -- the retained directions
    evr : (k,) explained-variance ratio of each retained direction
    evr_all : (min(N,D),) full explained-variance spectrum
    """
    n, d = X.shape
    if n < 2:
        raise ValueError(f"need at least 2 vectors for PCA, got {n}")

    Xc = X - X.mean(dim=0, keepdim=True)
    # full_matrices=False -> min(n, d) singular values, i.e. every nonzero one
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

    eigvals = (S ** 2) / (n - 1)
    total = eigvals.sum()
    if total <= 0:
        return X.new_zeros((0, d)), X.new_zeros(0), eigvals
    evr_all = eigvals / total

    mask = evr_all > tau
    return Vh[mask], evr_all[mask], evr_all


def resolve_tau(mode, c, n, d):
    """tau = C / d_embed  (paper Sec. 5.1) or C / min(N, D)  (rebuttal)."""
    if mode == "fixed":
        return c / d
    if mode == "n_dependent":
        return c / min(n, d)
    raise ValueError(f"unknown tau-mode: {mode}")


# --------------------------------------------------------------------------- #
# Overlap metrics
# --------------------------------------------------------------------------- #

def subspace_overlap(V_a, V_b):
    """Squared Frobenius overlap ||V_a V_b^T||_F^2 of two orthonormal row bases.

    Equals sum of squared cosines of the principal angles between the two
    subspaces, and lies in [0, min(k_a, k_b)]. Symmetric in its arguments --
    the two Figure 9 panels differ only in which dimension it is divided by.
    """
    if V_a.shape[0] == 0 or V_b.shape[0] == 0:
        return 0.0
    return float((V_a @ V_b.T).pow(2).sum())


def spanned_fraction(V_ref, V_sub):
    """Panel 1: how much of the reference subspace the subset recovers."""
    k_ref = V_ref.shape[0]
    if k_ref == 0:
        return float("nan")
    return subspace_overlap(V_ref, V_sub) / k_ref


def spurious_fraction(V_ref, V_sub):
    """Panel 2: how much of the subset's subspace is absent from the reference."""
    k_sub = V_sub.shape[0]
    if k_sub == 0:
        return float("nan")
    return 1.0 - subspace_overlap(V_sub, V_ref) / k_sub


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def stratified_indices(t_labels, n_target, rng):
    """Draw n_target row indices, proportionally across timesteps.

    Keeps the timestep composition of every subset identical to the pool's, so
    that points on the curve differ in sample size only. Uniform sampling would
    let low-N points inherit an arbitrary timestep mix, confounding sample size
    with noise level.
    """
    strata = {}
    for t in np.unique(t_labels):
        strata[t] = np.flatnonzero(t_labels == t)

    total = len(t_labels)
    picked = []
    # floor the proportional quota, then distribute the remainder by largest
    # fractional part so the total lands exactly on n_target
    quotas, fracs = {}, []
    for t, idx in strata.items():
        exact = n_target * len(idx) / total
        q = int(np.floor(exact))
        q = min(q, len(idx))
        quotas[t] = q
        fracs.append((exact - np.floor(exact), t))

    short = n_target - sum(quotas.values())
    for _, t in sorted(fracs, reverse=True):
        if short <= 0:
            break
        if quotas[t] < len(strata[t]):
            quotas[t] += 1
            short -= 1

    for t, idx in strata.items():
        q = quotas[t]
        if q > 0:
            picked.append(rng.choice(idx, size=q, replace=False))

    out = np.concatenate(picked) if picked else np.array([], dtype=np.int64)
    rng.shuffle(out)
    return out


def default_grid(m):
    """Log-spaced N grid up to the pool size, denser at the low end."""
    grid = [250, 500, 1000, 2000, 4000, 8000, 12000, 16000,
            20000, 25000, 30000, 40000, 50000]
    return [n for n in grid if n <= m]


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def run_sweep(G, t_labels, args):
    device = torch.device(args.device)
    G = G.to(device)
    d = G.shape[1]

    if args.normalize == "l2":
        # Unit-norm every gradient before decomposing. Gradient magnitude rises
        # steeply with the noise level (Fig. 10 spans ~50x across t), so an
        # unnormalised pool stratified over 50 timesteps is dominated by the
        # high-noise strata and is close to a single-timestep decomposition with
        # extra rows. Normalising removes that weighting -- but it also changes
        # what PCA estimates (directions on a sphere, not the raw distribution),
        # so treat the two settings as distinct variants and report both.
        norms = G.norm(dim=1, keepdim=True)
        dead = (norms.squeeze(1) == 0).sum().item()
        if dead:
            print(f"  {dead} zero-norm gradients left unnormalised")
        G = G / norms.clamp_min(1e-12)

    # ---- reference subspace -------------------------------------------------
    ref_n = args.reference_n or G.shape[0]
    rng = np.random.default_rng(args.seed)

    if ref_n >= G.shape[0]:
        ref_idx = np.arange(G.shape[0])
    else:
        ref_idx = stratified_indices(t_labels, ref_n, rng)

    if args.holdout:
        # Appendix B compares subsets against a reference drawn from the same
        # rows, so the largest N is a self-comparison and is forced to
        # (1.0, 0.0). Splitting the pool makes every point an honest
        # out-of-sample estimate -- a deviation from the paper, but it removes
        # the degenerate right edge.
        perm = rng.permutation(G.shape[0])
        half = G.shape[0] // 2
        ref_idx, pool_idx = perm[:half], perm[half:]
        print(f"holdout: reference from {len(ref_idx):,} rows, "
              f"subsets from a disjoint {len(pool_idx):,}")
    else:
        pool_idx = np.arange(G.shape[0])

    tau_ref = resolve_tau(args.tau_mode, args.tau_c, len(ref_idx), d)
    V_ref, _, evr_ref = pca_subspace(G[ref_idx], tau_ref)
    k_ref = V_ref.shape[0]
    print(f"reference: N={len(ref_idx):,}  tau={tau_ref:.3e}  k_ref={k_ref}")
    if k_ref == 0:
        raise RuntimeError("reference subspace is empty -- tau is too large")

    pool_labels = t_labels[pool_idx]
    grid = args.grid or default_grid(len(pool_idx))

    results = []
    for n in grid:
        if n > len(pool_idx):
            continue
        row = {"n": n, "spanned": [], "spurious": [], "k": [], "tau": None}
        for rep in range(args.reps):
            sub_rng = np.random.default_rng(args.seed + 1000 * rep + n)
            local = stratified_indices(pool_labels, n, sub_rng)
            idx = pool_idx[local]

            tau = resolve_tau(args.tau_mode, args.tau_c, n, d)
            row["tau"] = tau
            V_n, _, _ = pca_subspace(G[idx], tau)

            row["spanned"].append(spanned_fraction(V_ref, V_n))
            row["spurious"].append(spurious_fraction(V_ref, V_n))
            row["k"].append(int(V_n.shape[0]))

        results.append(row)
        print(f"  N={n:>6,}  k={np.mean(row['k']):6.1f}  "
              f"spanned={np.mean(row['spanned']):.4f} "
              f"(sd {np.std(row['spanned']):.4f})  "
              f"spurious={np.mean(row['spurious']):.4f} "
              f"(sd {np.std(row['spurious']):.4f})")

    return {
        "config": {
            "grads_root": args.grads_root,
            "normalize": args.normalize,
            "tau_mode": args.tau_mode,
            "tau_c": args.tau_c,
            "tau_reference": tau_ref,
            "reference_n": len(ref_idx),
            "k_reference": k_ref,
            "holdout": args.holdout,
            "reps": args.reps,
            "seed": args.seed,
            "d_embed": d,
            "pool_total": int(G.shape[0]),
            "n_strata": int(len(np.unique(t_labels))),
        },
        "reference_spectrum": evr_ref.detach().cpu().numpy().tolist()[:512],
        "points": results,
    }


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def plot(payload, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = payload["points"]
    ns = [p["n"] for p in pts]
    sp_m = [float(np.mean(p["spanned"])) for p in pts]
    sp_s = [float(np.std(p["spanned"])) for p in pts]
    hl_m = [float(np.mean(p["spurious"])) for p in pts]
    hl_s = [float(np.std(p["spurious"])) for p in pts]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].errorbar(ns, sp_m, yerr=sp_s, marker="o", color="tab:blue",
                     capsize=3, lw=1.5)
    axes[0].axhline(1.0, ls=":", c="gray", lw=1)
    axes[0].set_title("Subspace Reconstruction")
    axes[0].set_xlabel("Number of Vectors ($N$)")
    axes[0].set_ylabel("Fraction of space spanned", color="tab:blue")
    axes[0].set_ylim(0, 1.05)

    axes[1].errorbar(ns, hl_m, yerr=hl_s, marker="s", color="tab:red",
                     capsize=3, lw=1.5)
    axes[1].axhline(0.0, ls=":", c="gray", lw=1)
    axes[1].set_title('Subspace "Hallucinations"')
    axes[1].set_xlabel("Number of Vectors ($N$)")
    axes[1].set_ylabel("Fraction of noise", color="tab:red")
    axes[1].set_ylim(-0.05, 1.05)

    cfg = payload["config"]
    fig.suptitle(
        f"normalize={cfg['normalize']}  tau={cfg['tau_mode']} (C={cfg['tau_c']})  "
        f"k_ref={cfg['k_reference']}  reps={cfg['reps']}",
        fontsize=9, y=1.02,
    )
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # component count vs N -- feeds the LID estimate directly, and the paper
    # never shows it
    fig, ax = plt.subplots(figsize=(5.5, 4))
    k_m = [float(np.mean(p["k"])) for p in pts]
    k_s = [float(np.std(p["k"])) for p in pts]
    ax.errorbar(ns, k_m, yerr=k_s, marker="o", color="tab:green", capsize=3)
    ax.axhline(cfg["k_reference"], ls="--", c="gray", lw=1,
               label=f"reference ({cfg['k_reference']})")
    ax.set_xlabel("Number of Vectors ($N$)")
    ax.set_ylabel("Retained components $k$")
    ax.set_title("Estimated LID vs pool size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.replace(".png", "_k.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grads-root", required=True,
                   help="directory containing t{T}/shards/*.npy")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--normalize", choices=["none", "l2"], default="none",
                   help="unit-norm each gradient before PCA (default: none)")
    p.add_argument("--tau-mode", choices=["fixed", "n_dependent"], default="fixed",
                   help="fixed: C/d_embed (Sec. 5.1); n_dependent: C/min(N,D) (rebuttal)")
    p.add_argument("--tau-c", type=float, default=2.5)
    p.add_argument("--grid", type=int, nargs="+", default=None,
                   help="N values to sweep (default: log-spaced up to pool size)")
    p.add_argument("--reps", type=int, default=3,
                   help="independent subsets per N")
    p.add_argument("--reference-n", type=int, default=None,
                   help="reference pool size (default: all gradients)")
    p.add_argument("--holdout", action="store_true",
                   help="build the reference from a disjoint half of the pool")
    p.add_argument("--embed-slice", type=int, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="column slice, for multi-token shards wider than 2048")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    G, t_labels = load_pool(args.grads_root, embed_slice=args.embed_slice)
    payload = run_sweep(G, t_labels, args)

    tag = f"{args.normalize}_{args.tau_mode}"
    json_path = os.path.join(args.out_dir, f"sweep_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    png_path = os.path.join(args.out_dir, f"sweep_{tag}.png")
    plot(payload, png_path)

    print(f"\nwrote {json_path}")
    print(f"wrote {png_path}")
    print(f"wrote {png_path.replace('.png', '_k.png')}")


if __name__ == "__main__":
    main()