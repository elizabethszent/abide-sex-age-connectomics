import numpy as np
from pathlib import Path
import bct

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
GROUP_MATS = [
    BASE / r"results\group_connectomes\F_ASD_Zmean.npy",
    BASE / r"results\group_connectomes\F_CTL_Zmean.npy",
    BASE / r"results\group_connectomes\M_ASD_Zmean.npy",
    BASE / r"results\group_connectomes\M_CTL_Zmean.npy",
]

OUT_NPY = BASE / r"results\group_connectomes\CC200_modules_signed_asym1000.npy"
OUT_TXT = BASE / r"results\group_connectomes\CC200_modules_signed_asym1000.txt"

TARGET_MIN, TARGET_MAX = 7, 20

N_TRIALS = 1000
SEED_BASE = 42

# Louvain resolution parameter (gamma). Higher -> more/smaller modules.
GAMMA = 1.1

# Scale negative weights by a factor < 1 so they influence modularity less.
# 1.0 = treat negatives equally strong as given
# 0.5 = negatives count half as much as their magnitude
NEG_SCALE = 0.5

# Signed modularity option (asymmetric)
SIGNED_MODE = "negative_asym"


def load_and_average_group_mats(paths):
    mats = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing matrix: {p}")
        mats.append(np.load(p))
    A = np.mean(mats, axis=0)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Grand mean matrix not square: {A.shape}")
    return A


def preprocess_signed_weights(A: np.ndarray, neg_scale: float) -> np.ndarray:
    """
    Keep both positive and negative weights (signed), zero diagonal,
    optionally down-weight negative edges, and symmetrize.
    """
    W = A.astype(float).copy()
    np.fill_diagonal(W, 0.0)

    # down-weight negatives so positives matter more
    if neg_scale < 1.0:
        neg_mask = W < 0
        W[neg_mask] *= float(neg_scale)

    # symmetrize (protect against tiny asymmetry)
    W = 0.5 * (W + W.T)

    # sanity checks
    if not np.isfinite(W).all():
        bad = np.argwhere(~np.isfinite(W))
        raise ValueError(f"W contains NaN/Inf at e.g. {bad[:5].tolist()}")

    return W


def relabel_ci_to_1based(ci: np.ndarray) -> np.ndarray:
    """
    Map community labels to contiguous 1..K.
    """
    ci = np.asarray(ci).astype(int)
    uniq = sorted(set(ci.tolist()))
    remap = {lab: i + 1 for i, lab in enumerate(uniq)}
    return np.array([remap[x] for x in ci], dtype=int)


def louvain_1000_restarts_signed(
    W: np.ndarray,
    gamma: float,
    n_trials: int = 1000,
    seed_base: int = 42,
    target_min: int = 7,
    target_max: int = 20,
    signed_mode: str = "negative_asym",
):
    """
    Option A: run Louvain n_trials times (random restarts) at ONE fixed gamma,
    using signed asymmetric modularity (BCT: bct.community_louvain).

    Selection:
      - Prefer solutions with k in [target_min, target_max]
      - Among those: maximize Q, tie-breaker fewer modules (smaller k)
      - If none in range: closest k to range, then maximize Q
    """
    cands = []
    failures = 0

    for t in range(n_trials):
        seed = seed_base + t
        try:
            # bctpy returns (ci, Q)
            ci, Q = bct.community_louvain(W, gamma=float(gamma), B=signed_mode, seed=seed)
            ci = np.asarray(ci).astype(int)
            k = len(set(ci.tolist()))
            cands.append((k, float(Q), ci, seed))
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"[warn] Louvain failed at seed={seed}: {type(e).__name__}: {e}")
            continue

    if not cands:
        raise RuntimeError("All Louvain trials failed — check W matrix and bctpy install/version.")

    print(f"Completed {len(cands)} / {n_trials} trials (failures={failures})")

    def in_range(k: int) -> bool:
        return target_min <= k <= target_max

    elig = [c for c in cands if in_range(c[0])]

    if elig:
        # pick: highest Q, then fewer modules
        k, Q, ci, seed = sorted(elig, key=lambda t: (-t[1], t[0]))[0]
        return k, Q, ci, seed

    # fallback: closest-to-range then highest Q
    def dist_to_range(k: int) -> int:
        if k < target_min:
            return target_min - k
        if k > target_max:
            return k - target_max
        return 0

    k, Q, ci, seed = sorted(cands, key=lambda t: (dist_to_range(t[0]), -t[1]))[0]
    return k, Q, ci, seed


def main():
    A = load_and_average_group_mats(GROUP_MATS)
    n = A.shape[0]
    print(f"Grand mean matrix shape: {A.shape}")

    W = preprocess_signed_weights(A, neg_scale=NEG_SCALE)
    print(
        f"W stats: min={W.min():.4f}, max={W.max():.4f}, "
        f"neg_frac={(W < 0).mean():.3f}, pos_frac={(W > 0).mean():.3f}"
    )

    k, Q, ci, seed = louvain_1000_restarts_signed(
        W,
        gamma=GAMMA,
        n_trials=N_TRIALS,
        seed_base=SEED_BASE,
        target_min=TARGET_MIN,
        target_max=TARGET_MAX,
        signed_mode=SIGNED_MODE,
    )

    modules = relabel_ci_to_1based(ci)

    print(
        f"Louvain signed-asym (BCT): runs={N_TRIALS}, gamma={GAMMA:.3f}, "
        f"neg_scale={NEG_SCALE:.3f}, modules={k}, Q={Q:.4f}, best_seed={seed}"
    )

    # Save outputs
    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, modules)
    print(f"Saved modules to {OUT_NPY}")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("ROI_index\tModule\n")
        for i, m in enumerate(modules, start=1):
            f.write(f"{i}\t{m}\n")
    print(f"Saved human-readable modules to {OUT_TXT}")

    # quick module size summary
    unique, counts = np.unique(modules, return_counts=True)
    print("\nModule sizes:")
    for u, c in zip(unique, counts):
        print(f"module {u}: {c}")


if __name__ == "__main__":
    main()
