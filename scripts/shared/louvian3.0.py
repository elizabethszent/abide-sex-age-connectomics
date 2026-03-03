import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import bct

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# =========================
# INPUT: ALL-SUBJECTS GRAND MEAN (new)
# =========================
#ALL_MEAN = BASE / r"results\group_connectomes\ALL_SUBJECTS_Zmean.npy"
ALL_MEAN = BASE / r"results\group_connectomes\ABIDE1_CC200\OVERALL_ageSexMatched_fd-0.2_mean_r.npy"

#ALL_IDS  = BASE / r"results\group_connectomes\ALL_SUBJECTS_file_ids.csv"
ALL_IDS  = BASE / r"results\group_connectomes\ABIDE1_CC200\OVERALL_ageSexMatched_fd-0.2_selected_subjects.csv"

# =========================
# OUTPUTS (renamed so we don't overwrite your prior run)
# =========================
OUT_NPY = BASE / r"results\group_connectomes\CC200_modules_ALLSUBJ_signed_asym1000.npy"
OUT_TXT = BASE / r"results\group_connectomes\CC200_modules_ALLSUBJ_signed_asym1000.txt"

# =========================
# LOUVAIN SETTINGS
# =========================
N_TRIALS = 1000
SEED_BASE = 42

# Louvain resolution (gamma) — tune this to hit your desired K
GAMMA = 1.64

# Make positives count more than negatives (0.5 = negatives half strength)
NEG_SCALE = 0.5

# Signed modularity option (asymmetric)
SIGNED_MODE = "negative_asym"

# Target and balance constraints (to avoid tiny "unrealistic" modules)
K_TARGET = 7
MIN_SIZE = 10

EXPECTED_N = 200


def load_all_subjects_grand_mean() -> np.ndarray:
    """
    Load the grand mean matrix across ALL subjects (built from individual matrices),
    instead of averaging 4 group means.
    """
    if not ALL_MEAN.exists():
        raise FileNotFoundError(f"Missing ALL_SUBJECTS_Zmean.npy: {ALL_MEAN}")

    A = np.load(ALL_MEAN)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"ALL_SUBJECTS_Zmean not square: {A.shape}")

    if A.shape != (EXPECTED_N, EXPECTED_N):
        raise ValueError(f"Expected {EXPECTED_N}x{EXPECTED_N}, got {A.shape}")

    if not np.isfinite(A).all():
        bad = np.argwhere(~np.isfinite(A))
        raise ValueError(f"ALL_SUBJECTS_Zmean has NaN/Inf at e.g. {bad[:5].tolist()}")

    # optional: print N included (from IDs file)
    if ALL_IDS.exists():
        df_ids = pd.read_csv(ALL_IDS)
        if "FILE_ID" in df_ids.columns:
            n_ids = int(df_ids["FILE_ID"].astype(str).str.strip().nunique())
            print(f"Using ALL_SUBJECTS_Zmean built from N={n_ids} subjects (from ALL_SUBJECTS_file_ids.csv)")
        else:
            print("Using ALL_SUBJECTS_Zmean (ALL_SUBJECTS_file_ids.csv exists but has no FILE_ID column)")
    else:
        print("Using ALL_SUBJECTS_Zmean (ALL_SUBJECTS_file_ids.csv not found to report N)")

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

    # symmetrize
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
    signed_mode: str = "negative_asym",
):
    """
    Run Louvain n_trials times (random restarts) at ONE fixed gamma
    using signed asymmetric modularity (BCT: bct.community_louvain).

    Uses np.random.seed(seed) for broader bctpy compatibility (instead of seed= kwarg).

    Selection:
      1) Prefer k == K_TARGET and min community size >= MIN_SIZE, maximize Q
      2) Else prefer min community size >= MIN_SIZE, closest k to K_TARGET, then max Q
      3) Else closest k to K_TARGET, then max Q
    """
    cands = []
    failures = 0
    first_errors = []

    for t in range(n_trials):
        seed = seed_base + t
        try:
            np.random.seed(seed)  # compatible seeding across bctpy versions
            ci, Q = bct.community_louvain(W, gamma=float(gamma), B=signed_mode)

            ci = np.asarray(ci).astype(int)
            sizes = Counter(ci.tolist())
            k = len(sizes)
            min_size = min(sizes.values())
            cands.append((k, min_size, float(Q), ci, seed))
        except Exception as e:
            failures += 1
            if len(first_errors) < 5:
                first_errors.append((seed, type(e).__name__, str(e)))
            continue

    if not cands:
        print("First errors:")
        for seed, etype, msg in first_errors:
            print(f"  seed={seed}: {etype}: {msg}")
        raise RuntimeError("All Louvain trials failed.")

    print(f"Completed {len(cands)} / {n_trials} trials (failures={failures})")

    # 1) Hit target K and avoid tiny modules
    good = [c for c in cands if (c[0] == K_TARGET and c[1] >= MIN_SIZE)]
    if good:
        k, min_size, Q, ci, seed = sorted(good, key=lambda t: (-t[2], -t[1]))[0]
        return k, Q, ci, seed

    # 2) At least avoid tiny modules, choose closest K
    good2 = [c for c in cands if c[1] >= MIN_SIZE]
    if good2:
        k, min_size, Q, ci, seed = sorted(
            good2, key=lambda t: (abs(t[0] - K_TARGET), -t[2], -t[1])
        )[0]
        return k, Q, ci, seed

    # 3) Fallback: closest K then max Q
    k, min_size, Q, ci, seed = sorted(cands, key=lambda t: (abs(t[0] - K_TARGET), -t[2]))[0]
    return k, Q, ci, seed


def main():
    A = load_all_subjects_grand_mean()
    print(f"ALL-subjects grand mean matrix shape: {A.shape}")

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
        signed_mode=SIGNED_MODE,
    )

    modules = relabel_ci_to_1based(ci)

    print(
        f"Louvain signed-asym (BCT): runs={N_TRIALS}, gamma={GAMMA:.3f}, "
        f"neg_scale={NEG_SCALE:.3f}, modules={k}, Q={Q:.4f}, best_seed={seed}, "
        f"K_TARGET={K_TARGET}, MIN_SIZE={MIN_SIZE}"
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

    # module size summary
    uniq, cnts = np.unique(modules, return_counts=True)
    print("\nModule sizes:")
    for u, c in zip(uniq, cnts):
        print(f"module {u}: {c}")


if __name__ == "__main__":
    main()
