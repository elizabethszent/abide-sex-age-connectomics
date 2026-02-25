import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

CONN_DIR = ROOT / "data/connectomes/cpac/nofilt_noglobal/cc200_z"

# Use whichever module file you actually want:
MODULE_FILE = ROOT / "results/group_connectomes/CC200_modules_ALLSUBJ_signed_asym1000.txt"
# MODULE_FILE = ROOT / "results/group_connectomes/CC200_modules_signed_asym1000.txt"

FEMALE_META = ROOT / "data/female/female_metadata_included.csv"
MALE_META   = ROOT / "data/male/male_metadata_included.csv"

OUT_DIR = ROOT / "results/hubs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Keep your threshold
DENSITY = 0.0797

# -------- Option A age bins --------
BINS   = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
RIGHT  = False  # [0,10), [10,13), [13,18), [18,200)

AGE_GROUPS = LABELS

# ----------------------------
# Load modules: ROI_index  Module
# ----------------------------
mods = pd.read_csv(MODULE_FILE, sep=r"\s+")
mods = mods.sort_values("ROI_index")

roi2mod = mods["Module"].to_numpy()  # length 200, values 1..K
N_ROI   = len(roi2mod)
N_MOD   = int(roi2mod.max())

print(f"Loaded module file with {N_ROI} ROIs, {N_MOD} modules")

# ----------------------------
# Helpers
# ----------------------------
def threshold_top_density(mat: np.ndarray, density: float) -> np.ndarray:
    """
    Keep top 'density' fraction of absolute edges (upper triangle),
    preserve original sign/weight, symmetrize, zero diagonal.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix not square: {mat.shape}")
    n = mat.shape[0]

    mat = mat.copy()
    np.fill_diagonal(mat, 0.0)

    iu, ju = np.triu_indices(n, k=1)
    vals = np.abs(mat[iu, ju])

    m = len(vals)
    k = int(np.floor(density * m))
    if k < 1:
        out = np.zeros_like(mat)
        np.fill_diagonal(out, 0.0)
        return out

    thresh = np.partition(vals, -k)[-k]
    mask   = np.abs(mat) >= thresh

    out = np.where(mask, mat, 0.0)
    out = np.triu(out, 1)
    out = out + out.T
    np.fill_diagonal(out, 0.0)
    return out


def compute_pc_z_strength_posneg(W: np.ndarray, roi2mod: np.ndarray):
    """
    Signed network version:
      strength_pos: sum of positive weights per node
      strength_neg: sum of |negative weights| per node (positive-valued)
      PC and Z computed using ABS weights (stable for signed FC).
    """
    if W.shape != (N_ROI, N_ROI):
        raise ValueError(f"Expected {(N_ROI, N_ROI)}, got {W.shape}")

    N = W.shape[0]
    M = int(roi2mod.max())

    # Split weights
    Wp   = np.where(W > 0, W, 0.0)
    Wn   = np.where(W < 0, -W, 0.0)   # magnitude of negatives (positive)
    Wabs = Wp + Wn                   # == np.abs(W)

    # Strengths
    strength_pos = Wp.sum(axis=1)
    strength_neg = Wn.sum(axis=1)
    strength_abs = Wabs.sum(axis=1)

    # Strength to each module (ABS)
    k_by_mod = np.zeros((N, M), dtype=float)
    for m in range(1, M + 1):
        idx = np.where(roi2mod == m)[0]
        if idx.size == 0:
            continue
        k_by_mod[:, m - 1] = Wabs[:, idx].sum(axis=1)

    # Participation coefficient (ABS)
    pc = np.zeros(N, dtype=float)
    for i in range(N):
        ki = strength_abs[i]
        if ki <= 0:
            pc[i] = 0.0
        else:
            frac = k_by_mod[i, :] / ki
            pc[i] = 1.0 - np.sum(frac ** 2)

    # Within-module degree z (ABS)
    z = np.zeros(N, dtype=float)
    for m in range(1, M + 1):
        idx = np.where(roi2mod == m)[0]
        if idx.size == 0:
            continue

        k_within = k_by_mod[idx, m - 1]
        mu = k_within.mean()
        sd = k_within.std(ddof=1)
        z[idx] = (k_within - mu) / sd if sd > 0 else 0.0

    return pc, z, strength_pos, strength_neg


def add_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AGE_GROUP"] = pd.cut(
        df["AGE_AT_SCAN"],
        bins=BINS,
        labels=LABELS,
        right=RIGHT,
        include_lowest=True
    )
    return df


def process_sex(sex_label: str, meta_path: Path):
    print(f"\n{sex_label.upper()}")

    meta = pd.read_csv(meta_path)
    meta.columns = meta.columns.str.strip()
    meta["FILE_ID"] = meta["FILE_ID"].astype(str).str.strip()
    meta = add_age_group_column(meta)

    required = {"FILE_ID", "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd", "AGE_GROUP"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"{meta_path} missing columns: {missing}")

    for age_group in AGE_GROUPS:
        sub = meta[meta["AGE_GROUP"] == age_group].copy()
        if sub.empty:
            print(f"  [{sex_label} {age_group}] No subjects, skipping.")
            continue

        print(f"  [{sex_label} {age_group}] N={len(sub)}")
        print(sub["DX_GROUP"].value_counts().rename({1: "ASD", 2: "CTL"}))

        rows = []
        n_bad = 0

        for _, row in sub.iterrows():
            fid = row["FILE_ID"]
            conn_fp = CONN_DIR / f"{fid}.npy"

            if not conn_fp.exists():
                n_bad += 1
                continue

            mat = np.load(conn_fp)
            if mat.shape != (N_ROI, N_ROI):
                n_bad += 1
                continue
            if not np.isfinite(mat).all():
                n_bad += 1
                continue

            W = threshold_top_density(mat, density=DENSITY)

            # Ensure symmetry + clean diagonal
            W = 0.5 * (W + W.T)
            np.fill_diagonal(W, 0.0)

            pc, z, strength_pos, strength_neg = compute_pc_z_strength_posneg(W, roi2mod)

            for node_idx in range(N_ROI):
                rows.append({
                    "FILE_ID": fid,
                    "sex": sex_label,
                    "AGE_GROUP": age_group,
                    "DX_GROUP": int(row["DX_GROUP"]),
                    "AGE_AT_SCAN": float(row["AGE_AT_SCAN"]),
                    "func_mean_fd": float(row["func_mean_fd"]),
                    "node": node_idx + 1,             # 1..200
                    "module": int(roi2mod[node_idx]), # 1..K
                    "PC": float(pc[node_idx]),
                    "z": float(z[node_idx]),
                    "strength_pos": float(strength_pos[node_idx]),
                    "strength_neg": float(strength_neg[node_idx]),
                })

        if not rows:
            print(f"  [{sex_label} {age_group}] No usable subjects after QC.")
            continue

        out_df = pd.DataFrame(rows)
        out_path = OUT_DIR / f"{sex_label}_{age_group}_pc_z_strengthPosNeg_revised.csv"
        out_df.to_csv(out_path, index=False)

        print(
            f"  [{sex_label} {age_group}] Saved {len(out_df)} node-rows -> {out_path} "
            f"(skipped {n_bad})"
        )


# ----------------------------
# Run
# ----------------------------
process_sex("female", FEMALE_META)
process_sex("male", MALE_META)

print("\nDone.")
