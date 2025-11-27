import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")

CONN_DIR    = ROOT / "data/connectomes/cpac/nofilt_noglobal/cc200_z"
MODULE_FILE = ROOT / "results/group_connectomes/CC200_modules.txt"

FEMALE_META = ROOT / "data/female/female_metadata_included.csv"
MALE_META   = ROOT / "data/male/male_metadata_included.csv"

OUT_DIR = ROOT / "results/hubs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

DENSITY = 0.10 #top 10% absolute edges

#age groups we care about
AGE_GROUPS = ["child", "teen", "young_adult"]


# CC200_modules.txt format: ROI_index  Module
mods = pd.read_csv(MODULE_FILE, sep=r"\s+")
mods = mods.sort_values("ROI_index")

roi2mod = mods["Module"].to_numpy() # length 200, values 1..7
N_ROI   = len(roi2mod)
N_MOD   = int(roi2mod.max())

print(f"Loaded module file with {N_ROI} ROIs, {N_MOD} modules")

#for convenience: list of indices per module
mod_idx = {m: np.where(roi2mod == m)[0] for m in range(1, N_MOD + 1)}


def threshold_top_density(mat: np.ndarray, density: float = 0.10) -> np.ndarray:
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
        return np.zeros_like(mat)

    thresh = np.partition(vals, -k)[-k]
    mask   = np.abs(mat) >= thresh

    out = np.where(mask, mat, 0.0)
    out = np.triu(out, 1)
    out = out + out.T
    np.fill_diagonal(out, 0.0)
    return out


def compute_pc_z_weighted(W: np.ndarray, roi2mod: np.ndarray):
    if W.shape != (N_ROI, N_ROI):
        raise ValueError(f"Expected {(N_ROI, N_ROI)}, got {W.shape}")

    N = W.shape[0]
    M = int(roi2mod.max())

    #node strength 
    k = W.sum(axis=1)  #shape (N,)

    #strength to each module
    k_by_mod = np.zeros((N, M), dtype=float)
    for m in range(1, M + 1):
        idx = np.where(roi2mod == m)[0]
        if idx.size == 0:
            continue
        k_by_mod[:, m - 1] = W[:, idx].sum(axis=1)

    #participation coefficient
    pc = np.zeros(N, dtype=float)
    for i in range(N):
        if k[i] <= 0:
            pc[i] = 0.0
        else:
            frac = k_by_mod[i, :] / k[i]
            pc[i] = 1.0 - np.sum(frac ** 2)

    #within-module degree z
    z = np.zeros(N, dtype=float)
    for m in range(1, M + 1):
        idx = np.where(roi2mod == m)[0]
        if idx.size == 0:
            continue

        k_within = k_by_mod[idx, m - 1]
        mu = k_within.mean()
        sd = k_within.std(ddof=1)

        if sd > 0:
            z[idx] = (k_within - mu) / sd
        else:
            z[idx] = 0.0

    return pc, z


def add_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [0, 13, 18, 30, 45, 120]
    labels = ["child", "teen", "young_adult", "adult", "older"]
    df = df.copy()
    df["AGE_GROUP"] = pd.cut(df["AGE_AT_SCAN"], bins=bins,
                             labels=labels, right=False)
    return df


def process_sex(sex_label: str, meta_path: Path):
    print(f"\n {sex_label.upper()} ")

    meta = pd.read_csv(meta_path)
    meta.columns = meta.columns.str.strip()
    meta = add_age_group_column(meta)

    required = {"FILE_ID", "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd", "AGE_GROUP"}
    missing  = required - set(meta.columns)
    if missing:
        raise ValueError(f"{meta_path} is missing columns: {missing}")

    for age_group in AGE_GROUPS:
        sub = meta[meta["AGE_GROUP"] == age_group].copy()
        if sub.empty:
            print(f"  [{sex_label} {age_group}] No subjects, skipping.")
            continue

        print(f"\n  [{sex_label} {age_group}] N = {len(sub)}")
        print(sub["DX_GROUP"].value_counts().rename({1: "ASD", 2: "CTL"}))

        rows = []
        n_bad = 0

        for _, row in sub.iterrows():
            fid     = row["FILE_ID"]
            conn_fp = CONN_DIR / f"{fid}.npy"

            if not conn_fp.exists():
                print(f"    [WARN] missing connectome: {conn_fp}")
                n_bad += 1
                continue

            mat = np.load(conn_fp)
            if mat.shape != (N_ROI, N_ROI):
                print(f"    [WARN] skipping {fid}: shape {mat.shape}")
                n_bad += 1
                continue

            W = threshold_top_density(mat, density=DENSITY)
            pc, z = compute_pc_z_weighted(W, roi2mod)

            for node_idx in range(N_ROI):
                rec = {
                    "FILE_ID": fid,
                    "sex": sex_label,
                    "AGE_GROUP": age_group,
                    "DX_GROUP": row["DX_GROUP"],
                    "AGE_AT_SCAN": row["AGE_AT_SCAN"],
                    "func_mean_fd": row["func_mean_fd"],
                    "node": node_idx + 1, #1..200
                    "module": int(roi2mod[node_idx]), #1..7
                    "PC": float(pc[node_idx]),
                    "z": float(z[node_idx]),
                }
                rows.append(rec)

        if not rows:
            print(f"  [{sex_label} {age_group}] No usable subjects after QC.")
            continue

        out_df  = pd.DataFrame(rows)
        out_path = OUT_DIR / f"{sex_label}_{age_group}_pc_z.csv"
        out_df.to_csv(out_path, index=False)
        print(
            f"  [{sex_label} {age_group}] "
            f"Saved {len(out_df)} node-rows -> {out_path} (skipped {n_bad} subjects)"
        )


process_sex("female", FEMALE_META)
process_sex("male", MALE_META)

print("\nDone.")
