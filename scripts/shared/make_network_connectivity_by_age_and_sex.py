import numpy as np
import pandas as pd
from pathlib import Path
import os

ROOT = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")

CONN_DIR = ROOT / "data/connectomes/cpac/nofilt_noglobal/cc200_z"
MODULE_FILE = ROOT / "results/group_connectomes/CC200_modules.txt"

FEMALE_META = ROOT / "data/female/female_metadata_included.csv"
MALE_META   = ROOT / "data/male/male_metadata_included.csv"

OUT_DIR = ROOT / "results/network_level/2"
OUT_DIR.mkdir(exist_ok=True, parents=True)

#age groups we care about
AGE_GROUPS = ["child", "teen", "young_adult"]

#Load module assignments (Louvain solution) 

mods = pd.read_csv(MODULE_FILE, sep=r"\s+")
mods = mods.sort_values("ROI_index")

#roi2mod[i] = module id for ROI i
roi2mod = mods["Module"].to_numpy()

N_ROI = len(roi2mod)
N_MOD = int(roi2mod.max())

print(f"Loaded module file with {N_ROI} ROIs, {N_MOD} modules")

#precompute ROI indices for each module (1..N_MOD)
mod_idx = {m: np.where(roi2mod == m)[0] for m in range(1, N_MOD + 1)}


def compute_network_means(mat: np.ndarray) -> dict:
    if mat.shape != (N_ROI, N_ROI):
        raise ValueError(f"Expected {(N_ROI, N_ROI)}, got {mat.shape}")

    out = {}
    for i in range(1, N_MOD + 1):
        idx_i = mod_idx[i]
        for j in range(i, N_MOD + 1):
            idx_j = mod_idx[j]
            block = mat[np.ix_(idx_i, idx_j)]

            if i == j:
                #within-module: upper triangle excluding diagonal
                r, c = np.triu_indices(len(idx_i), k=1)
                vals = block[r, c]
            else:
                vals = block.ravel()

            vals = vals[~np.isnan(vals)]
            out[f"net{i}_{j}"] = float(vals.mean()) if vals.size > 0 else np.nan

    return out


def add_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 13, 18, 30, 45, 120]
    labels = ["child", "teen", "young_adult", "adult", "older"]
    df = df.copy()
    df["AGE_GROUP"] = pd.cut(df["AGE_AT_SCAN"], bins=bins,
                             labels=labels, right=False)
    return df


def build_for_sex(sex_label: str, meta_path: Path):
    print(f"\n=== {sex_label.upper()} ===")

    meta = pd.read_csv(meta_path)
    #create AGE_GROUP from age
    meta = add_age_group_column(meta)

    required_cols = {"FILE_ID", "DX_GROUP", "AGE_AT_SCAN",
                     "func_mean_fd", "AGE_GROUP"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"{meta_path} is missing columns: {missing}")

    for age_group in AGE_GROUPS:
        sub = meta[meta["AGE_GROUP"] == age_group].copy()
        if sub.empty:
            print(f"[{sex_label} {age_group}] No subjects, skipping.")
            continue

        print(f"\n[{sex_label} {age_group}] N={len(sub)}")
        print(sub["DX_GROUP"].value_counts().rename({1: "ASD", 2: "CTL"}))

        rows = []
        n_bad = 0

        for _, row in sub.iterrows():
            fid = row["FILE_ID"]
            conn_fp = CONN_DIR / f"{fid}.npy"

            if not conn_fp.exists():
                print(f"  [WARN] missing connectome: {conn_fp}")
                n_bad += 1
                continue

            mat = np.load(conn_fp)
            if mat.shape != (N_ROI, N_ROI):
                print(f"  [WARN] skipping {fid}: shape {mat.shape}")
                n_bad += 1
                continue

            nets = compute_network_means(mat)
            rec = {
                "FILE_ID": fid,
                "DX_GROUP": row["DX_GROUP"],
                "AGE_AT_SCAN": row["AGE_AT_SCAN"],
                "func_mean_fd": row["func_mean_fd"],
                "AGE_GROUP": row["AGE_GROUP"],
            }
            rec.update(nets)
            rows.append(rec)

        if not rows:
            print(f"[{sex_label} {age_group}] No usable subjects after QC/shape checks.")
            continue

        out_df = pd.DataFrame(rows)
        out_path = OUT_DIR / f"{sex_label}_{age_group}_network_connectivity_wide.csv"
        out_df.to_csv(out_path, index=False)

        print(
            f"[{sex_label} {age_group}] "
            f"Saved {len(out_df)} subjects to {out_path} "
            f"(skipped {n_bad} problematic subjects)"
        )


#run for female and male
build_for_sex("female", FEMALE_META)
build_for_sex("male", MALE_META)

print("\nDone.")