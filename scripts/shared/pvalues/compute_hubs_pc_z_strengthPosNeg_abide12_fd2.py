import numpy as np
import pandas as pd
from pathlib import Path

# --- PROJECT PATHS ---
ROOT = Path(r"C:\Users\eliza\Connectomics\TERMproject\abide-sex-age-connectomics")
CONN_DIR = ROOT / "data/connectomes/cpac/nofilt_noglobal/cc200_z"
MODULE_FILE = ROOT / "results/group_connectomes/CC200_modules_ALLSUBJ_signed_asym1000.txt"
META_COMBINED = ROOT / "data/metadata/ABIDE12_phenotypes_combined.csv"
OUT_DIR = ROOT / "results/hubs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# --- SETTINGS ---
DENSITY = 0.0797
BINS    = [0, 10, 13, 18, 200]
LABELS  = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
RIGHT   = False  # [0,10), [10,13), [13,18), [18,200)

# ----------------------------
# Helpers
# ----------------------------
def threshold_top_density(mat: np.ndarray, density: float) -> np.ndarray:
    n = mat.shape[0]
    mat = mat.copy()
    np.fill_diagonal(mat, 0.0)
    iu, ju = np.triu_indices(n, k=1)
    vals = np.abs(mat[iu, ju])
    k = int(np.floor(density * len(vals)))
    if k < 1: return np.zeros_like(mat)
    thresh = np.partition(vals, -k)[-k]
    mask = np.abs(mat) >= thresh
    out = np.where(mask, mat, 0.0)
    out = np.triu(out, 1)
    out = out + out.T
    np.fill_diagonal(out, 0.0)
    return out

def compute_pc_z_strength_posneg(W: np.ndarray, roi2mod: np.ndarray, n_roi, n_mod):
    Wp = np.where(W > 0, W, 0.0)
    Wn = np.where(W < 0, -W, 0.0)
    Wabs = Wp + Wn
    strength_pos = Wp.sum(axis=1)
    strength_neg = Wn.sum(axis=1)
    strength_abs = Wabs.sum(axis=1)

    k_by_mod = np.zeros((n_roi, n_mod), dtype=float)
    for m in range(1, n_mod + 1):
        idx = np.where(roi2mod == m)[0]
        if idx.size > 0:
            k_by_mod[:, m - 1] = Wabs[:, idx].sum(axis=1)

    pc = np.zeros(n_roi, dtype=float)
    for i in range(n_roi):
        ki = strength_abs[i]
        if ki > 0:
            pc[i] = 1.0 - np.sum((k_by_mod[i, :] / ki)**2)

    z = np.zeros(n_roi, dtype=float)
    for m in range(1, n_mod + 1):
        idx = np.where(roi2mod == m)[0]
        if idx.size > 0:
            k_within = k_by_mod[idx, m - 1]
            mu, sd = k_within.mean(), k_within.std(ddof=1)
            z[idx] = (k_within - mu) / sd if sd > 0 else 0.0
    return pc, z, strength_pos, strength_neg

# ----------------------------
# Main Processing
# ----------------------------
def main():
    # 1. Load Modules
    if not MODULE_FILE.exists():
        raise FileNotFoundError(f"Module file not found: {MODULE_FILE}")
    mods = pd.read_csv(MODULE_FILE, sep=r"\s+")
    mods = mods.sort_values("ROI_index")
    roi2mod = mods["Module"].to_numpy()
    n_roi, n_mod = len(roi2mod), int(roi2mod.max())
    print(f"[INFO] Loaded module file: {n_roi} ROIs, {n_mod} modules")

    # 2. Load Metadata
    print(f"[INFO] Reading metadata: {META_COMBINED.name}")
    meta = pd.read_csv(META_COMBINED)
    meta.columns = meta.columns.str.strip().str.upper()

    # FIX: Remove duplicate column names that cause ValueError
    if meta.columns.duplicated().any():
        dupes = meta.columns[meta.columns.duplicated()].unique().tolist()
        print(f"[WARN] Duplicate columns detected: {dupes}. Keeping first occurrence.")
        meta = meta.loc[:, ~meta.columns.duplicated()]

    # Map SEX and AGE
    if "SEX" not in meta.columns:
        raise KeyError("Could not find 'SEX' column in metadata.")
    meta["SEX_LABEL"] = meta["SEX"].replace({1: "male", 2: "female", "1": "male", "2": "female"})

    if "AGE_AT_SCAN" not in meta.columns and "AGE" in meta.columns:
        meta["AGE_AT_SCAN"] = meta["AGE"]
    
    if "AGE_AT_SCAN" not in meta.columns:
        raise KeyError("Could not find 'AGE_AT_SCAN' or 'AGE' column in metadata.")
    
    meta["AGE_GROUP"] = pd.cut(meta["AGE_AT_SCAN"], bins=BINS, labels=LABELS, right=RIGHT, include_lowest=True)
    meta["FILE_ID"] = meta["FILE_ID"].astype(str).str.strip()

    # 3. Process Subjects
    for sex in ["female", "male"]:
        for age_group in LABELS:
            sub = meta[(meta["SEX_LABEL"] == sex) & (meta["AGE_GROUP"] == age_group)].copy()
            if sub.empty:
                continue

            print(f"  [PROCESSING] {sex} {age_group} (N={len(sub)})")
            rows = []
            n_bad = 0

            for _, row in sub.iterrows():
                fid = row["FILE_ID"]
                # Try both .npy and no extension if needed
                conn_fp = CONN_DIR / f"{fid}.npy"
                
                if not conn_fp.exists():
                    n_bad += 1
                    continue

                try:
                    mat = np.load(conn_fp)
                except Exception:
                    n_bad += 1
                    continue

                if mat.shape != (n_roi, n_roi) or not np.isfinite(mat).all():
                    n_bad += 1
                    continue

                W = threshold_top_density(mat, density=DENSITY)
                pc, z, sp, sn = compute_pc_z_strength_posneg(W, roi2mod, n_roi, n_mod)

                for node_idx in range(n_roi):
                    rows.append({
                        "subject_int": int(row["SUB_ID"]),
                        "FILE_ID": fid,
                        "DX_GROUP": int(row["DX_GROUP"]),
                        "node": node_idx + 1,
                        "module": int(roi2mod[node_idx]),
                        "PC": float(pc[node_idx]),
                        "z": float(z[node_idx]),
                        "strength_pos": float(sp[node_idx]),
                        "strength_neg": float(sn[node_idx]),
                    })

            if rows:
                out_df = pd.DataFrame(rows)
                out_name = f"{sex}_{age_group}_pc_z_strengthPosNeg_abide12_fd2.csv"
                out_path = OUT_DIR / out_name
                out_df.to_csv(out_path, index=False)
                print(f"    [SAVED] {out_name} (skipped {n_bad} files)")

if __name__ == "__main__":
    main()