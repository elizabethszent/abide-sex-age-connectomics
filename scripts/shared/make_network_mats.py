import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

#200×200 Fisher-z connectomes
CONN_DIR = BASE / r"data\connectomes\cpac\nofilt_noglobal\cc200_z"

#Louvain modules (your 7-module solution)
MODULE_PATH = BASE / r"results\group_connectomes\CC200_modules.npy"

#Metadata (QC-filtered)
FEMALE_META = BASE / r"data\female\female_metadata_included.csv"
MALE_META   = BASE / r"data\male\male_metadata_included.csv"

OUT_DIR = BASE / r"results\network_level"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#helpers
def load_modules():
    modules = np.load(MODULE_PATH)
    if modules.ndim != 1 or modules.shape[0] != 200:
        raise ValueError(f"Expected modules shape (200,), got {modules.shape}")
    # relabel to 0..K-1
    uniq = np.unique(modules)
    mapping = {old: i for i, old in enumerate(uniq)}
    modules = np.array([mapping[x] for x in modules], dtype=int)
    return modules, len(uniq)

def subj_list(meta_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    if "FILE_ID" not in df.columns:
        raise ValueError(f"{meta_csv} is missing FILE_ID column")
    return df

def conn_path(file_id: str) -> Path:
    return CONN_DIR / f"{file_id}.npy"

def make_netmat(C: np.ndarray, modules: np.ndarray, K: int) -> np.ndarray:
    C = C.copy()
    np.fill_diagonal(C, 0.0)

    netmat = np.zeros((K, K), dtype=float)
    counts = np.zeros((K, K), dtype=int)

    for i in range(200):
        mi = modules[i]
        for j in range(i + 1, 200):
            mj = modules[j]
            w = C[i, j]
            netmat[mi, mj] += w
            netmat[mj, mi] += w
            counts[mi, mj] += 1
            counts[mj, mi] += 1

    mask = counts > 0
    netmat[mask] /= counts[mask]
    return netmat

#main
modules, K = load_modules()
print(f"Loaded module labels: K = {K}")

for label, meta_path in [
    ("female", FEMALE_META),
    ("male",   MALE_META),
]:
    df = subj_list(meta_path)
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        fid = str(row["FILE_ID"])
        cp = conn_path(fid)
        if not cp.exists():
            print(f"[{label}] missing connectome for {fid} at {cp}")
            skipped += 1
            continue

        C = np.load(cp)
        if C.shape != (200, 200):
            print(f"[{label}] skipping {fid}: connectome shape {C.shape}, expected (200, 200)")
            skipped += 1
            continue

        netmat = make_netmat(C, modules, K)
        tri = netmat[np.triu_indices(K)]

        row_dict = {
            "FILE_ID": fid,
            "DX_GROUP": row.get("DX_GROUP", np.nan),
            "AGE_AT_SCAN": row.get("AGE_AT_SCAN", np.nan),
            "SITE_ID": row.get("SITE_ID", np.nan),
            "func_mean_fd": row.get("func_mean_fd", np.nan),
        }
        for (i, j), v in zip(zip(*np.triu_indices(K)), tri):
            row_dict[f"net_{i}_{j}"] = v
        rows.append(row_dict)

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / f"{label}_network_mats.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(out_df)} subjects (skipped {skipped})")
