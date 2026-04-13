import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
CONN_DIR  = ROOT / "connectomes" / "CC200" / "ABIDE1"
PHENO_DIR = ROOT / "phenotypes" / "ABIDE1"

OUT = ROOT / "results" / "group_connectomes" / "ABIDE1_CC200"
OUT.mkdir(parents=True, exist_ok=True)

FD_LIST = ["0.2", "0.3"]

# ABIDE conventions:
# SEX: 1=Male, 2=Female
# DX_GROUP: 1=ASD, 2=Control
GROUPS = [
    ("F_ASD", 2, 1),
    ("F_CTL", 2, 2),
    ("M_ASD", 1, 1),
    ("M_CTL", 1, 2),
]

def to_subid7(x) -> str | None:
    """Convert phenotype SUB_ID to 7-digit string used in filenames (e.g., 50004 -> '0050004')."""
    if pd.isna(x):
        return None
    try:
        return f"{int(float(x)):07d}"
    except Exception:
        return None

def find_npz(subid7: str, fd: str) -> Path | None:
    """Find connectome file; prefer run-1 if multiple runs exist."""
    pat = f"sub-{subid7}_task-rest_run-*_atlas-CC200_fd-{fd}_connectome.npz"
    cands = sorted(CONN_DIR.glob(pat))
    if not cands:
        return None
    for fp in cands:
        if "_run-1_" in fp.name:
            return fp
    return cands[0]

def load_z(fp: Path) -> np.ndarray:
    """Load Fisher-z matrix from .npz (key 'z') and sanity-check shape."""
    with np.load(fp, allow_pickle=False) as d:
        z = np.asarray(d["z"])
    if z.shape != (200, 200):
        raise ValueError(f"{fp.name}: expected (200,200), got {z.shape}")
    return z

# ---- load all phenotype CSVs (site-split) ----
dfs = []
for fp in sorted(PHENO_DIR.glob("phenotypic_*.csv")):
    df = pd.read_csv(fp)

    required = {"SUB_ID", "DX_GROUP", "SEX"}
    if not required.issubset(df.columns):
        print(f"[WARN] Skipping {fp.name} (missing one of {sorted(required)})")
        continue

    d2 = df[["SUB_ID", "DX_GROUP", "SEX"]].copy()
    d2["SITE_ID"] = df["SITE_ID"] if "SITE_ID" in df.columns else fp.stem.replace("phenotypic_", "")
    d2["subid7"] = d2["SUB_ID"].map(to_subid7)

    d2["DX_GROUP"] = pd.to_numeric(d2["DX_GROUP"], errors="coerce")
    d2["SEX"] = pd.to_numeric(d2["SEX"], errors="coerce")
    d2 = d2.dropna(subset=["subid7", "DX_GROUP", "SEX"])

    dfs.append(d2)

if not dfs:
    raise RuntimeError(f"No usable phenotype CSVs found in {PHENO_DIR}")

pheno = pd.concat(dfs, ignore_index=True)
print(f"[INFO] Loaded phenotype rows: {len(pheno)}")
print("[INFO] Unique sites:", pheno["SITE_ID"].nunique())

# ---- build group means (across all sites) ----
for fd in FD_LIST:
    print(f"\n=== FD {fd} ===")

    for gname, sex_val, dx_val in GROUPS:
        subset = pheno[(pheno["SEX"] == sex_val) & (pheno["DX_GROUP"] == dx_val)]
        subids = subset["subid7"].unique()

        mats = []
        used = []
        missing = 0
        loadfail = 0

        for subid7 in subids:
            fp = find_npz(subid7, fd)
            if fp is None:
                missing += 1
                continue
            try:
                mats.append(load_z(fp))
                used.append(subid7)
            except Exception as e:
                loadfail += 1
                print(f"[WARN] {gname} sub-{subid7} fd={fd}: {e}")

        if not mats:
            print(f"[INFO] {gname}: no usable connectomes (missing={missing}, loadfail={loadfail})")
            continue

        mats = np.stack(mats, axis=0)          # (n_subj, 200, 200)
        mean_z = np.nanmean(mats, axis=0)      # Fisher-z mean
        mean_r = np.tanh(mean_z)               # optional: back to correlation

        out_z = OUT / f"{gname}_fd-{fd}_mean_z.npy"
        out_r = OUT / f"{gname}_fd-{fd}_mean_r.npy"
        out_used = OUT / f"{gname}_fd-{fd}_used_subids.txt"

        np.save(out_z, mean_z)
        np.save(out_r, mean_r)
        out_used.write_text("\n".join(used), encoding="utf-8")

        print(f"{gname}: used={len(used)} missing={missing} loadfail={loadfail}")
        print(f"  saved {out_z}")

print("\n[DONE] Saved 4 group means for fd=0.2 and fd=0.3.")