import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
CONN_DIR  = ROOT / "connectomes" / "CC200" / "ABIDE1"
PHENO_DIR = ROOT / "phenotypes" / "ABIDE1"
OUT_DIR   = ROOT / "results" / "group_connectomes" / "ABIDE1_CC200"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FD_LIST = ["0.2", "0.3"]

# ABIDE conventions:
# SEX: 1=Male, 2=Female
# DX_GROUP: 1=ASD, 2=Control

def to_subid7(x) -> str | None:
    if pd.isna(x):
        return None
    try:
        return f"{int(float(x)):07d}"
    except Exception:
        return None

def age_group(age: float) -> str | None:
    if pd.isna(age):
        return None
    a = float(age)
    if a <= 9:
        return "child_0_9"
    if a <= 12:
        return "preteen_10_12"
    if a <= 17:
        return "teen_13_17"
    return "adult_18_plus"

def build_connectome_index(fd: str) -> dict[str, Path]:
    """
    Map subid7 -> npz path for this fd. Prefer run-1 if multiple.
    """
    idx = {}
    pat = f"*fd-{fd}_connectome.npz"
    for fp in CONN_DIR.glob(pat):
        m = re.search(r"sub-(\d{7})_task-rest_run-(\d+)_", fp.name)
        if not m:
            continue
        subid7 = m.group(1)
        run = int(m.group(2))

        # prefer run-1; otherwise keep first seen
        if subid7 not in idx:
            idx[subid7] = fp
        else:
            # if current is run-1 and stored isn't run-1, replace
            if run == 1 and "_run-1_" not in idx[subid7].name:
                idx[subid7] = fp
    return idx

def load_npz_meta(fp: Path) -> tuple[float | None, float | None]:
    """
    Return (n_kept, n_total) if present; else (None, None).
    """
    with np.load(fp, allow_pickle=False) as d:
        n_kept = float(d["n_kept"]) if "n_kept" in d.files else None
        n_total = float(d["n_total"]) if "n_total" in d.files else None
    return n_kept, n_total

def load_z(fp: Path) -> np.ndarray:
    with np.load(fp, allow_pickle=False) as d:
        z = np.asarray(d["z"])
    if z.shape != (200, 200):
        raise ValueError(f"{fp.name}: expected (200,200), got {z.shape}")
    return z

def robust_standardize(x: np.ndarray) -> np.ndarray:
    """
    Standardize with median + MAD-ish scale to be stable with outliers.
    """
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = mad if (mad is not None and mad > 1e-12) else (np.nanstd(x) + 1e-12)
    return (x - med) / scale

# ---- Load all phenotypic_*.csv ----
dfs = []
for fp in sorted(PHENO_DIR.glob("phenotypic_*.csv")):
    df = pd.read_csv(fp)
    needed = {"SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN"}
    if not needed.issubset(df.columns):
        print(f"[WARN] Skipping {fp.name} (missing one of {sorted(needed)})")
        continue
    d2 = df.copy()
    d2["subid7"] = d2["SUB_ID"].map(to_subid7)
    d2["age_group"] = d2["AGE_AT_SCAN"].map(age_group)
    dfs.append(d2)

if not dfs:
    raise RuntimeError(f"No usable phenotypic_*.csv found in {PHENO_DIR}")

pheno = pd.concat(dfs, ignore_index=True)

# Keep only relevant columns (but tolerate missing FIQ)
keep_cols = ["SITE_ID", "SUB_ID", "subid7", "DX_GROUP", "SEX", "AGE_AT_SCAN", "age_group"]
if "FIQ" in pheno.columns:
    keep_cols.append("FIQ")

pheno = pheno[keep_cols].copy()
pheno["DX_GROUP"] = pd.to_numeric(pheno["DX_GROUP"], errors="coerce")
pheno["SEX"] = pd.to_numeric(pheno["SEX"], errors="coerce")
pheno["AGE_AT_SCAN"] = pd.to_numeric(pheno["AGE_AT_SCAN"], errors="coerce")
if "FIQ" in pheno.columns:
    pheno["FIQ"] = pd.to_numeric(pheno["FIQ"], errors="coerce")

pheno = pheno.dropna(subset=["subid7", "DX_GROUP", "SEX", "AGE_AT_SCAN", "age_group"])
pheno["subid7"] = pheno["subid7"].astype(str)

print(f"[INFO] Phenotype rows loaded: {len(pheno)}")

# ---- For each FD, match males to females per (DX_GROUP, age_group), then compute mean ----
for fd in FD_LIST:
    print(f"\n====================")
    print(f"FD = {fd}")
    print(f"====================")

    conn_idx = build_connectome_index(fd)
    print(f"[INFO] Connectomes available for fd={fd}: {len(conn_idx)}")

    # Keep only subjects that have a connectome file for this fd
    has_conn = pheno["subid7"].isin(conn_idx.keys())
    df = pheno[has_conn].copy()
    df["conn_path"] = df["subid7"].map(conn_idx)

    # Add frac_kept from npz scalars (only for subjects in df)
    frac_kept = []
    for p in df["conn_path"]:
        nk, nt = load_npz_meta(p)
        if nk is None or nt is None or nt == 0:
            frac_kept.append(np.nan)
        else:
            frac_kept.append(nk / nt)
    df["frac_kept"] = frac_kept

    print(f"[INFO] Subjects with connectomes for fd={fd}: {len(df)}")

    selected_rows = []

    # Iterate strata: diagnosis + age group
    for (dx, ag), stratum in df.groupby(["DX_GROUP", "age_group"]):
        females = stratum[stratum["SEX"] == 2].copy()
        males   = stratum[stratum["SEX"] == 1].copy()

        nF = len(females)
        nM = len(males)
        if nF == 0 or nM == 0:
            continue

        n = min(nF, nM)  # should usually be nF (since males > females)

        # If we ever have more females than males, downsample females too (rare)
        if nF > n:
            # pick "most representative females" (closest to female centroid = itself)
            females = females.sample(n=n, random_state=42)

        # Build covariates for representativeness matching
        # Use AGE_AT_SCAN, FIQ (if available), frac_kept
        cols = ["AGE_AT_SCAN", "frac_kept"]
        if "FIQ" in females.columns:
            cols.append("FIQ")

        # Fill missing covariates with stratum medians
        for c in cols:
            med = np.nanmedian(stratum[c].to_numpy())
            females[c] = females[c].fillna(med)
            males[c] = males[c].fillna(med)

        # Female centroid in standardized space
        F = females[cols].to_numpy(dtype=float)
        M = males[cols].to_numpy(dtype=float)

        # robust standardize each column using combined stratum values
        ZF = []
        ZM = []
        for j, c in enumerate(cols):
            combined = np.concatenate([F[:, j], M[:, j]])
            z_comb = robust_standardize(combined)
            zF = z_comb[: F.shape[0]]
            zM = z_comb[F.shape[0] :]
            ZF.append(zF)
            ZM.append(zM)
        ZF = np.stack(ZF, axis=1)
        ZM = np.stack(ZM, axis=1)

        centroid = np.mean(ZF, axis=0)

        # Score males by distance to female centroid (smaller = more representative)
        dists = np.sum((ZM - centroid) ** 2, axis=1)
        males = males.assign(match_score=dists)
        males_sel = males.nsmallest(n, "match_score")

        # Keep all females (after optional downsample) and matched males
        females = females.assign(match_score=np.nan)
        selected_rows.append(females)
        selected_rows.append(males_sel)

        print(f"[STRATUM] DX={int(dx)} age={ag}: females={len(females)} males_selected={len(males_sel)} (males_avail={nM})")

    if not selected_rows:
        raise RuntimeError(f"No matched subjects found for fd={fd}")

    sel = pd.concat(selected_rows, ignore_index=True)
    sel = sel.drop_duplicates(subset=["subid7"])  # safety

    # Sanity: counts by sex
    sex_counts = sel["SEX"].value_counts().to_dict()
    print(f"[INFO] Selected total={len(sel)} sex_counts={sex_counts}")

    # ---- Compute streaming nan-mean of z matrices ----
    sum_mat = np.zeros((200, 200), dtype=np.float64)
    cnt_mat = np.zeros((200, 200), dtype=np.int32)

    for fp in sel["conn_path"]:
        z = load_z(fp).astype(np.float64)
        mask = ~np.isnan(z)
        sum_mat[mask] += z[mask]
        cnt_mat[mask] += 1

    mean_z = np.full((200, 200), np.nan, dtype=np.float64)
    valid = cnt_mat > 0
    mean_z[valid] = sum_mat[valid] / cnt_mat[valid]
    mean_r = np.tanh(mean_z)

    out_z = OUT_DIR / f"OVERALL_ageSexMatched_fd-{fd}_mean_z.npy"
    out_r = OUT_DIR / f"OVERALL_ageSexMatched_fd-{fd}_mean_r.npy"
    out_csv = OUT_DIR / f"OVERALL_ageSexMatched_fd-{fd}_selected_subjects.csv"

    np.save(out_z, mean_z)
    np.save(out_r, mean_r)
    sel.to_csv(out_csv, index=False)

    print(f"[DONE] Saved:")
    print(f"  {out_z}")
    print(f"  {out_r}")
    print(f"  {out_csv}")