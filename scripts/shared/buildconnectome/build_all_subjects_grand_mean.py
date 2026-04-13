import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

CONN_DIR = ROOT / r"data\connectomes\cpac\nofilt_noglobal\cc200_z"

# use cleaned if available, else fallback
PHENO_CLEAN = ROOT / r"data\Phenotypic_V1_0b_preprocessed1_clean.csv"
PHENO_RAW   = ROOT / r"data\Phenotypic_V1_0b_preprocessed1.csv"
PHENO_CSV = PHENO_CLEAN if PHENO_CLEAN.exists() else PHENO_RAW

OUT_DIR = ROOT / r"results\group_connectomes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MEAN  = OUT_DIR / "ALL_SUBJECTS_Zmean.npy"          # (200,200) grand mean
OUT_STACK = OUT_DIR / "ALL_SUBJECTS_STACK.npy"          # (N,200,200) optional
OUT_IDS   = OUT_DIR / "ALL_SUBJECTS_file_ids.csv"

EXPECTED_N = 200

def load_connectome(fid: str) -> np.ndarray | None:
    fp = CONN_DIR / f"{fid}.npy"
    if not fp.exists():
        return None
    try:
        A = np.load(fp)
    except Exception:
        return None
    if A.shape != (EXPECTED_N, EXPECTED_N):
        return None
    if not np.isfinite(A).all():
        # keep it strict for this “confirm correctness” run
        return None
    return A.astype(np.float64)

def main():
    if not PHENO_CSV.exists():
        raise FileNotFoundError(f"Missing phenotypic CSV: {PHENO_CSV}")
    if not CONN_DIR.exists():
        raise FileNotFoundError(f"Missing connectome directory: {CONN_DIR}")

    ph = pd.read_csv(PHENO_CSV)
    ph.columns = ph.columns.str.strip()

    if "FILE_ID" not in ph.columns:
        raise ValueError("Phenotypic CSV is missing FILE_ID column")

    # clean file IDs
    ph["FILE_ID"] = ph["FILE_ID"].astype(str).str.strip()
    ph = ph[ph["FILE_ID"].notna()].copy()
    ph = ph[ph["FILE_ID"] != ""].copy()
    ph = ph[ph["FILE_ID"] != "no_filename"].copy()

    # unique subjects (avoid duplicates)
    file_ids = ph["FILE_ID"].unique().tolist()

    mats = []
    kept_ids = []
    missing = 0
    badshape_or_nan = 0

    for fid in file_ids:
        A = load_connectome(fid)
        if A is None:
            # count why it was dropped
            fp = CONN_DIR / f"{fid}.npy"
            if not fp.exists():
                missing += 1
            else:
                badshape_or_nan += 1
            continue
        mats.append(A)
        kept_ids.append(fid)

    N = len(mats)
    print(f"Phenotypic FILE_IDs (unique): {len(file_ids)}")
    print(f"Kept subjects with valid 200x200 finite connectomes: {N}")
    print(f"Dropped (missing .npy): {missing}")
    print(f"Dropped (bad shape or NaN/Inf): {badshape_or_nan}")

    if N < 10:
        raise RuntimeError("Too few subjects loaded — check paths or file naming.")

    stack = np.stack(mats, axis=0)          # (N,200,200)
    meanA = stack.mean(axis=0)              # (200,200)
    meanA = 0.5 * (meanA + meanA.T)         # symmetrize defensively
    np.fill_diagonal(meanA, 0.0)

    np.save(OUT_STACK, stack)
    np.save(OUT_MEAN, meanA)
    pd.DataFrame({"FILE_ID": kept_ids}).to_csv(OUT_IDS, index=False)

    print(f"\nSaved stack -> {OUT_STACK}")
    print(f"Saved grand mean -> {OUT_MEAN}")
    print(f"Saved included IDs -> {OUT_IDS}")
    print(f"Grand mean stats: min={meanA.min():.6f}, max={meanA.max():.6f}, neg_frac={(meanA<0).mean():.6f}")

if __name__ == "__main__":
    main()
