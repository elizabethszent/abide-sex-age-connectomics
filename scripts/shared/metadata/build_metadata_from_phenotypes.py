# scripts/shared/metadata/build_abide2_metadata_included.py
# Builds:
#   data/female/female_metadata_included_abide2.csv
#   data/male/male_metadata_included_abide2.csv
#   data/metadata/ABIDE2_phenotypes_combined.csv
#
# Uses BOTH ABIDE2 phenotype files:
#   phenotypes/ABIDE2/abide2_composite_pheno.csv
#   phenotypes/ABIDE2/abide2_composite_pheno_longitudinal.csv   (if present)
#
# Notes:
# - ABIDE2 files often use AGE_AT_SCAN_YEARS (not AGE_AT_SCAN)
# - Some ABIDE2 files use DX_GROUP or DX_GROUP_OLD; we try a few candidates
# - Some use SITE_ID or SITE; we try a few candidates
# - We output FILE_ID as a plain numeric string (e.g., "28743") so it matches
#   your ABIDE2 connectome naming like: sub-28743_ses-1_task-rest_run-1.npy

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

PHENO_DIR = ROOT / "phenotypes" / "ABIDE2"

PHENO_FILES = [
    PHENO_DIR / "abide2_composite_pheno.csv",
    PHENO_DIR / "abide2_composite_pheno_longitudinal.csv",
]

# Output locations (ABIDE2-specific so you don't overwrite ABIDE1)
FEMALE_OUT   = ROOT / "data" / "female" / "female_metadata_included_abide2.csv"
MALE_OUT     = ROOT / "data" / "male"   / "male_metadata_included_abide2.csv"
COMBINED_OUT = ROOT / "data" / "metadata" / "ABIDE2_phenotypes_combined.csv"

# Requested bins
BINS   = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
RIGHT  = False  # [0,10), [10,13), [13,18), [18,200)

# -------------------------
# Robust CSV reader
# -------------------------
def read_csv_robust(fp: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(fp, encoding="utf-8", encoding_errors="replace")


def pick_first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.upper(): c for c in df.columns}  # map upper->original
    for cand in candidates:
        if cand.upper() in cols:
            return cols[cand.upper()]
    return None


def load_abide2_phenos() -> pd.DataFrame:
    frames = []
    used = 0

    for fp in PHENO_FILES:
        if not fp.exists():
            continue
        df = read_csv_robust(fp)
        df.columns = df.columns.str.strip()

        # find required columns (ABIDE2 varies)
        col_sub  = pick_first_present(df, ["SUB_ID", "SUBID", "SUBJECT", "SUBJECT_ID"])
        col_sex  = pick_first_present(df, ["SEX"])
        col_site = pick_first_present(df, ["SITE_ID", "SITE", "SITEID"])
        col_dx   = pick_first_present(df, ["DX_GROUP", "DX_GROUP_OLD", "DX", "DIAGNOSIS"])
        col_age  = pick_first_present(df, ["AGE_AT_SCAN", "AGE_AT_SCAN_YEARS", "AGE", "AGE_YEARS"])

        if not (col_sub and col_sex and col_site and col_dx and col_age):
            print(f"[WARN] Skipping {fp.name} (missing one of: SUB_ID, SEX, SITE, DX, AGE)")
            continue

        keep = {
            "SUB_ID": col_sub,
            "SEX": col_sex,
            "SITE_ID": col_site,
            "DX_GROUP": col_dx,
            "AGE_AT_SCAN": col_age,
        }

        out = df[[keep[k] for k in keep]].copy()
        out = out.rename(columns={v: k for k, v in keep.items()})

        # If func_mean_fd exists in this table, keep it; else fill later
        col_fd = pick_first_present(df, ["func_mean_fd", "MEAN_FD", "FUNC_MEAN_FD"])
        if col_fd is not None:
            out["func_mean_fd"] = df[col_fd].copy()
        else:
            out["func_mean_fd"] = np.nan

        out["PHENO_FILE"] = fp.name
        frames.append(out)
        used += 1

    if not frames:
        raise FileNotFoundError(
            f"No usable ABIDE2 phenotype files found under {PHENO_DIR}.\n"
            f"Expected at least: {PHENO_FILES[0].name} (and optionally longitudinal)."
        )

    ph = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded phenotype rows: {len(ph)} from {used} file(s).")
    return ph


def standardize(ph: pd.DataFrame) -> pd.DataFrame:
    df = ph.copy()

    # strip strings
    for c in ["SITE_ID", "PHENO_FILE"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # numeric cleanup
    df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce")
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
    df["func_mean_fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]).copy()

    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)

    # ABIDE convention
    df["sex"] = df["SEX"].map({1: "male", 2: "female"}).fillna("unknown")

    # For ABIDE2 connectomes you showed: sub-28743_ses-1_task-rest_run-1.npy
    # Your hubs script extracts digits from FILE_ID, so using numeric-only is safest.
    df["FILE_ID"] = df["SUB_ID"].astype(str)

    # Helpful extras (not strictly required)
    df["SUB_ID_Z7"] = df["SUB_ID"].astype(str).str.zfill(7)
    df["BIDS_ID"] = "sub-" + df["SUB_ID"].astype(str)  # ABIDE2 typically not zero-padded

    # Age bins
    df["AGE_GROUP"] = pd.cut(
        df["AGE_AT_SCAN"],
        bins=BINS,
        labels=LABELS,
        right=RIGHT,
        include_lowest=True,
    )

    # De-duplicate: keep first occurrence per subject (prefer rows with non-null fd if available)
    df["_fd_missing"] = df["func_mean_fd"].isna().astype(int)
    df = df.sort_values(["SUB_ID", "_fd_missing", "SITE_ID", "PHENO_FILE"]).drop_duplicates(
        subset=["SUB_ID"], keep="first"
    )
    df = df.drop(columns=["_fd_missing"])

    return df


def write_outputs(df: pd.DataFrame):
    COMBINED_OUT.parent.mkdir(parents=True, exist_ok=True)
    FEMALE_OUT.parent.mkdir(parents=True, exist_ok=True)
    MALE_OUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(COMBINED_OUT, index=False)
    print(f"[SAVED] {COMBINED_OUT}")

    female = df[df["sex"] == "female"].copy()
    male = df[df["sex"] == "male"].copy()

    female.to_csv(FEMALE_OUT, index=False)
    male.to_csv(MALE_OUT, index=False)

    print(f"[SAVED] {FEMALE_OUT}  (N={len(female)})")
    print(f"[SAVED] {MALE_OUT}    (N={len(male)})")

    print("\n[COUNTS] Sex:")
    print(df["sex"].value_counts(dropna=False))

    print("\n[COUNTS] Sex x DX_GROUP:")
    print(pd.crosstab(df["sex"], df["DX_GROUP"]))

    print("\n[COUNTS] Sex x AGE_GROUP:")
    print(pd.crosstab(df["sex"], df["AGE_GROUP"]))


def main():
    ph = load_abide2_phenos()
    df = standardize(ph)

    print(f"[INFO] Unique subjects: {df['SUB_ID'].nunique()}")
    print(f"[INFO] Unique sites   : {df['SITE_ID'].nunique()}")

    write_outputs(df)
    print("\n[DONE] ABIDE2 metadata rebuilt from composite phenotype files.")


if __name__ == "__main__":
    main()