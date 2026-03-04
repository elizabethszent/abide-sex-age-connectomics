import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\Connectomics\TERMProject\abide-sex-age-connectomics")

# ==========================
# INPUTS: ABIDE12
# ==========================
PHENO_DIR1 = ROOT / "phenotypes" / "ABIDE1"
PHENO_DIR2 = ROOT / "phenotypes" / "ABIDE2"

# ==========================
# OUTPUTS (ABIDE12-specific)
# ==========================
FEMALE_OUT   = ROOT / "data" / "female" / "female_metadata_included_abide12.csv"
MALE_OUT     = ROOT / "data" / "male"   / "male_metadata_included_abide12.csv"
COMBINED_OUT = ROOT / "data" / "metadata" / "ABIDE12_phenotypes_combined.csv"

# Age bins
BINS   = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
RIGHT  = False  # [0,10), [10,13), [13,18), [18,200)

# We will accept these core fields (with some flexible renaming)
REQUIRED_CORE = {"SUB_ID", "DX_GROUP", "SEX"}
AGE_CANDIDATES = ["AGE_AT_SCAN", "AGE_AT_SCAN_YEARS", "AGE", "AGE_YRS"]
SITE_CANDIDATES = ["SITE_ID", "SITE"]


def read_csv_robust(fp: Path) -> pd.DataFrame:
    """Try common encodings so ABIDE2 composite/longitudinal doesn't crash."""
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(fp, encoding="utf-8", encoding_errors="replace")


def canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, uppercase, collapse whitespace to underscores."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().upper()) for c in df.columns]
    return df


def find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_all_pheno_csvs(pheno_dir: Path, dataset_label: str) -> pd.DataFrame:
    fps = sorted(pheno_dir.rglob("*.csv"))
    if not fps:
        raise FileNotFoundError(f"No CSVs found under {pheno_dir}")

    frames = []
    skipped = 0

    for fp in fps:
        try:
            df = read_csv_robust(fp)
            df = canon_cols(df)
        except Exception:
            skipped += 1
            continue

        age_col = find_first_col(df, AGE_CANDIDATES)
        site_col = find_first_col(df, SITE_CANDIDATES)

        # must have core + some age column
        if (age_col is None) or (not REQUIRED_CORE.issubset(df.columns)):
            skipped += 1
            continue

        # keep flexible site col if present
        keep = []
        if site_col is not None:
            keep.append(site_col)
        keep += ["SUB_ID", "DX_GROUP", "SEX", age_col]

        # optional motion column (if already present in some tables)
        if "FUNC_MEAN_FD" in df.columns:
            keep.append("FUNC_MEAN_FD")
        elif "FUNC_MEAN_FD " in df.columns:
            keep.append("FUNC_MEAN_FD ")

        sub = df[keep].copy()
        sub = sub.rename(columns={age_col: "AGE_AT_SCAN"})
        if site_col is not None and site_col != "SITE_ID":
            sub = sub.rename(columns={site_col: "SITE_ID"})

        sub["PHENO_FILE"] = fp.name
        sub["DATASET"] = dataset_label
        frames.append(sub)

    if not frames:
        raise RuntimeError(f"No usable phenotype CSVs found in {pheno_dir} (missing required columns).")

    out = pd.concat(frames, ignore_index=True)
    print(f"[INFO] {dataset_label}: Loaded phenotype rows: {len(out)} (skipped files: {skipped})")
    return out


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = canon_cols(df)

    # Ensure required columns exist
    if "SITE_ID" not in df.columns:
        df["SITE_ID"] = np.nan
    if "FUNC_MEAN_FD" not in df.columns:
        df["FUNC_MEAN_FD"] = np.nan

    # numeric cleanup
    df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce")
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN"]).copy()

    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)
    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

    # ABIDE convention: SEX 1=male, 2=female
    df["sex"] = df["SEX"].map({1: "male", 2: "female"}).fillna("unknown")

    # Helpful ID variants:
    df["SUB_ID_Z7"] = df["SUB_ID"].astype(str).str.zfill(7)  # "0050004"
    df["FILE_ID_NUM"] = df["SUB_ID_Z7"]
    df["BIDS_ID"] = "sub-" + df["SUB_ID_Z7"]

    # Default FILE_ID used by a lot of your older scripts
    df["FILE_ID"] = df["FILE_ID_NUM"]

    # Keep func_mean_fd column expected by hubs scripts (even if NaN)
    df["func_mean_fd"] = pd.to_numeric(df.get("FUNC_MEAN_FD", np.nan), errors="coerce")

    # Age bins
    df["AGE_GROUP"] = pd.cut(
        df["AGE_AT_SCAN"],
        bins=BINS,
        labels=LABELS,
        right=RIGHT,
        include_lowest=True
    )

    # Deduplicate by subject:
    # if multiple rows exist (longitudinal), keep earliest AGE_AT_SCAN
    df = df.sort_values(["SUB_ID", "AGE_AT_SCAN"], ascending=[True, True]) \
           .drop_duplicates(subset=["SUB_ID"], keep="first") \
           .reset_index(drop=True)

    return df


def write_outputs(df: pd.DataFrame):
    COMBINED_OUT.parent.mkdir(parents=True, exist_ok=True)
    FEMALE_OUT.parent.mkdir(parents=True, exist_ok=True)
    MALE_OUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(COMBINED_OUT, index=False)
    print(f"[SAVED] {COMBINED_OUT}")

    female = df[df["sex"] == "female"].copy()
    male   = df[df["sex"] == "male"].copy()

    female.to_csv(FEMALE_OUT, index=False)
    male.to_csv(MALE_OUT, index=False)

    print(f"[SAVED] {FEMALE_OUT}  (N={len(female)})")
    print(f"[SAVED] {MALE_OUT}    (N={len(male)})")

    # Quick counts
    print("\n[COUNTS] Sex:")
    print(df["sex"].value_counts(dropna=False))

    print("\n[COUNTS] Sex x DX_GROUP:")
    print(pd.crosstab(df["sex"], df["DX_GROUP"]))

    print("\n[COUNTS] Sex x AGE_GROUP:")
    print(pd.crosstab(df["sex"], df["AGE_GROUP"]))

    print("\n[COUNTS] Dataset:")
    if "DATASET" in df.columns:
        print(df["DATASET"].value_counts(dropna=False))

    print("\n[COUNTS] Dataset x Sex:")
    if "DATASET" in df.columns:
        print(pd.crosstab(df["DATASET"], df["sex"]))


def main():
    df1 = load_all_pheno_csvs(PHENO_DIR1, "ABIDE1")
    df2 = load_all_pheno_csvs(PHENO_DIR2, "ABIDE2")
    df = pd.concat([df1, df2], ignore_index=True)

    df = standardize(df)

    print(f"[INFO] Unique subjects: {df['SUB_ID'].nunique()}")
    print(f"[INFO] Unique sites   : {df['SITE_ID'].nunique()}")

    write_outputs(df)
    print("\n[DONE] ABIDE12 metadata rebuilt from phenotypes.")


if __name__ == "__main__":
    main()