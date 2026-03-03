import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
PHENO_DIR = ROOT / "phenotypes"

# Output locations (match what your hubs script expects)
FEMALE_OUT = ROOT / "data" / "female" / "female_metadata_included.csv"
MALE_OUT   = ROOT / "data" / "male"   / "male_metadata_included.csv"
COMBINED_OUT = ROOT / "data" / "metadata" / "ABIDE_phenotypes_combined.csv"

# Your requested bins
BINS   = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
RIGHT  = False  # [0,10), [10,13), [13,18), [18,200)

REQUIRED = {"SITE_ID", "SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN"}

def load_all_pheno_csvs(pheno_dir: Path) -> pd.DataFrame:
    fps = sorted(pheno_dir.rglob("*.csv"))
    if not fps:
        raise FileNotFoundError(f"No CSVs found under {pheno_dir}")

    frames = []
    skipped = 0

    for fp in fps:
        try:
            df = pd.read_csv(fp)
        except Exception:
            skipped += 1
            continue

        df.columns = df.columns.str.strip()
        if not REQUIRED.issubset(df.columns):
            skipped += 1
            continue

        keep = list(REQUIRED)
        if "func_mean_fd" in df.columns:
            keep.append("func_mean_fd")

        sub = df[keep].copy()
        sub["PHENO_FILE"] = fp.name
        frames.append(sub)

    if not frames:
        raise RuntimeError(f"No usable phenotypic CSVs found in {pheno_dir} (missing required columns).")

    out = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded phenotype rows: {len(out)} (skipped files: {skipped})")
    return out


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # numeric cleanup
    df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce")
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]).copy()

    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)

    # ABIDE convention: SEX 1=male, 2=female (commonly)
    df["sex"] = df["SEX"].map({1: "male", 2: "female"}).fillna("unknown")

    # Helpful ID variants:
    df["SUB_ID_Z7"] = df["SUB_ID"].astype(str).str.zfill(7)
    df["FILE_ID_NUM"] = df["SUB_ID_Z7"]              # e.g., "0050004"
    df["BIDS_ID"] = "sub-" + df["SUB_ID_Z7"]         # e.g., "sub-0050004"

    # Choose a default FILE_ID:
    # If you used connectomes named "0050004.npy" -> FILE_ID_NUM
    # If you used connectomes named "sub-0050004.npy" -> BIDS_ID
    # Defaulting to FILE_ID_NUM is usually safest for older pipelines.
    df["FILE_ID"] = df["FILE_ID_NUM"]

    # func_mean_fd may not exist in phenos; keep column anyway so older scripts don't crash
    if "func_mean_fd" not in df.columns:
        df["func_mean_fd"] = np.nan

    # Age bins
    df["AGE_GROUP"] = pd.cut(
        df["AGE_AT_SCAN"],
        bins=BINS,
        labels=LABELS,
        right=RIGHT,
        include_lowest=True
    )

    # Drop duplicates by SUB_ID if any (keep first)
    df = df.sort_values(["SITE_ID", "SUB_ID"]).drop_duplicates(subset=["SUB_ID"], keep="first")

    return df


def write_outputs(df: pd.DataFrame):
    COMBINED_OUT.parent.mkdir(parents=True, exist_ok=True)
    FEMALE_OUT.parent.mkdir(parents=True, exist_ok=True)
    MALE_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Save combined
    df.to_csv(COMBINED_OUT, index=False)
    print(f"[SAVED] {COMBINED_OUT}")

    # Split by sex
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


def main():
    df = load_all_pheno_csvs(PHENO_DIR)
    df = standardize(df)

    print(f"[INFO] Unique subjects: {df['SUB_ID'].nunique()}")
    print(f"[INFO] Unique sites   : {df['SITE_ID'].nunique()}")

    write_outputs(df)
    print("\n[DONE] Metadata rebuilt from phenotypes.")


if __name__ == "__main__":
    main()