# scripts/shared/qualitycheck.py
import os
import re
from glob import glob
import pandas as pd

# ----------------------------
# CONFIG: edit if needed
# ----------------------------
PHENO_CSV = r"data\Phenotypic_V1_0b_preprocessed1.csv"
CONNECTOME_DIR = r"data\connectomes\cpac\nofilt_noglobal\cc200_z"  # where your .npy files live

# Motion thresholds to compare
FD_CUTOFFS = [0.2, 0.3, 0.5]

# Age bins you’ve used before (edit if you want)
AGE_BINS = [0, 10, 13, 18, 30, 45, 120]
AGE_LABELS = ["child(0-9)","preteen(10-12)", "teen(13-18)", "young_adult(18-30)", "adult(30-45)", "older(45+)"]

# ABIDE conventions (commonly used in PCP phenofile)
DX_MAP = {1: "ASD", 2: "Control"}
SEX_MAP = {1: "Male", 2: "Female"}

# ----------------------------
# Helpers
# ----------------------------
def extract_id7_from_filename(fname: str) -> str | None:
    """Extract 7-digit ID from filenames like Caltech_0051456.npy"""
    m = re.search(r"(\d{7})", fname)
    return m.group(1) if m else None

def pretty_counts(df: pd.DataFrame, rows, cols=None, title=None):
    if title:
        print("\n" + title)
    if cols is None:
        print(df.groupby(rows).size().sort_values(ascending=False))
    else:
        print(df.pivot_table(index=rows, columns=cols, values="FILE_ID", aggfunc="count", fill_value=0))

def cutoff_table(df: pd.DataFrame, label: str):
    """Print keep/drop counts for each FD cutoff."""
    print(f"\n=== Motion cutoff impact ({label}) ===")
    for c in FD_CUTOFFS:
        keep = df["func_mean_fd"].notna() & (df["func_mean_fd"] < c)
        drop = df["func_mean_fd"].notna() & (df["func_mean_fd"] >= c)
        miss = df["func_mean_fd"].isna()
        print(f"FD < {c}: keep={keep.sum():4d}  drop={drop.sum():4d}  missing_fd={miss.sum():2d}")

# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Load phenofile
    pheno = pd.read_csv(PHENO_CSV)
    if "FILE_ID" not in pheno.columns:
        raise ValueError("Phenofile must contain FILE_ID column.")

    # Normalize ID7 for joining
    pheno["ID7"] = pheno["FILE_ID"].astype(str).str.extract(r"(\d{7})")

    # Map readable labels if columns exist
    if "DX_GROUP" in pheno.columns:
        pheno["DX_LABEL"] = pheno["DX_GROUP"].map(DX_MAP).fillna(pheno["DX_GROUP"].astype(str))
    else:
        pheno["DX_LABEL"] = "Unknown"

    if "SEX" in pheno.columns:
        pheno["SEX_LABEL"] = pheno["SEX"].map(SEX_MAP).fillna(pheno["SEX"].astype(str))
    else:
        pheno["SEX_LABEL"] = "Unknown"

    # Age bins (optional)
    if "AGE_AT_SCAN" in pheno.columns:
        pheno["AGE_BIN"] = pd.cut(
            pheno["AGE_AT_SCAN"],
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=True,
            include_lowest=True,
        )
    else:
        pheno["AGE_BIN"] = "Unknown"

    # 2) List connectome files on disk
    files = glob(os.path.join(CONNECTOME_DIR, "*.npy"))
    # There’s also a _backup_bad_shapes folder; ignore it automatically because we only take *.npy in root.
    print("Connectome files:", len(files))
    print("Example files:")
    for f in files[:5]:
        print(" ", f)

    # Extract ID7 from connectome filenames
    found_id7 = set()
    bad = 0
    for f in files:
        id7 = extract_id7_from_filename(os.path.basename(f))
        if id7:
            found_id7.add(id7)
        else:
            bad += 1

    print("Unique numeric IDs (7 digits):", len(found_id7))
    if bad:
        print("Warning: files without detectable 7-digit ID:", bad)

    # 3) Join phenofile to “subjects with connectomes”
    have = pheno[pheno["ID7"].isin(found_id7)].copy()
    miss = pheno[~pheno["ID7"].isin(found_id7)].copy()

    print("\nPhenofile total:", len(pheno))
    print("Have connectomes:", len(have))
    print("Missing connectomes:", len(miss))

    # 4) Basic motion summary for what you ACTUALLY use
    print("\nMotion (func_mean_fd) for subjects WITH connectomes:")
    if "func_mean_fd" in have.columns:
        print(have["func_mean_fd"].describe())
        print("Max mean FD among used subjects:", have["func_mean_fd"].max())
    else:
        print("No func_mean_fd column found in phenofile. (Unexpected for PCP QA.)")

    # 5) Missing reasons (if provided)
    if "reason" in miss.columns:
        print("\nTop reasons among missing-connectome subjects:")
        print(miss["reason"].value_counts(dropna=False).head(10))

    # 6) Cutoff impact tables
    if "func_mean_fd" in have.columns:
        cutoff_table(have, "ALL subjects with connectomes")

        # 6a) By sex x diagnosis at each cutoff (key for your “female sample size” concern)
        if {"SEX_LABEL", "DX_LABEL"}.issubset(have.columns):
            for c in FD_CUTOFFS:
                keep = have["func_mean_fd"].notna() & (have["func_mean_fd"] < c)
                subset = have[keep].copy()
                pretty_counts(
                    subset,
                    rows="SEX_LABEL",
                    cols="DX_LABEL",
                    title=f"\nCounts AFTER FD < {c} (SEX x DX) among subjects with connectomes",
                )

        # 6b) By age bin x sex x diagnosis (optional but very useful)
        if {"AGE_BIN", "SEX_LABEL", "DX_LABEL"}.issubset(have.columns):
            for c in FD_CUTOFFS:
                keep = have["func_mean_fd"].notna() & (have["func_mean_fd"] < c)
                subset = have[keep].copy()
                print(f"\nCounts AFTER FD < {c} (AGE_BIN x SEX x DX):")
                print(subset.groupby(["AGE_BIN", "SEX_LABEL", "DX_LABEL"]).size())

    # 7) Write out CSVs you can show your supervisor
    os.makedirs("qc_out", exist_ok=True)

    have.to_csv(r"qc_out\abide_have_connectomes_joined.csv", index=False)
    miss.to_csv(r"qc_out\abide_missing_connectomes.csv", index=False)

    print("\nWrote:")
    print("  qc_out/abide_have_connectomes_joined.csv")
    print("  qc_out/abide_missing_connectomes.csv")
    print("\nNext: open qc_out/abide_have_connectomes_joined.csv and filter/sort by func_mean_fd to see outliers.")

if __name__ == "__main__":
    main()
