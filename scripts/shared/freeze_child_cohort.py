# scripts/shared/freeze_child_cohort.py
from pathlib import Path
import pandas as pd
import numpy as np


AGE_MIN = 7.0  
AGE_MAX = 13.0  

ROOT = Path(__file__).resolve().parents[2]  #project root
DATA_DIR = ROOT / "data"

FEMALE_META = DATA_DIR / "female" / "female_metadata_included.csv"
MALE_META   = DATA_DIR / "male"   / "male_metadata_included.csv"

def process_sex(sex_label: str, meta_path: Path, out_dir: Path):
    print(f"\n=== {sex_label.upper()} ===")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)
    #be paranoid about column name whitespace
    df.columns = [c.strip() for c in df.columns]

    required = ["FILE_ID", "DX_GROUP", "AGE_AT_SCAN"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{sex_label}: missing required column '{col}' in {meta_path}")

    #quick sanity
    print(f"Total included (all ages): {len(df)}")
    print("AGE_AT_SCAN range:", float(df["AGE_AT_SCAN"].min()), "→", float(df["AGE_AT_SCAN"].max()))

    #child-only subset
    child = df[(df["AGE_AT_SCAN"] >= AGE_MIN) & (df["AGE_AT_SCAN"] <= AGE_MAX)].copy()

    print(f"Child window: {AGE_MIN}–{AGE_MAX} years")
    print(f"Child cohort size: {len(child)}")

    if len(child) == 0:
        print("WARNING: no subjects in this age window!")
    else:
        #group-wise counts
        grp_counts = child["DX_GROUP"].value_counts().sort_index()
        print("\nCounts by DX_GROUP (1=ASD, 2=Control):")
        print(grp_counts.to_string())

        #optional: basic age summary
        print("\nAge summary (child cohort):")
        print(child["AGE_AT_SCAN"].describe())

    #make output dir
    out_dir.mkdir(parents=True, exist_ok=True)

    #save child metadata + subject list
    child_meta_path = out_dir / "child_metadata_included.csv"
    child_ids_path  = out_dir / "child_subjects_included.txt"

    child.to_csv(child_meta_path, index=False)
    child["FILE_ID"].to_csv(child_ids_path, index=False, header=False)

    print(f"\nSaved child metadata -> {child_meta_path}")
    print(f"Saved child subject IDs -> {child_ids_path}")


if __name__ == "__main__":
    print(f"Using child age window: {AGE_MIN}–{AGE_MAX} years")

    female_out = DATA_DIR / "female"
    male_out   = DATA_DIR / "male"

    process_sex("female", FEMALE_META, female_out)
    process_sex("male",   MALE_META,   male_out)

    print("\nDone.")
