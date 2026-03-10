import re
import numpy as np
import pandas as pd
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "phenotypes").exists():
            return p
    raise FileNotFoundError("Could not find repo root containing 'phenotypes/'.")


ROOT = find_repo_root(Path(__file__).resolve().parent)
PHENO_DIR = ROOT / "phenotypes"

# Explicitly use only the combined phenotype files
ABIDE1_PHENO = PHENO_DIR / "ABIDE_phenotypes_combined.csv"
ABIDE2_PHENO = PHENO_DIR / "ABIDE2_phenotypes_combined.csv"

# Connectome folders for each FD cutoff
CONNECTOME_DIRS = {
    "fd_0p2": ROOT / "results" / "connectomes" / "ABIDE12" / "ABIDE12" / "fd_0p2",
    "fd_0p3": ROOT / "results" / "connectomes" / "ABIDE12" / "ABIDE12" / "fd_0p3",
}

# Output locations
FEMALE_DIR = ROOT / "data" / "female"
MALE_DIR = ROOT / "data" / "male"
COMBINED_OUT = ROOT / "data" / "metadata" / "ABIDE12_phenotypes_combined.csv"

# Requested bins
BINS = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
RIGHT = False

REQUIRED_COLS = ["SITE_ID", "SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN"]
REQUIRED_SET = set(REQUIRED_COLS)


def read_csv_flexible(fp: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(fp, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "read_csv_flexible",
        b"",
        0,
        1,
        f"Could not decode {fp} with utf-8, utf-8-sig, cp1252, or latin1",
    )


def load_one_combined(fp: Path, dataset_name: str) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(f"Missing phenotype file: {fp}")

    df = read_csv_flexible(fp)
    df.columns = [str(c).upper().strip() for c in df.columns]

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        print(f"[WARN] {fp.name} has duplicate columns after normalization: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()].copy()

    if not REQUIRED_SET.issubset(df.columns):
        raise RuntimeError(
            f"{fp.name} is missing required columns: {REQUIRED_SET - set(df.columns)}"
        )

    keep = REQUIRED_COLS.copy()
    if "FUNC_MEAN_FD" in df.columns:
        keep.append("FUNC_MEAN_FD")

    out = df[keep].copy()
    out["PHENO_FILE"] = fp.name
    out["DATASET"] = dataset_name
    return out


def load_combined_phenos() -> pd.DataFrame:
    frames = [
        load_one_combined(ABIDE1_PHENO, "ABIDE1"),
        load_one_combined(ABIDE2_PHENO, "ABIDE2"),
    ]
    out = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded phenotype rows: {len(out)}")
    return out


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "FUNC_MEAN_FD" in df.columns and "func_mean_fd" not in df.columns:
        df = df.rename(columns={"FUNC_MEAN_FD": "func_mean_fd"})

    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

    df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce")
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]).copy()

    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)

    df["sex"] = df["SEX"].map({1: "male", 2: "female"}).fillna("unknown")

    df["SUB_ID_Z7"] = df["SUB_ID"].astype(str).str.zfill(7)
    df["FILE_ID_NUM"] = df["SUB_ID_Z7"]
    df["BIDS_ID"] = "sub-" + df["SUB_ID_Z7"]
    df["FILE_ID"] = df["FILE_ID_NUM"]

    if "func_mean_fd" not in df.columns:
        df["func_mean_fd"] = np.nan

    df["AGE_GROUP"] = pd.cut(
        df["AGE_AT_SCAN"],
        bins=BINS,
        labels=LABELS,
        right=RIGHT,
        include_lowest=True,
    )

    df = df.sort_values(["DATASET", "SITE_ID", "SUB_ID"]).drop_duplicates(
        subset=["SUB_ID"], keep="first"
    )

    return df


def subject_ids_with_connectomes(conn_root: Path) -> set[int]:
    if not conn_root.exists():
        print(f"[WARN] Connectome directory does not exist: {conn_root}")
        return set()

    files = list(conn_root.rglob("*.npy"))
    if not files:
        print(f"[WARN] No .npy files found under {conn_root}")
        return set()

    ids = set()
    rx_sub = re.compile(r"sub-(\d+)")
    rx_digits = re.compile(r"^(\d{5,})$")

    for p in files:
        stem = p.stem

        m = rx_sub.search(stem)
        if m:
            ids.add(int(m.group(1)))
            continue

        m = rx_digits.match(stem)
        if m:
            ids.add(int(m.group(1)))
            continue

    print(f"[INFO] {conn_root.name}: found {len(files)} connectome files")
    print(f"[INFO] {conn_root.name}: found {len(ids)} unique subjects with connectomes")
    return ids


def write_outputs(df: pd.DataFrame) -> None:
    COMBINED_OUT.parent.mkdir(parents=True, exist_ok=True)
    FEMALE_DIR.mkdir(parents=True, exist_ok=True)
    MALE_DIR.mkdir(parents=True, exist_ok=True)

    # Save full combined phenotype table
    df.to_csv(COMBINED_OUT, index=False)
    print(f"[SAVED] {COMBINED_OUT}")

    for cutoff_name, conn_dir in CONNECTOME_DIRS.items():
        conn_ids = subject_ids_with_connectomes(conn_dir)
        df_with_conn = df[df["SUB_ID"].isin(conn_ids)].copy()

        female = df_with_conn[df_with_conn["sex"] == "female"].copy()
        male = df_with_conn[df_with_conn["sex"] == "male"].copy()

        female_out = FEMALE_DIR / f"female_metadata_included_{cutoff_name}.csv"
        male_out = MALE_DIR / f"male_metadata_included_{cutoff_name}.csv"
        combined_out = COMBINED_OUT.parent / f"ABIDE12_phenotypes_combined_{cutoff_name}.csv"

        female.to_csv(female_out, index=False)
        male.to_csv(male_out, index=False)
        df_with_conn.to_csv(combined_out, index=False)

        print(f"\n[SAVED] {female_out}  (N={len(female)})")
        print(f"[SAVED] {male_out}    (N={len(male)})")
        print(f"[SAVED] {combined_out} (N={len(df_with_conn)})")

        print(f"\n[COUNTS] {cutoff_name}: Sex")
        print(df_with_conn["sex"].value_counts(dropna=False))

        print(f"\n[COUNTS] {cutoff_name}: Sex x DX_GROUP")
        print(pd.crosstab(df_with_conn["sex"], df_with_conn["DX_GROUP"]))

        print(f"\n[COUNTS] {cutoff_name}: Sex x AGE_GROUP")
        print(pd.crosstab(df_with_conn["sex"], df_with_conn["AGE_GROUP"]))


def main():
    df = load_combined_phenos()
    df = standardize(df)

    print(f"[INFO] Unique subjects: {df['SUB_ID'].nunique()}")
    print(f"[INFO] Unique sites   : {df['SITE_ID'].nunique()}")

    write_outputs(df)
    print("\n[DONE] Metadata rebuilt from ABIDE1 + ABIDE2 combined phenotype files.")


if __name__ == "__main__":
    main()