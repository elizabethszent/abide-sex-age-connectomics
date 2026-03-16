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

# current combined files
ABIDE1_COMBINED = PHENO_DIR / "ABIDE_phenotypes_combined.csv"
ABIDE2_COMBINED = PHENO_DIR / "ABIDE2_phenotypes_combined.csv"

# original phenotype sources
ABIDE1_ORIG_DIR = PHENO_DIR / "ABIDE1"
ABIDE2_ORIG_FILES = [
    PHENO_DIR / "ABIDE2" / "abide2_composite_pheno.csv",
    PHENO_DIR / "ABIDE2" / "abide2_composite_pheno_longitudinal.csv",
]

CONNECTOME_DIRS = {
    "fd_0p2": ROOT / "results" / "connectomes" / "ABIDE12" / "ABIDE12" / "fd_0p2",
    "fd_0p3": ROOT / "results" / "connectomes" / "ABIDE12" / "ABIDE12" / "fd_0p3",
}

FEMALE_DIR = ROOT / "data" / "female"
MALE_DIR = ROOT / "data" / "male"
COMBINED_OUT = ROOT / "data" / "metadata" / "ABIDE12_phenotypes_combined.csv"

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


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).upper().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.upper().strip()
        if key in cols:
            return cols[key]
    return None


def first_nonnull(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def derive_right_handed(series: pd.Series) -> pd.Series:
    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        return pd.Series(np.where(x.notna(), (x > 0).astype(float), np.nan), index=s.index)

    s = s.astype(str).str.strip().str.upper()
    out = pd.Series(np.nan, index=s.index, dtype=float)

    right_vals = {"R", "RIGHT", "RIGHTHANDED", "RIGHT-HANDED", "RH"}
    non_right_vals = {
        "L", "LEFT", "LEFTHANDED", "LEFT-HANDED", "LH",
        "A", "AMB", "AMBI", "AMBIDEXTROUS", "MIXED", "M", "BOTH"
    }

    out[s.isin(right_vals)] = 1.0
    out[s.isin(non_right_vals)] = 0.0
    return out


def load_base_combined(fp: Path, dataset_name: str) -> pd.DataFrame:
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

    func_fd_col = first_existing_column(df, ["FUNC_MEAN_FD", "func_mean_fd"])
    if func_fd_col is not None:
        keep.append(func_fd_col)

    out = df[keep].copy()
    out["PHENO_FILE"] = fp.name
    out["DATASET"] = dataset_name

    if func_fd_col is not None:
        out["func_mean_fd"] = pd.to_numeric(out[func_fd_col], errors="coerce")
        out = out.drop(columns=[func_fd_col], errors="ignore")
    else:
        out["func_mean_fd"] = np.nan

    return out


def load_current_combined_phenos() -> pd.DataFrame:
    frames = [
        load_base_combined(ABIDE1_COMBINED, "ABIDE1"),
        load_base_combined(ABIDE2_COMBINED, "ABIDE2"),
    ]
    out = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded base combined phenotype rows: {len(out)}")
    return out


def extract_enrichment_from_df(df: pd.DataFrame, dataset_name: str, source_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    sub_col = first_existing_column(df, ["SUB_ID"])
    if sub_col is None:
        raise ValueError(f"{source_name} missing SUB_ID")

    fiq_col = first_existing_column(df, ["FIQ", "FULL_IQ", "IQ"])
    handed_col = first_existing_column(
        df,
        ["HANDEDNESS_CATEGORY", "HANDEDNESS", "HANDEDNESS_SCORES", "EHI_SCORES"]
    )

    out = pd.DataFrame()
    out["SUB_ID"] = pd.to_numeric(df[sub_col], errors="coerce")
    out["DATASET"] = dataset_name
    out["ENRICH_SOURCE"] = source_name

    if fiq_col is not None:
        out["FIQ"] = pd.to_numeric(df[fiq_col], errors="coerce")
    else:
        out["FIQ"] = np.nan

    if handed_col is not None:
        out["RIGHT_HANDED"] = derive_right_handed(df[handed_col])
    else:
        out["RIGHT_HANDED"] = np.nan

    out = out.dropna(subset=["SUB_ID"]).copy()
    out["SUB_ID"] = out["SUB_ID"].astype(int)

    return out


def load_abide1_enrichment() -> pd.DataFrame:
    if not ABIDE1_ORIG_DIR.exists():
        raise FileNotFoundError(f"Missing ABIDE1 phenotype dir: {ABIDE1_ORIG_DIR}")

    files = sorted(ABIDE1_ORIG_DIR.glob("phenotypic_*.csv"))
    if not files:
        raise FileNotFoundError(f"No ABIDE1 phenotypic_*.csv files found in {ABIDE1_ORIG_DIR}")

    frames = []
    for fp in files:
        try:
            df = read_csv_flexible(fp)
            frames.append(extract_enrichment_from_df(df, "ABIDE1", fp.name))
        except Exception as e:
            print(f"[WARN] Could not enrich from {fp.name}: {type(e).__name__}: {e}")

    if not frames:
        raise RuntimeError("No usable ABIDE1 enrichment tables found")

    out = pd.concat(frames, ignore_index=True)
    out = (
        out.sort_values(["SUB_ID", "ENRICH_SOURCE"])
        .groupby(["DATASET", "SUB_ID"], as_index=False)
        .agg(
            FIQ=("FIQ", first_nonnull),
            RIGHT_HANDED=("RIGHT_HANDED", first_nonnull),
        )
    )
    return out


def load_abide2_enrichment() -> pd.DataFrame:
    frames = []
    for fp in ABIDE2_ORIG_FILES:
        if not fp.exists():
            continue
        try:
            df = read_csv_flexible(fp)
            frames.append(extract_enrichment_from_df(df, "ABIDE2", fp.name))
        except Exception as e:
            print(f"[WARN] Could not enrich from {fp.name}: {type(e).__name__}: {e}")

    if not frames:
        print("[WARN] No ABIDE2 original enrichment files found; ABIDE2 FIQ/RIGHT_HANDED will be NaN")
        return pd.DataFrame(columns=["DATASET", "SUB_ID", "FIQ", "RIGHT_HANDED"])

    out = pd.concat(frames, ignore_index=True)
    out = (
        out.sort_values(["SUB_ID", "ENRICH_SOURCE"])
        .groupby(["DATASET", "SUB_ID"], as_index=False)
        .agg(
            FIQ=("FIQ", first_nonnull),
            RIGHT_HANDED=("RIGHT_HANDED", first_nonnull),
        )
    )
    return out


def load_enrichment_table() -> pd.DataFrame:
    a1 = load_abide1_enrichment()
    a2 = load_abide2_enrichment()
    out = pd.concat([a1, a2], ignore_index=True)
    print(f"[INFO] Loaded enrichment rows: {len(out)}")
    return out


def merge_enrichment(base_df: pd.DataFrame, enrich_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.merge(enrich_df, on=["DATASET", "SUB_ID"], how="left")

    if "FIQ" not in df.columns:
        df["FIQ"] = np.nan
    if "RIGHT_HANDED" not in df.columns:
        df["RIGHT_HANDED"] = np.nan

    return df


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

    df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce")
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
    df["func_mean_fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
    df["FIQ"] = pd.to_numeric(df["FIQ"], errors="coerce")
    df["RIGHT_HANDED"] = pd.to_numeric(df["RIGHT_HANDED"], errors="coerce")

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]).copy()

    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)

    df["sex"] = df["SEX"].map({1: "male", 2: "female"}).fillna("unknown")

    df["SUB_ID_Z7"] = df["SUB_ID"].astype(str).str.zfill(7)
    df["FILE_ID_NUM"] = df["SUB_ID_Z7"]
    df["BIDS_ID"] = "sub-" + df["SUB_ID_Z7"]
    df["FILE_ID"] = df["FILE_ID_NUM"]

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

        print(f"\n[COUNTS] {cutoff_name}: RIGHT_HANDED")
        print(df_with_conn["RIGHT_HANDED"].value_counts(dropna=False))

        print(f"\n[SUMMARY] {cutoff_name}: FIQ")
        print(df_with_conn["FIQ"].describe())


def main():
    base_df = load_current_combined_phenos()
    enrich_df = load_enrichment_table()
    df = merge_enrichment(base_df, enrich_df)
    df = standardize(df)

    print(f"[INFO] Unique subjects: {df['SUB_ID'].nunique()}")
    print(f"[INFO] Unique sites   : {df['SITE_ID'].nunique()}")
    print(f"[INFO] FIQ non-null subjects: {df['FIQ'].notna().sum()}")
    print(f"[INFO] RIGHT_HANDED non-null subjects: {df['RIGHT_HANDED'].notna().sum()}")

    write_outputs(df)
    print("\n[DONE] Metadata rebuilt and enriched with FIQ and RIGHT_HANDED.")


if __name__ == "__main__":
    main()