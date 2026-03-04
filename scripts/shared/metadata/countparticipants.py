import re
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

DATASET = "ABIDE2"  


CONN_DIR = ROOT / "outputs" / "cc200" / "abide2" / "subjects" / "npy_real"


PHENO_DIR = ROOT / "phenotypes" / "ABIDE2"
PHENO_FILES = [
    PHENO_DIR / "abide2_composite_pheno.csv",
    PHENO_DIR / "abide2_composite_pheno_longitudinal.csv",
]


BINS   = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]

SEX_MAP = {1: "Male", 2: "Female"}
DX_MAP  = {1: "ASD",  2: "Control"}

FD_LABEL = "fd-0.2"  # label for output table


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Try a few encodings so weird bytes don't crash pandas."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def pick_first_existing(cols, candidates):
    """Return the first candidate column name that exists in cols, else None."""
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None


def guess_age_column(cols):
    """
    ABIDE2 files sometimes don't use AGE_AT_SCAN.
    We try common alternatives, then fall back to 'AGE' substrings.
    """
    cols = list(cols)

    # common / likely names (uppercased)
    candidates = [
        "AGE_AT_SCAN",
        "AGE_AT_SCAN_YRS",
        "AGE_AT_SCAN_YEARS",
        "AGE_AT_SCAN_YEAR",
        "AGE_AT_SCAN_MONTHS",
        "AGE",
        "AGE_YRS",
        "AGE_YEARS",
        "AGE_AT_MRI",
        "AGE_AT_MRI_YRS",
        "AGE_AT_MRI_YEARS",
        "AGE_AT_SCAN_1",
        "AGE_AT_SCAN_2",
    ]
    hit = pick_first_existing(cols, candidates)
    if hit:
        return hit

    # heuristic: anything with both AGE and SCAN/MRI
    for c in cols:
        cl = c.upper()
        if "AGE" in cl and ("SCAN" in cl or "MRI" in cl):
            return c

    # fallback: any column containing AGE
    for c in cols:
        if "AGE" in c.upper():
            return c

    return None


def standardize_pheno(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]

    # Required-ish columns (SUB_ID / SEX / DX_GROUP) are usually present.
    # AGE column may be different; we detect it.
    sub_col = pick_first_existing(df.columns, ["SUB_ID", "SUBID", "SUBJECT", "SUBJECT_ID"])
    sex_col = pick_first_existing(df.columns, ["SEX", "GENDER"])
    dx_col  = pick_first_existing(df.columns, ["DX_GROUP", "DX", "DIAGNOSIS", "DXGROUP"])
    age_col = guess_age_column(df.columns)

    missing = [x for x in [sub_col, sex_col, dx_col, age_col] if x is None]
    if missing:
        # If this file isn't the right kind of phenotype table, skip it gracefully
        raise RuntimeError(
            f"Skipping {source_name}: could not find required columns. "
            f"Found SUB={sub_col}, SEX={sex_col}, DX={dx_col}, AGE={age_col}."
        )

    out = pd.DataFrame({
        "SUB_ID": df[sub_col],
        "SEX": df[sex_col],
        "DX_GROUP": df[dx_col],
        "AGE_AT_SCAN": df[age_col],  # normalize into this standard name
    })

    # Optional: SITE_ID if present
    site_col = pick_first_existing(df.columns, ["SITE_ID", "SITE"])
    if site_col:
        out["SITE_ID"] = df[site_col]
    else:
        out["SITE_ID"] = pd.NA

    out["source_file"] = source_name
    return out


def load_abide2_phenotypes(files: list[Path]) -> pd.DataFrame:
    dfs = []
    loaded = []
    for fp in files:
        if not fp.exists():
            continue
        raw = read_csv_robust(fp)
        try:
            ph = standardize_pheno(raw, fp.name)
        except RuntimeError:
            continue
        dfs.append(ph)
        loaded.append(fp.name)

    if not dfs:
        raise FileNotFoundError(
            f"No usable phenotype tables found. Looked for: {[str(f) for f in files]}"
        )

    ph = pd.concat(dfs, ignore_index=True)

    # Normalize types and drop rows missing critical info
    for col in ["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]:
        ph[col] = pd.to_numeric(ph[col], errors="coerce")
    ph = ph.dropna(subset=["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]).copy()

    ph["SUB_ID"] = ph["SUB_ID"].astype(int)
    ph["SEX"] = ph["SEX"].astype(int)
    ph["DX_GROUP"] = ph["DX_GROUP"].astype(int)

    
    ph = ph.drop_duplicates(subset=["SUB_ID"], keep="first")

    # Bin ages
    ph["age_group"] = pd.cut(
        ph["AGE_AT_SCAN"],
        bins=BINS,
        labels=LABELS,
        right=False,
        include_lowest=True,
    )

    ph["sex_name"] = ph["SEX"].map(SEX_MAP).fillna(ph["SEX"].astype(str))
    ph["dx_name"] = ph["DX_GROUP"].map(DX_MAP).fillna(ph["DX_GROUP"].astype(str))
    ph["group_label"] = ph["sex_name"] + "_" + ph["dx_name"]

    print(f"[INFO] Phenotype sources used: {loaded}")
    return ph


def collect_subject_ids_and_run_counts(conn_dir: Path):
    files = list(conn_dir.glob("*.npy"))  # your npy_real is flat (no subfolders)
    rx_sub = re.compile(r"sub-(\d+)")
    rx_run = re.compile(r"run-(\d+)")

    ids = set()
    run_counts = {}

    for p in files:
        m = rx_sub.search(p.name)
        if m:
            ids.add(int(m.group(1)))
        r = rx_run.search(p.name)
        if r:
            key = f"run-{r.group(1)}"
            run_counts[key] = run_counts.get(key, 0) + 1

    return files, ids, run_counts


def counts_table(ph: pd.DataFrame, ids: set[int], fd_label: str) -> pd.DataFrame:
    sub = ph[ph["SUB_ID"].isin(ids)].copy()
    sub = sub.dropna(subset=["age_group"])

    g = (
        sub.groupby(["age_group", "group_label"])["SUB_ID"]
        .nunique()
        .reset_index(name="n_subjects")
    )

    wide = (
        g.pivot(index="age_group", columns="group_label", values="n_subjects")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    expected_cols = ["Female_ASD", "Female_Control", "Male_ASD", "Male_Control"]
    for col in expected_cols:
        if col not in wide.columns:
            wide[col] = 0

    wide["Total"] = wide[expected_cols].sum(axis=1)
    wide.insert(0, "fd_threshold", fd_label)

    wide["age_group"] = pd.Categorical(wide["age_group"], categories=LABELS, ordered=True)
    wide = wide.sort_values("age_group").reset_index(drop=True)

    col_order = ["fd_threshold", "age_group"] + expected_cols + ["Total"]
    return wide[col_order]


def main():
    print(f"[INFO] Dataset={DATASET}")
    print(f"[INFO] Connectomes dir: {CONN_DIR}")
    print(f"[INFO] Phenotypes dir:  {PHENO_DIR}")
    print(f"[INFO] Will consider phenotype files: {[p.name for p in PHENO_FILES]}")

    ph = load_abide2_phenotypes(PHENO_FILES)
    print(f"[INFO] Loaded phenotypes: {len(ph)} unique subjects")

    if not CONN_DIR.exists():
        raise FileNotFoundError(f"Connectome directory not found: {CONN_DIR}")

    files, ids, run_counts = collect_subject_ids_and_run_counts(CONN_DIR)
    print(f"[INFO] Total .npy files found = {len(files)}")
    print(f"[INFO] Unique subjects with connectomes = {len(ids)}")

    if run_counts:
        print("[INFO] Run breakdown (file counts):")
        for k in sorted(run_counts.keys()):
            print(f"  {k}: {run_counts[k]}")

    ph_ids = set(ph["SUB_ID"].tolist())
    missing_pheno = sorted([i for i in ids if i not in ph_ids])
    if missing_pheno:
        print(f"[WARN] {len(missing_pheno)} subjects have connectomes but no phenotype row (first 20): {missing_pheno[:20]}")

    tab = counts_table(ph, ids, FD_LABEL)
    print(f"\n=== Sex and Diagnosis counts by age bin ({DATASET}, {FD_LABEL}) ===")
    print(tab.to_string(index=False) if len(tab) else tab)

    usable_n = int(tab["Total"].sum()) if len(tab) else 0
    print(f"\n[INFO] Usable subjects (has phenotype + age bin) = {usable_n}")

    out_dir = ROOT / "results" / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"sex_dx_by_age_bins_{DATASET.lower()}.csv"
    tab.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    main()