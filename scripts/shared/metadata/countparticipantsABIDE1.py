import re
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\eliza\Connectomics\TERMProject\abide-sex-age-connectomics")

# ABIDE1 paths
ABIDE1_CONN_DIR  = ROOT / "connectomes" / "CC200" / "ABIDE1" / "FDpersubject2"
ABIDE1_PHENO_DIR = ROOT / "phenotypes" / "ABIDE1"

# ABIDE2 paths
ABIDE2_CONN_DIR  = ROOT / "outputs" / "cc200" / "abide2" / "subjects" / "npy_real"
ABIDE2_PHENO_DIR = ROOT / "phenotypes" / "ABIDE2"

# Output
OUT_DIR = ROOT / "results" / "qc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Age bins
BINS   = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]

# ABIDE conventions
SEX_MAP = {1: "Male", 2: "Female"}
DX_MAP  = {1: "ASD",  2: "Control"}

RX_SUB = re.compile(r"sub-(\d+)")
RX_RUN = re.compile(r"run-(\d+)")


def read_csv_robust(fp: Path) -> pd.DataFrame:
    """Try a few encodings so ABIDE2 composite/longitudinal files don't crash."""
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(fp, encoding="utf-8", encoding_errors="replace")


def canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip whitespace, uppercase, collapse spaces."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().upper()) for c in df.columns]
    return df


def load_phenotypes_any(pheno_dir: Path) -> pd.DataFrame:
    """
    Load *any* CSVs in the folder that contain at least:
      SUB_ID, SEX, DX_GROUP, AGE_AT_SCAN (or common variants)
    We also keep SITE_ID if present.

    For ABIDE2 composite: AGE_AT_SCAN sometimes has trailing spaces -> fixed by canon_cols().
    For longitudinal: multiple rows per SUB_ID -> keep the row with minimum AGE_AT_SCAN.
    """
    if not pheno_dir.exists():
        raise FileNotFoundError(f"Phenotype dir not found: {pheno_dir}")

    csvs = sorted(pheno_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in: {pheno_dir}")

    frames = []
    skipped = 0

    for fp in csvs:
        try:
            df = read_csv_robust(fp)
            df = canon_cols(df)
        except Exception:
            skipped += 1
            continue

        # accept a few AGE column name variants just in case
        age_col = None
        for candidate in ("AGE_AT_SCAN", "AGE_AT_SCAN_YEARS", "AGE", "AGE_YRS"):
            if candidate in df.columns:
                age_col = candidate
                break

        need = {"SUB_ID", "SEX", "DX_GROUP"}
        if (age_col is None) or (not need.issubset(df.columns)):
            skipped += 1
            continue

        keep = [c for c in ("SITE_ID", "SUB_ID", "SEX", "DX_GROUP", age_col) if c in df.columns]
        sub = df[keep].copy()
        if age_col != "AGE_AT_SCAN":
            sub = sub.rename(columns={age_col: "AGE_AT_SCAN"})
        sub["source_file"] = fp.name
        frames.append(sub)

    if not frames:
        raise RuntimeError(
            f"No usable phenotype CSVs in {pheno_dir}. "
            f"Need columns like SUB_ID, SEX, DX_GROUP, AGE_AT_SCAN."
        )

    ph = pd.concat(frames, ignore_index=True)

    # numeric cleanup
    ph["SUB_ID"] = pd.to_numeric(ph["SUB_ID"], errors="coerce")
    ph["SEX"] = pd.to_numeric(ph["SEX"], errors="coerce")
    ph["DX_GROUP"] = pd.to_numeric(ph["DX_GROUP"], errors="coerce")
    ph["AGE_AT_SCAN"] = pd.to_numeric(ph["AGE_AT_SCAN"], errors="coerce")

    # drop rows missing core fields
    ph = ph.dropna(subset=["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]).copy()
    ph["SUB_ID"] = ph["SUB_ID"].astype(int)
    ph["SEX"] = ph["SEX"].astype(int)
    ph["DX_GROUP"] = ph["DX_GROUP"].astype(int)

    # If longitudinal has multiple rows per subject, keep the earliest scan age (most consistent “baseline”)
    ph = ph.sort_values(["SUB_ID", "AGE_AT_SCAN"], ascending=[True, True]).drop_duplicates(
        subset=["SUB_ID"], keep="first"
    )

    # age bins
    ph["age_group"] = pd.cut(
        ph["AGE_AT_SCAN"],
        bins=BINS,
        labels=LABELS,
        right=False,
        include_lowest=True,
    )

    # labels
    ph["sex_name"] = ph["SEX"].map(SEX_MAP).fillna(ph["SEX"].astype(str))
    ph["dx_name"] = ph["DX_GROUP"].map(DX_MAP).fillna(ph["DX_GROUP"].astype(str))
    ph["group_label"] = ph["sex_name"] + "_" + ph["dx_name"]

    print(f"[INFO] Loaded phenotypes from {pheno_dir} -> {len(ph)} unique subjects (skipped files={skipped})")
    return ph


def scan_connectomes(conn_dir: Path):
    """
    Scan .npy connectomes, return:
      - total file count
      - unique subject IDs set
      - run breakdown dict
    """
    if not conn_dir.exists():
        raise FileNotFoundError(f"Connectome dir not found: {conn_dir}")

    files = list(conn_dir.rglob("*.npy"))

    ids = set()
    run_counts = {}

    for p in files:
        ms = RX_SUB.search(p.name)
        if ms:
            ids.add(int(ms.group(1)))

        mr = RX_RUN.search(p.name)
        if mr:
            k = f"run-{mr.group(1)}"
            run_counts[k] = run_counts.get(k, 0) + 1

    return len(files), ids, run_counts


def counts_table(ph: pd.DataFrame, ids: set[int], label: str) -> pd.DataFrame:
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
    wide.insert(0, "dataset", label)

    wide["age_group"] = pd.Categorical(wide["age_group"], categories=LABELS, ordered=True)
    wide = wide.sort_values("age_group").reset_index(drop=True)

    col_order = ["dataset", "age_group"] + expected_cols + ["Total"]
    return wide[col_order]


def run_one(dataset_name: str, conn_dir: Path, pheno_dir: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"[RUN] {dataset_name}")
    print(f"[INFO] Connectomes: {conn_dir}")
    print(f"[INFO] Phenotypes : {pheno_dir}")

    ph = load_phenotypes_any(pheno_dir)

    n_files, ids, run_counts = scan_connectomes(conn_dir)
    print(f"[INFO] Total .npy files found = {n_files}")
    print(f"[INFO] Unique subjects with connectomes = {len(ids)}")

    if run_counts:
        print("[INFO] Run breakdown (file counts):")
        for k in sorted(run_counts.keys(), key=lambda s: int(s.split("-")[1])):
            print(f"  {k}: {run_counts[k]}")

    ph_ids = set(ph["SUB_ID"].tolist())
    missing_pheno = sorted([i for i in ids if i not in ph_ids])
    if missing_pheno:
        print(f"[WARN] {len(missing_pheno)} subjects have connectomes but no phenotype row (first 10): {missing_pheno[:10]}")

    tab = counts_table(ph, ids, dataset_name)
    print(f"\n=== Sex and Diagnosis counts by age bin ({dataset_name}) ===")
    print(tab.to_string(index=False) if len(tab) else tab)

    out_csv = OUT_DIR / f"sex_dx_by_age_bins_{dataset_name.lower()}.csv"
    tab.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}")

    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="all",
        choices=["abide1", "abide2", "abide12", "all"],
        help="Which dataset(s) to compute.",
    )
    args = ap.parse_args()

    tabs = []

    if args.dataset in ("abide1", "all"):
        tabs.append(run_one("ABIDE1", ABIDE1_CONN_DIR, ABIDE1_PHENO_DIR))

    if args.dataset in ("abide2", "all"):
        tabs.append(run_one("ABIDE2", ABIDE2_CONN_DIR, ABIDE2_PHENO_DIR))

    if args.dataset in ("abide12", "all"):
        # Combined: load both phenos + union connectome IDs
        print("\n" + "=" * 80)
        print("[RUN] ABIDE12 (ABIDE1 + ABIDE2 combined)")

        ph1 = load_phenotypes_any(ABIDE1_PHENO_DIR)
        ph2 = load_phenotypes_any(ABIDE2_PHENO_DIR)
        ph = pd.concat([ph1, ph2], ignore_index=True)

        _, ids1, _ = scan_connectomes(ABIDE1_CONN_DIR)
        _, ids2, _ = scan_connectomes(ABIDE2_CONN_DIR)
        ids = set(ids1) | set(ids2)

        print(f"[INFO] Combined phenotype subjects = {ph['SUB_ID'].nunique()}")
        print(f"[INFO] Combined unique subjects with connectomes = {len(ids)}")

        tab = counts_table(ph, ids, "ABIDE12")
        print(f"\n=== Sex and Diagnosis counts by age bin (ABIDE12) ===")
        print(tab.to_string(index=False) if len(tab) else tab)

        out_csv = OUT_DIR / "sex_dx_by_age_bins_abide12.csv"
        tab.to_csv(out_csv, index=False)
        print(f"\n[SAVED] {out_csv}")
        tabs.append(tab)

    # Optional: also write a single combined CSV of whichever ones ran
    if tabs:
        final = pd.concat(tabs, ignore_index=True)
        out_csv = OUT_DIR / "sex_dx_by_age_bins_ALL_REQUESTED.csv"
        final.to_csv(out_csv, index=False)
        print(f"\n[SAVED] {out_csv}")


if __name__ == "__main__":
    main()