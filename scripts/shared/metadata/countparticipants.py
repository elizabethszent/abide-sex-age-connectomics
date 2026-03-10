import re
import glob
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

CONN_DIR = ROOT / "results" / "connectomes" / "ABIDE12" / "ABIDE12" / "fd_0p3" / "matrices"


PHENO_DIR = ROOT / "phenotypes" 

BINS   = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]

SEX_MAP = {1: "Male", 2: "Female"}
DX_MAP = {1: "ASD", 2: "Control"}

THRESHOLDS = ["0.3"]

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


def load_phenotypes(pheno_dir: Path) -> pd.DataFrame:
    files = sorted(pheno_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {pheno_dir}")

    dfs = []
    need = {"SUB_ID", "SEX", "AGE_AT_SCAN", "DX_GROUP"}

    for fp in files:
        name = fp.name.lower()

        if "site_lookup" in name or "lookup" in name:
            print(f"[INFO] Skipping helper file {fp.name}")
            continue

        try:
            df = read_csv_flexible(fp)
        except Exception as e:
            print(f"[WARN] Could not read {fp.name}: {e}")
            continue

        df.columns = [c.upper().strip() for c in df.columns]

        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].tolist()
            print(f"[WARN] {fp.name} has duplicate columns after normalization: {dupes}")
            df = df.loc[:, ~df.columns.duplicated()].copy()

        if not need.issubset(df.columns):
            print(f"[WARN] Skipping {fp.name} (missing {need - set(df.columns)})")
            continue

        keep = ["SITE_ID", "SUB_ID", "SEX", "AGE_AT_SCAN", "DX_GROUP"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()
        df["source_file"] = fp.name
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No usable phenotype CSVs in {pheno_dir}")

    ph = pd.concat(dfs, ignore_index=True)

    ph["SUB_ID"] = pd.to_numeric(ph["SUB_ID"], errors="coerce")
    ph["SEX"] = pd.to_numeric(ph["SEX"], errors="coerce")
    ph["AGE_AT_SCAN"] = pd.to_numeric(ph["AGE_AT_SCAN"], errors="coerce")
    ph["DX_GROUP"] = pd.to_numeric(ph["DX_GROUP"], errors="coerce")

    ph = ph.dropna(subset=["SUB_ID", "SEX", "AGE_AT_SCAN", "DX_GROUP"]).copy()

    ph["SUB_ID"] = ph["SUB_ID"].astype(int)
    ph["SEX"] = ph["SEX"].astype(int)
    ph["DX_GROUP"] = ph["DX_GROUP"].astype(int)

    ph = ph.drop_duplicates(subset=["SUB_ID"], keep="first")

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

    return ph


def subject_ids_with_threshold(conn_dir: Path) -> set[int]:
    """Finds unique SUB_IDs from the .npy files."""
    files = list(conn_dir.rglob("*.npy"))

    ids = set()
    rx = re.compile(r"sub-(\d+)")
    for p in files:
        m = rx.search(p.name)
        if m:
            ids.add(int(m.group(1)))
    return ids


def counts_table(ph: pd.DataFrame, ids: set[int], thr: str) -> pd.DataFrame:
    sub = ph[ph["SUB_ID"].isin(ids)].copy()
    sub = sub.dropna(subset=["age_group"])

    # Group by age_group and our new combined group_label
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

    # Ensure all 4 core columns exist even if a category has 0 subjects
    expected_cols = ["Female_ASD", "Female_Control", "Male_ASD", "Male_Control"]
    for col in expected_cols:
        if col not in wide.columns:
            wide[col] = 0

    wide["Total"] = wide[expected_cols].sum(axis=1)
    wide.insert(0, "fd_threshold", f"fd-{thr}")

    wide["age_group"] = pd.Categorical(wide["age_group"], categories=LABELS, ordered=True)
    wide = wide.sort_values("age_group").reset_index(drop=True)

    # Reorder columns for readability
    col_order = ["fd_threshold", "age_group"] + expected_cols + ["Total"]
    final_cols = [c for c in col_order if c in wide.columns]
    
    return wide[final_cols]


def main():
    ph = load_phenotypes(PHENO_DIR)
    print(f"[INFO] Loaded phenotypes: {len(ph)} unique subjects")

    out_dir = ROOT / "results" / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tables = []
    
    for thr in THRESHOLDS:
        ids = subject_ids_with_threshold(CONN_DIR)
        print(f"[INFO] fd-{thr}: subjects with connectomes found on disk = {len(ids)}")

        missing_pheno = sorted([i for i in ids if i not in set(ph["SUB_ID"])])
        if missing_pheno:
            print(f"[WARN] fd-{thr}: {len(missing_pheno)} subjects have connectomes but no phenotype row (showing first 10): {missing_pheno[:10]}")

        tab = counts_table(ph, ids, thr)
        all_tables.append(tab)

        print(f"\n=== Sex and Diagnosis counts by age bin (fd-{thr}) ===")
        print(tab.to_string(index=False))

    final = pd.concat(all_tables, ignore_index=True)
    out_csv = out_dir / "sex_dx_by_age_bins.csv"
    final.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}")


if __name__ == "__main__":
    main()