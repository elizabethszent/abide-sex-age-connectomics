import re
import glob
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONN_DIR = ROOT / "connectomes" / "CC200" / "ABIDE2"
PHENO_DIR = ROOT / "phenotypes" / "ABIDE2"


BINS   = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]

SEX_MAP = {1: "Male", 2: "Female"}
THRESHOLDS = ["0.2", "0.3"]

def load_phenotypes(pheno_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(pheno_dir / "abide2*.csv")))
    
    if not files:
        raise FileNotFoundError(f"No abide2*.csv found in {pheno_dir.absolute()}")

    dfs = []
    for fp in files:
        #Added encoding="latin1" to handle non-UTF-8 characters
        df = pd.read_csv(fp, encoding="latin1")
        
        # Strip spaces from column names (handles 'AGE_AT_SCAN ')
        df.columns = df.columns.str.strip()
        
        need = {"SUB_ID", "SEX", "AGE_AT_SCAN"}
        if not need.issubset(df.columns):
            print(f"[WARN] Skipping {Path(fp).name} (missing columns: {need - set(df.columns)})")
            continue
            
        keep = ["SITE_ID", "SUB_ID", "SEX", "AGE_AT_SCAN"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()
        df["source_file"] = Path(fp).name
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No usable phenotype CSVs found in {pheno_dir}")

    ph = pd.concat(dfs, ignore_index=True)

    # normalize types
    ph["SUB_ID"] = pd.to_numeric(ph["SUB_ID"], errors="coerce")
    ph["SEX"] = pd.to_numeric(ph["SEX"], errors="coerce")
    ph["AGE_AT_SCAN"] = pd.to_numeric(ph["AGE_AT_SCAN"], errors="coerce")
    ph = ph.dropna(subset=["SUB_ID", "SEX", "AGE_AT_SCAN"]).copy()
    ph["SUB_ID"] = ph["SUB_ID"].astype(int)
    ph["SEX"] = ph["SEX"].astype(int)

    ph = ph.drop_duplicates(subset=["SUB_ID"], keep="first")

    ph["age_group"] = pd.cut(
        ph["AGE_AT_SCAN"],
        bins=BINS,
        labels=LABELS,
        right=False,
        include_lowest=True,
    )

    ph["sex_name"] = ph["SEX"].map(SEX_MAP).fillna(ph["SEX"].astype(str))

    return ph

def subject_ids_with_threshold(conn_dir: Path, thr: str) -> set[int]:
    patt = f"*fd-{thr}*connectome*.npz"
    files = list(conn_dir.rglob(patt))
    if not files:
        files = list(conn_dir.rglob(f"*fd-{thr}*.npz"))

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

    # observed=False avoids issues with empty categories in newer Pandas versions
    g = (
        sub.groupby(["age_group", "sex_name"], observed=False)["SUB_ID"]
        .nunique()
        .reset_index(name="n_subjects")
    )

    wide = (
        g.pivot(index="age_group", columns="sex_name", values="n_subjects")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    if "Male" not in wide.columns: wide["Male"] = 0
    if "Female" not in wide.columns: wide["Female"] = 0

    wide["Total"] = wide["Male"] + wide["Female"]
    wide.insert(0, "fd_threshold", f"fd-{thr}")

    wide["age_group"] = pd.Categorical(wide["age_group"], categories=LABELS, ordered=True)
    wide = wide.sort_values("age_group").reset_index(drop=True)

    return wide

def main():
    ph = load_phenotypes(PHENO_DIR)
    print(f"[INFO] Loaded phenotypes: {len(ph)} unique subjects")

    out_dir = ROOT / "results" / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tables = []
    for thr in THRESHOLDS:
        ids = subject_ids_with_threshold(CONN_DIR, thr)
        print(f"[INFO] fd-{thr}: subjects with connectomes found on disk = {len(ids)}")

        missing_pheno = sorted([i for i in ids if i not in set(ph["SUB_ID"])])
        if missing_pheno:
            print(f"[WARN] fd-{thr}: {len(missing_pheno)} subjects missing phenotype rows. Showing first 10: {missing_pheno[:10]}")

        tab = counts_table(ph, ids, thr)
        all_tables.append(tab)

        print(f"\n=== Sex counts by age bin (fd-{thr}) ===")
        print(tab.to_string(index=False))

    final = pd.concat(all_tables, ignore_index=True)
    out_csv = out_dir / "sex_by_age_bins_by_fd_threshold.csv"
    final.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}")

if __name__ == "__main__":
    main()