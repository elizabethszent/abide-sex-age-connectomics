import re
import numpy as np
import pandas as pd
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "phenotypes").exists() and (p / "results").exists():
            return p
    raise FileNotFoundError(
        "Could not find repo root containing both 'phenotypes/' and 'results/'."
    )


ROOT = find_repo_root(Path(__file__).resolve().parent)

PHENO_DIR = ROOT / "phenotypes"
ABIDE1_PHENO = PHENO_DIR / "ABIDE_phenotypes_combined.csv"
ABIDE2_PHENO = PHENO_DIR / "ABIDE2_phenotypes_combined.csv"

FD_ROOTS = {
    "0.2": ROOT / "results" / "connectomes" / "ABIDE12" / "ABIDE12" / "fd_0p2",
    "0.3": ROOT / "results" / "connectomes" / "ABIDE12" / "ABIDE12" / "fd_0p3",
}

OUT_DIR = ROOT / "results" / "group_connectomes" / "ABIDE12_CC200"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FD_LIST = ["0.2", "0.3"]

# ABIDE conventions:
# SEX: 1=Male, 2=Female
# DX_GROUP: 1=ASD, 2=Control
REQUIRED_COLS = {"SITE_ID", "SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN"}


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


def to_subid7(x) -> str | None:
    if pd.isna(x):
        return None
    try:
        return f"{int(float(x)):07d}"
    except Exception:
        return None


def age_group(age: float) -> str | None:
    if pd.isna(age):
        return None
    a = float(age)
    if a <= 9:
        return "child_0_9"
    if a <= 12:
        return "preteen_10_12"
    if a <= 17:
        return "teen_13_17"
    return "adult_18_plus"


def load_one_combined(fp: Path, dataset_name: str) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(f"Missing phenotype file: {fp}")

    df = read_csv_flexible(fp)
    df.columns = [str(c).upper().strip() for c in df.columns]

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        print(f"[WARN] {fp.name} has duplicate columns after normalization: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()].copy()

    if not REQUIRED_COLS.issubset(df.columns):
        raise RuntimeError(
            f"{fp.name} is missing required columns: {REQUIRED_COLS - set(df.columns)}"
        )

    keep = ["SITE_ID", "SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN"]
    if "FIQ" in df.columns:
        keep.append("FIQ")

    out = df[keep].copy()
    out["PHENO_FILE"] = fp.name
    out["DATASET"] = dataset_name
    return out


def load_combined_phenotypes() -> pd.DataFrame:
    dfs = [
        load_one_combined(ABIDE1_PHENO, "ABIDE1"),
        load_one_combined(ABIDE2_PHENO, "ABIDE2"),
    ]

    pheno = pd.concat(dfs, ignore_index=True)

    pheno["SUB_ID"] = pd.to_numeric(pheno["SUB_ID"], errors="coerce")
    pheno["DX_GROUP"] = pd.to_numeric(pheno["DX_GROUP"], errors="coerce")
    pheno["SEX"] = pd.to_numeric(pheno["SEX"], errors="coerce")
    pheno["AGE_AT_SCAN"] = pd.to_numeric(pheno["AGE_AT_SCAN"], errors="coerce")
    if "FIQ" in pheno.columns:
        pheno["FIQ"] = pd.to_numeric(pheno["FIQ"], errors="coerce")

    pheno = pheno.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN"]).copy()

    pheno["SUB_ID"] = pheno["SUB_ID"].astype(int)
    pheno["DX_GROUP"] = pheno["DX_GROUP"].astype(int)
    pheno["SEX"] = pheno["SEX"].astype(int)
    pheno["SITE_ID"] = pheno["SITE_ID"].astype(str).str.strip()

    pheno["subid7"] = pheno["SUB_ID"].map(to_subid7)
    pheno["age_group"] = pheno["AGE_AT_SCAN"].map(age_group)

    pheno = pheno.dropna(subset=["subid7", "age_group"]).copy()
    pheno["subid7"] = pheno["subid7"].astype(str)

    keep_cols = [
        "SITE_ID",
        "SUB_ID",
        "subid7",
        "DX_GROUP",
        "SEX",
        "AGE_AT_SCAN",
        "age_group",
        "DATASET",
        "PHENO_FILE",
    ]
    if "FIQ" in pheno.columns:
        keep_cols.append("FIQ")

    pheno = pheno[keep_cols].copy()
    pheno = pheno.sort_values(["DATASET", "SITE_ID", "SUB_ID"]).drop_duplicates(
        subset=["SUB_ID"], keep="first"
    )

    return pheno


def get_matrix_dir(fd_root: Path) -> Path:
    """
    Prefer matrices_z because this script averages z-matrices.
    Fall back to matrices if matrices_z is missing.
    """
    cand_z = fd_root / "matrices_z"
    cand_r = fd_root / "matrices"

    if cand_z.exists():
        return cand_z
    if cand_r.exists():
        return cand_r

    raise FileNotFoundError(
        f"Could not find either 'matrices_z' or 'matrices' under {fd_root}"
    )


def extract_subid7_from_name(name: str, stem: str) -> str | None:
    """
    Supports:
    - sub-29006 -> 0029006
    - sub-0050004 -> 0050004
    - plain stem 29006 or 0050004
    """
    m = re.search(r"sub-(\d+)", name)
    if m:
        try:
            return f"{int(m.group(1)):07d}"
        except Exception:
            return None

    m = re.match(r"(\d+)$", stem)
    if m:
        try:
            return f"{int(m.group(1)):07d}"
        except Exception:
            return None

    return None


def extract_run_number(name: str) -> int:
    m = re.search(r"run-(\d+)", name)
    if m:
        return int(m.group(1))
    return 999


def build_connectome_index(matrix_dir: Path) -> dict[str, Path]:
    """
    Map subid7 -> matrix path.
    Supports both .npz and .npy files.
    Prefers run-1 if multiple files exist for a subject.
    """
    idx: dict[str, Path] = {}

    files = list(matrix_dir.rglob("*.npz")) + list(matrix_dir.rglob("*.npy"))
    if not files:
        print(f"[WARN] No connectome files found under {matrix_dir}")
        return idx

    for fp in files:
        subid7 = extract_subid7_from_name(fp.name, fp.stem)
        if subid7 is None:
            continue

        run = extract_run_number(fp.name)

        if subid7 not in idx:
            idx[subid7] = fp
        else:
            old_run = extract_run_number(idx[subid7].name)
            if run == 1 and old_run != 1:
                idx[subid7] = fp

    print(f"[INFO] {matrix_dir}: matched {len(idx)} unique subjects with connectomes")
    return idx


def load_conn_meta(fp: Path) -> tuple[float | None, float | None]:
    """
    Return (n_kept, n_total) if present in .npz; otherwise (None, None).
    """
    if fp.suffix != ".npz":
        return None, None

    with np.load(fp, allow_pickle=False) as d:
        n_kept = float(d["n_kept"]) if "n_kept" in d.files else None
        n_total = float(d["n_total"]) if "n_total" in d.files else None
    return n_kept, n_total


def load_z(fp: Path) -> np.ndarray:
    """
    Load a 200x200 z-matrix from either:
    - .npz with key 'z'
    - .npy matrix
    """
    if fp.suffix == ".npz":
        with np.load(fp, allow_pickle=False) as d:
            if "z" in d.files:
                z = np.asarray(d["z"])
            else:
                raise ValueError(f"{fp.name}: .npz does not contain key 'z'")
    elif fp.suffix == ".npy":
        z = np.asarray(np.load(fp, allow_pickle=False))
    else:
        raise ValueError(f"Unsupported file type: {fp}")

    if z.shape != (200, 200):
        raise ValueError(f"{fp.name}: expected (200, 200), got {z.shape}")
    return z


def robust_standardize(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = mad if (mad is not None and mad > 1e-12) else (np.nanstd(x) + 1e-12)
    return (x - med) / scale


def choose_matching_columns(stratum: pd.DataFrame) -> list[str]:
    cols = ["AGE_AT_SCAN"]

    if "frac_kept" in stratum.columns and np.isfinite(stratum["frac_kept"]).any():
        cols.append("frac_kept")

    if "FIQ" in stratum.columns and np.isfinite(stratum["FIQ"]).any():
        cols.append("FIQ")

    return cols


def main():
    pheno = load_combined_phenotypes()
    print(f"[INFO] Phenotype rows loaded: {len(pheno)}")
    print(f"[INFO] Unique subjects: {pheno['SUB_ID'].nunique()}")

    for fd in FD_LIST:
        print("\n====================")
        print(f"FD = {fd}")
        print("====================")

        fd_root = FD_ROOTS[fd]
        if not fd_root.exists():
            print(f"[WARN] Missing FD directory: {fd_root}")
            continue

        matrix_dir = get_matrix_dir(fd_root)
        print(f"[INFO] Using matrix directory: {matrix_dir}")

        conn_idx = build_connectome_index(matrix_dir)
        print(f"[INFO] Connectomes available for fd={fd}: {len(conn_idx)}")

        has_conn = pheno["subid7"].isin(conn_idx.keys())
        df = pheno[has_conn].copy()
        df["conn_path"] = df["subid7"].map(conn_idx)

        frac_kept = []
        for p in df["conn_path"]:
            nk, nt = load_conn_meta(p)
            if nk is None or nt is None or nt == 0:
                frac_kept.append(np.nan)
            else:
                frac_kept.append(nk / nt)
        df["frac_kept"] = frac_kept

        print(f"[INFO] Subjects with connectomes for fd={fd}: {len(df)}")

        selected_rows = []

        for (dx, ag), stratum in df.groupby(["DX_GROUP", "age_group"]):
            females = stratum[stratum["SEX"] == 2].copy()
            males = stratum[stratum["SEX"] == 1].copy()

            nF = len(females)
            nM = len(males)
            if nF == 0 or nM == 0:
                continue

            n = min(nF, nM)

            if nF > n:
                females = females.sample(n=n, random_state=42)

            cols = choose_matching_columns(stratum)

            for c in cols:
                med = np.nanmedian(stratum[c].to_numpy(dtype=float))
                females[c] = females[c].fillna(med)
                males[c] = males[c].fillna(med)

            F = females[cols].to_numpy(dtype=float)
            M = males[cols].to_numpy(dtype=float)

            ZF = []
            ZM = []
            for j in range(len(cols)):
                combined = np.concatenate([F[:, j], M[:, j]])
                z_comb = robust_standardize(combined)
                zF = z_comb[: F.shape[0]]
                zM = z_comb[F.shape[0] :]
                ZF.append(zF)
                ZM.append(zM)

            ZF = np.stack(ZF, axis=1)
            ZM = np.stack(ZM, axis=1)

            centroid = np.mean(ZF, axis=0)
            dists = np.sum((ZM - centroid) ** 2, axis=1)

            males = males.assign(match_score=dists)
            males_sel = males.nsmallest(n, "match_score")

            females = females.assign(match_score=np.nan)
            selected_rows.append(females)
            selected_rows.append(males_sel)

            print(
                f"[STRATUM] DX={int(dx)} age={ag}: "
                f"females={len(females)} males_selected={len(males_sel)} "
                f"(males_avail={nM}) covariates={cols}"
            )

        if not selected_rows:
            raise RuntimeError(f"No matched subjects found for fd={fd}")

        sel = pd.concat(selected_rows, ignore_index=True)
        sel = sel.drop_duplicates(subset=["subid7"])

        sex_counts = sel["SEX"].value_counts().to_dict()
        print(f"[INFO] Selected total={len(sel)} sex_counts={sex_counts}")

        sum_mat = np.zeros((200, 200), dtype=np.float64)
        cnt_mat = np.zeros((200, 200), dtype=np.int32)

        for fp in sel["conn_path"]:
            z = load_z(fp).astype(np.float64)
            mask = ~np.isnan(z)
            sum_mat[mask] += z[mask]
            cnt_mat[mask] += 1

        mean_z = np.full((200, 200), np.nan, dtype=np.float64)
        valid = cnt_mat > 0
        mean_z[valid] = sum_mat[valid] / cnt_mat[valid]
        mean_r = np.tanh(mean_z)

        out_z = OUT_DIR / f"OVERALL_ageSexMatched_fd-{fd}_mean_z.npy"
        out_r = OUT_DIR / f"OVERALL_ageSexMatched_fd-{fd}_mean_r.npy"
        out_csv = OUT_DIR / f"OVERALL_ageSexMatched_fd-{fd}_selected_subjects.csv"

        np.save(out_z, mean_z)
        np.save(out_r, mean_r)
        sel.to_csv(out_csv, index=False)

        print("[DONE] Saved:")
        print(f"  {out_z}")
        print(f"  {out_r}")
        print(f"  {out_csv}")


if __name__ == "__main__":
    main()