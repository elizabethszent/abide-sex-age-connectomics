import re
import numpy as np
import pandas as pd
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "results").exists() and (p / "data").exists():
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

OUT_DIR = ROOT / "results" / "hubs" / "pc_z_strength_sitecov"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "OVERALL_ageSexMatched_fd-0.2": {
        "module_file": ROOT / "results" / "group_connectomes" / "ABIDE12_CC200" / "ABIDE_modules_asym_min10_fd-0.2.txt",
        "metadata_file": ROOT / "data" / "metadata" / "ABIDE12_phenotypes_combined_fd_0p2.csv",
        "conn_dir": ROOT / "results" / "connectomes" / "ABIDE12" / "ABIDE12" / "fd_0p2" / "matrices",
    },
    "OVERALL_ageSexMatched_fd-0.3": {
        "module_file": ROOT / "results" / "group_connectomes" / "ABIDE12_CC200" / "ABIDE_modules_asym_min10_fd-0.3.txt",
        "metadata_file": ROOT / "data" / "metadata" / "ABIDE12_phenotypes_combined_fd_0p3.csv",
        "conn_dir": ROOT / "results" / "connectomes" / "ABIDE12" / "ABIDE12" / "fd_0p3" / "matrices",
    },
}

BINS = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
RIGHT = False


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
        f"Could not decode {fp}",
    )


def to_subid7(x) -> str | None:
    if pd.isna(x):
        return None
    try:
        return f"{int(float(x)):07d}"
    except Exception:
        return None


def add_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AGE_GROUP"] = pd.cut(
        df["AGE_AT_SCAN"],
        bins=BINS,
        labels=LABELS,
        right=RIGHT,
        include_lowest=True,
    )
    return df


def load_metadata(metadata_file: Path) -> pd.DataFrame:
    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_file}")

    df = read_csv_flexible(metadata_file)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{metadata_file.name} missing required columns: {sorted(missing)}")

    df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce")
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

    if "func_mean_fd" in df.columns:
        df["func_mean_fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
    else:
        df["func_mean_fd"] = np.nan

    if "FIQ" in df.columns:
        df["FIQ"] = pd.to_numeric(df["FIQ"], errors="coerce")
    else:
        df["FIQ"] = np.nan

    if "RIGHT_HANDED" in df.columns:
        df["RIGHT_HANDED"] = pd.to_numeric(df["RIGHT_HANDED"], errors="coerce")
    else:
        df["RIGHT_HANDED"] = np.nan

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]).copy()

    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)

    df["subid7"] = df["SUB_ID"].map(to_subid7)

    if "AGE_GROUP" not in df.columns:
        df = add_age_group_column(df)
    else:
        df["AGE_GROUP"] = df["AGE_GROUP"].astype(str).str.strip()
        df.loc[df["AGE_GROUP"].str.lower() == "nan", "AGE_GROUP"] = np.nan

    df = df.dropna(subset=["subid7", "AGE_GROUP"]).copy()
    df = df.sort_values(["SITE_ID", "SUB_ID"]).drop_duplicates(subset=["SUB_ID"], keep="first")

    keep_cols = [
        "SUB_ID",
        "subid7",
        "DX_GROUP",
        "SEX",
        "AGE_AT_SCAN",
        "AGE_GROUP",
        "SITE_ID",
        "func_mean_fd",
        "FIQ",
        "RIGHT_HANDED",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].copy()


def load_modules_txt(module_file: Path) -> np.ndarray:
    if not module_file.exists():
        raise FileNotFoundError(f"Missing module file: {module_file}")

    df = pd.read_csv(module_file, sep=r"\s+")
    if "ROI_index" not in df.columns or "Module" not in df.columns:
        raise ValueError(f"{module_file} must contain ROI_index and Module columns")

    df = df.sort_values("ROI_index")
    mods = df["Module"].to_numpy().astype(int)

    if mods.ndim != 1:
        raise ValueError(f"Expected 1D modules, got {mods.shape}")

    return mods


def extract_subid7_from_name(name: str, stem: str) -> str | None:
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


def build_connectome_index(conn_dir: Path) -> dict[str, Path]:
    if not conn_dir.exists():
        raise FileNotFoundError(f"Missing connectome directory: {conn_dir}")

    files = list(conn_dir.rglob("*.npy")) + list(conn_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No connectome files found under {conn_dir}")

    idx: dict[str, Path] = {}

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

    return idx


def load_matrix(fp: Path) -> np.ndarray:
    if fp.suffix == ".npy":
        mat = np.load(fp)
    elif fp.suffix == ".npz":
        with np.load(fp, allow_pickle=False) as d:
            if "r" in d.files:
                mat = np.asarray(d["r"])
            elif "corr" in d.files:
                mat = np.asarray(d["corr"])
            elif "z" in d.files:
                mat = np.asarray(d["z"])
            else:
                raise ValueError(f"{fp.name}: .npz does not contain key 'r', 'corr', or 'z'")
    else:
        raise ValueError(f"Unsupported matrix file type: {fp}")

    mat = np.asarray(mat, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix not square: {fp} -> {mat.shape}")

    return mat


def strengths_by_module(W: np.ndarray, roi2mod: np.ndarray, n_mod: int) -> np.ndarray:
    n = W.shape[0]
    out = np.zeros((n, n_mod), dtype=float)

    for m in range(1, n_mod + 1):
        idx = np.where(roi2mod == m)[0]
        if idx.size == 0:
            continue
        out[:, m - 1] = W[:, idx].sum(axis=1)

    return out


def participation_coefficient(k_by_mod: np.ndarray, k_total: np.ndarray) -> np.ndarray:
    n = k_by_mod.shape[0]
    pc = np.zeros(n, dtype=float)

    for i in range(n):
        ki = k_total[i]
        if ki <= 0:
            pc[i] = 0.0
        else:
            frac = k_by_mod[i, :] / ki
            pc[i] = 1.0 - np.sum(frac ** 2)

    return pc


def within_module_z(k_by_mod: np.ndarray, roi2mod: np.ndarray, n_mod: int) -> np.ndarray:
    n = k_by_mod.shape[0]
    z = np.zeros(n, dtype=float)

    for m in range(1, n_mod + 1):
        idx = np.where(roi2mod == m)[0]
        if idx.size == 0:
            continue

        k_within = k_by_mod[idx, m - 1]
        mu = k_within.mean()
        sd = k_within.std(ddof=1)
        z[idx] = (k_within - mu) / sd if sd > 0 else 0.0

    return z


def compute_all_metrics_from_full(W_full: np.ndarray, roi2mod: np.ndarray):
    n_mod = int(roi2mod.max())

    W = W_full.copy()
    np.fill_diagonal(W, 0.0)

    W_pos = np.where(W > 0, W, 0.0)
    W_neg = np.where(W < 0, -W, 0.0)
    W_abs = W_pos + W_neg

    strength_pos = W_pos.sum(axis=1)
    strength_neg = W_neg.sum(axis=1)
    strength_abs = W_abs.sum(axis=1)

    k_abs_by_mod = strengths_by_module(W_abs, roi2mod, n_mod)
    k_pos_by_mod = strengths_by_module(W_pos, roi2mod, n_mod)
    k_neg_by_mod = strengths_by_module(W_neg, roi2mod, n_mod)

    pc = participation_coefficient(k_abs_by_mod, strength_abs)
    pc_pos = participation_coefficient(k_pos_by_mod, strength_pos)
    pc_neg = participation_coefficient(k_neg_by_mod, strength_neg)

    z = within_module_z(k_abs_by_mod, roi2mod, n_mod)
    z_pos = within_module_z(k_pos_by_mod, roi2mod, n_mod)
    z_neg = within_module_z(k_neg_by_mod, roi2mod, n_mod)

    return {
        "PC": pc,
        "PC_pos": pc_pos,
        "PC_neg": pc_neg,
        "z": z,
        "z_pos": z_pos,
        "z_neg": z_neg,
        "strength_pos": strength_pos,
        "strength_neg": strength_neg,
    }


def run_one_scenario(scenario: str, cfg: dict):
    print(f"\n=== {scenario} ===")

    roi2mod = load_modules_txt(cfg["module_file"])
    n_roi = len(roi2mod)
    n_mod = int(roi2mod.max())
    print(f"[INFO] loaded modules: {n_roi} ROIs, {n_mod} modules")

    subjects = load_metadata(cfg["metadata_file"])
    print(f"[INFO] metadata subjects: {len(subjects)}")

    conn_idx = build_connectome_index(cfg["conn_dir"])
    print(f"[INFO] connectomes indexed: {len(conn_idx)}")

    subjects = subjects[subjects["subid7"].isin(conn_idx.keys())].copy()
    print(f"[INFO] subjects with connectomes after intersection: {len(subjects)}")

    rows = []
    n_missing = 0
    n_bad = 0

    for _, row in subjects.iterrows():
        subid7 = row["subid7"]
        conn_fp = conn_idx.get(subid7)

        if conn_fp is None:
            n_missing += 1
            continue

        try:
            mat = load_matrix(conn_fp)
        except Exception:
            n_bad += 1
            continue

        if mat.shape != (n_roi, n_roi):
            n_bad += 1
            continue

        if not np.isfinite(mat).all():
            n_bad += 1
            continue

        W_full = mat.copy()
        W_full = 0.5 * (W_full + W_full.T)
        np.fill_diagonal(W_full, 0.0)

        metrics = compute_all_metrics_from_full(W_full, roi2mod)

        for node_idx in range(n_roi):
            out_row = {
                "SUB_ID": int(row["SUB_ID"]),
                "FILE_ID": subid7,
                "SITE_ID": str(row["SITE_ID"]).strip(),
                "SEX": int(row["SEX"]),
                "DX_GROUP": int(row["DX_GROUP"]),
                "AGE_AT_SCAN": float(row["AGE_AT_SCAN"]),
                "AGE_GROUP": str(row["AGE_GROUP"]),
                "func_mean_fd": float(row["func_mean_fd"]) if pd.notna(row["func_mean_fd"]) else np.nan,
                "FIQ": float(row["FIQ"]) if pd.notna(row["FIQ"]) else np.nan,
                "RIGHT_HANDED": float(row["RIGHT_HANDED"]) if pd.notna(row["RIGHT_HANDED"]) else np.nan,
                "scenario": scenario,
                "node": node_idx + 1,
                "module": int(roi2mod[node_idx]),
            }

            for k, arr in metrics.items():
                out_row[k] = float(arr[node_idx])

            rows.append(out_row)

    if not rows:
        raise RuntimeError(f"{scenario}: no usable subject matrices found")

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / f"{scenario}_node_metrics.csv"
    out_df.to_csv(out_path, index=False)

    print(f"[SAVED] {out_path}")
    print(f"[INFO] node rows: {len(out_df)}")
    print(f"[INFO] subjects used: {out_df['SUB_ID'].nunique()}")
    print(f"[INFO] subjects with FIQ: {out_df[['SUB_ID', 'FIQ']].drop_duplicates()['FIQ'].notna().sum()}")
    print(f"[INFO] subjects with RIGHT_HANDED: {out_df[['SUB_ID', 'RIGHT_HANDED']].drop_duplicates()['RIGHT_HANDED'].notna().sum()}")
    print(f"[INFO] skipped missing connectome: {n_missing}")
    print(f"[INFO] skipped bad matrix: {n_bad}")

    counts = (
        out_df[["SUB_ID", "SEX", "DX_GROUP", "AGE_GROUP"]]
        .drop_duplicates()
        .groupby(["AGE_GROUP", "SEX", "DX_GROUP"])
        .size()
        .reset_index(name="n_subjects")
        .sort_values(["AGE_GROUP", "SEX", "DX_GROUP"])
    )
    print("[INFO] subject counts by age/sex/dx:")
    print(counts.to_string(index=False))


def main():
    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] out dir: {OUT_DIR}")

    for scenario, cfg in SCENARIOS.items():
        run_one_scenario(scenario, cfg)

    print("\n[DONE] node metrics rebuilt with fully connected PC, z, strength, plus FIQ and RIGHT_HANDED.")


if __name__ == "__main__":
    main()