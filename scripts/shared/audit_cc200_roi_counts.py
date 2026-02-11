import re
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- USER PATHS ----------------
ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

CONN_DIR = ROOT / r"data\connectomes\cpac\nofilt_noglobal\cc200_z"
PHENO_CSV = ROOT / r"data\Phenotypic_V1_0b_preprocessed1.csv"
ROI_DIR = ROOT / r"data\roi_timeseries\cpac\nofilt_noglobal\rois_cc200"

OUT_DIR = ROOT / r"results\qc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "cc200_roi_and_connectome_audit.csv"
# -------------------------------------------

EXPECTED_N = 200

def find_timeseries_file(file_id: str, roi_dir: Path) -> Path | None:
    """
    Tries to find a .1D file for this FILE_ID inside ROI_DIR.
    Handles patterns like: <FILE_ID>_rois*.1D or <FILE_ID>*.1D
    """
    patterns = [
        f"{file_id}_rois*.1D",
        f"{file_id}*.1D",
    ]
    for pat in patterns:
        hits = list(roi_dir.glob(pat))
        if hits:
            # prefer ones that contain "_rois" if multiple
            hits_sorted = sorted(hits, key=lambda p: ("_rois" not in p.name, len(p.name)))
            return hits_sorted[0]
    return None

def count_columns_fast_1d(path: Path) -> int | None:
    """
    Counts number of columns in an AFNI-style .1D file by reading the first non-empty line.
    Returns None if file can't be parsed.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Some .1D may have comments; skip if starts with '#'
                if line.startswith("#"):
                    continue
                # split on whitespace
                cols = re.split(r"\s+", line)
                return len(cols)
    except Exception:
        return None
    return None

def get_connectome_shape(file_id: str, conn_dir: Path) -> tuple[int, int] | None:
    fp = conn_dir / f"{file_id}.npy"
    if not fp.exists():
        return None
    try:
        mat = np.load(fp, mmap_mode="r")
        if mat.ndim != 2:
            return None
        return (int(mat.shape[0]), int(mat.shape[1]))
    except Exception:
        return None

def main():
    if not PHENO_CSV.exists():
        raise FileNotFoundError(f"Phenotypic CSV not found: {PHENO_CSV}")
    if not ROI_DIR.exists():
        raise FileNotFoundError(f"ROI_DIR not found: {ROI_DIR}")
    if not CONN_DIR.exists():
        raise FileNotFoundError(f"CONN_DIR not found: {CONN_DIR}")

    pheno = pd.read_csv(PHENO_CSV)
    pheno.columns = pheno.columns.str.strip()

    if "FILE_ID" not in pheno.columns:
        raise ValueError("Phenotypic file missing FILE_ID column")

    # Keep only columns that are helpful (don’t crash if missing)
    keep_cols = [c for c in ["FILE_ID", "SITE_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"] if c in pheno.columns]
    ph = pheno[keep_cols].copy()
    ph["FILE_ID"] = ph["FILE_ID"].astype(str).str.strip()
    ph = ph.drop_duplicates(subset=["FILE_ID"])

    rows = []
    missing_ts = 0
    missing_conn = 0

    for _, r in ph.iterrows():
        fid = r["FILE_ID"]

        ts_path = find_timeseries_file(fid, ROI_DIR)
        n_ts = None
        if ts_path is None:
            missing_ts += 1
        else:
            n_ts = count_columns_fast_1d(ts_path)

        conn_shape = get_connectome_shape(fid, CONN_DIR)
        if conn_shape is None:
            missing_conn += 1

        status = []
        if n_ts is None:
            status.append("NO_TS")
        elif n_ts != EXPECTED_N:
            status.append(f"TS_N={n_ts}")

        if conn_shape is None:
            status.append("NO_CONN")
        elif conn_shape != (EXPECTED_N, EXPECTED_N):
            status.append(f"CONN_SHAPE={conn_shape[0]}x{conn_shape[1]}")

        if not status:
            status_str = "OK_200"
        else:
            status_str = ";".join(status)

        rows.append({
            **{c: r.get(c, np.nan) for c in keep_cols},
            "ts_file": str(ts_path) if ts_path is not None else "",
            "n_roi_timeseries": n_ts if n_ts is not None else np.nan,
            "connectome_shape": f"{conn_shape[0]}x{conn_shape[1]}" if conn_shape is not None else "",
            "status": status_str,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote audit CSV -> {OUT_CSV}")

    # Print summaries that help you decide “drop ROI vs drop subject”
    print("\n=== Summary: timeseries ROI counts ===")
    if "n_roi_timeseries" in out.columns:
        print(out["n_roi_timeseries"].value_counts(dropna=False).sort_index())

    print("\n=== Summary: connectome shapes ===")
    print(out["connectome_shape"].replace("", np.nan).value_counts(dropna=False))

    print("\n=== Summary: status counts ===")
    print(out["status"].value_counts())

    print(f"\nMissing timeseries files: {missing_ts}")
    print(f"Missing connectome files: {missing_conn}")

    # Helpful: site breakdown of bad cases
    if "SITE_ID" in out.columns:
        bad = out[out["status"] != "OK_200"].copy()
        if not bad.empty:
            print("\n=== Bad cases by SITE_ID (top 15) ===")
            print(bad["SITE_ID"].value_counts().head(15))

            print("\n=== Bad cases by (SITE_ID, n_roi_timeseries) ===")
            print(
                bad.groupby(["SITE_ID", "n_roi_timeseries"])
                   .size()
                   .sort_values(ascending=False)
                   .head(25)
            )

if __name__ == "__main__":
    main()
