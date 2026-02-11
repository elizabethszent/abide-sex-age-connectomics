import numpy as np
import pandas as pd
from pathlib import Path
import shutil

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

AUDIT_CSV = ROOT / r"results\qc\cc200_roi_and_connectome_audit.csv"
ROI_DIR   = ROOT / r"data\roi_timeseries\cpac\nofilt_noglobal\rois_cc200"
CONN_DIR  = ROOT / r"data\connectomes\cpac\nofilt_noglobal\cc200_z"

BACKUP_DIR = CONN_DIR / "_backup_bad_shapes"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 200

# If your timeseries can contain NaNs, turning this on is a safe/typical choice.
NAN_TO_NUM = True


def find_timeseries_file(file_id: str) -> Path | None:
    patterns = [
        f"{file_id}_rois*.1D",
        f"{file_id}*.1D",
    ]
    for pat in patterns:
        hits = list(ROI_DIR.glob(pat))
        if hits:
            hits_sorted = sorted(hits, key=lambda p: ("_rois" not in p.name, len(p.name)))
            return hits_sorted[0]
    return None


def load_timeseries_1d(path: Path) -> np.ndarray:
    ts = np.loadtxt(path)
    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    if NAN_TO_NUM:
        ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)

    return ts  # shape (T, N)


def corr_and_fisher_z(ts: np.ndarray) -> np.ndarray:
    # ts: (T, N)
    C = np.corrcoef(ts.T)  # (N, N)

    # Ensure diagonal is exactly 0 before and after transform
    np.fill_diagonal(C, 0.0)

    # Fisher z transform (avoid inf at +/-1)
    eps = 1e-7
    C_clip = np.clip(C, -1 + eps, 1 - eps)
    Z = np.arctanh(C_clip)
    np.fill_diagonal(Z, 0.0)

    return Z


def main():
    if not AUDIT_CSV.exists():
        raise FileNotFoundError(f"Missing audit CSV: {AUDIT_CSV}")
    if not ROI_DIR.exists():
        raise FileNotFoundError(f"Missing ROI_DIR: {ROI_DIR}")
    if not CONN_DIR.exists():
        raise FileNotFoundError(f"Missing CONN_DIR: {CONN_DIR}")

    df = pd.read_csv(AUDIT_CSV)
    df["FILE_ID"] = df["FILE_ID"].astype(str).str.strip()

    # bad = not 200x200 OR missing
    bad = df[(df["connectome_shape"].fillna("") != "200x200")].copy()
    print(f"Found {len(bad)} subjects with non-200x200 (or missing) connectomes.")

    fixed = 0
    failed = 0
    skipped_already_good = 0
    backed_up = 0

    for _, r in bad.iterrows():
        fid = r["FILE_ID"]
        ts_path = find_timeseries_file(fid)
        if ts_path is None:
            print(f"[FAIL] {fid}: no timeseries file found")
            failed += 1
            continue

        out_path = CONN_DIR / f"{fid}.npy"

        # Extra safety: if the existing connectome is already 200x200, don't overwrite it.
        # (This protects you if the audit CSV is stale or you rerun after fixing.)
        if out_path.exists():
            try:
                old = np.load(out_path, mmap_mode="r")
                if getattr(old, "shape", None) == (EXPECTED_N, EXPECTED_N):
                    skipped_already_good += 1
                    continue
            except Exception:
                # If old file can't be loaded, we'll treat it as bad and rebuild.
                pass

        try:
            ts = load_timeseries_1d(ts_path)
            if ts.shape[1] != EXPECTED_N:
                print(f"[FAIL] {fid}: timeseries has {ts.shape[1]} cols, expected 200 ({ts_path.name})")
                failed += 1
                continue

            Z = corr_and_fisher_z(ts)
            if Z.shape != (EXPECTED_N, EXPECTED_N):
                print(f"[FAIL] {fid}: rebuilt matrix wrong shape {Z.shape}")
                failed += 1
                continue

            # Backup old if exists (and is not already backed up)
            if out_path.exists():
                backup_path = BACKUP_DIR / f"{fid}.npy"
                if not backup_path.exists():
                    shutil.copy2(out_path, backup_path)
                    backed_up += 1

            np.save(out_path, Z)
            fixed += 1

            if fixed <= 10:
                print(f"[OK] Rebuilt {fid} -> {out_path.name} from {ts_path.name}")

        except Exception as e:
            print(f"[FAIL] {fid}: {type(e).__name__}: {e}")
            failed += 1

    print("\nDone.")
    print(f"Fixed:              {fixed}")
    print(f"Failed:             {failed}")
    print(f"Skipped already OK: {skipped_already_good}")
    print(f"Backups created:    {backed_up}")
    print(f"Backups saved to:   {BACKUP_DIR}")


if __name__ == "__main__":
    main()
