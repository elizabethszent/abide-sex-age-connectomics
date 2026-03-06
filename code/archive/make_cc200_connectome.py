#!/usr/bin/env python3
"""
make_cc200_connectome.py

Inputs:
  - fMRIPrep preprocessed BOLD (MNI space) .nii.gz
  - fMRIPrep confounds table (desc-confounds_regressors.tsv)
  - CC200 atlas label image

Does:
  1) Motion QC:
     - drop whole run if mean FD > --mean-fd-cutoff (default 0.2)
     - censor volumes with FD > --vol-fd-cutoff (default 0.5)
     - optionally censor nonsteady volumes (if present); if missing -> warn + skip
  2) Extract ROI time series (mean per label), standardize each ROI
  3) Compute ROI-ROI Pearson correlation connectome (+ optional Fisher-z)
  4) Save:
       <out-dir>/<out-prefix>.npy
       <out-dir>/<out-prefix>_z.npy (optional)
       <meta-dir>/<out-prefix>.json (QC + provenance)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bold", required=True)
    p.add_argument("--confounds", required=True)
    p.add_argument("--atlas", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--out-prefix", required=True)

    p.add_argument("--mean-fd-cutoff", type=float, default=0.2)
    p.add_argument("--vol-fd-cutoff", type=float, default=0.5)
    p.add_argument("--fd-col", default="framewise_displacement")
    p.add_argument(
        "--nonsteady-col",
        default="non_steady_state_outlier",
        help="Prefix for nonsteady columns (exact or startswith). Missing columns are allowed.",
    )

    p.add_argument("--min-trs", type=int, default=50)
    p.add_argument("--fisher-z", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _find_column(df: pd.DataFrame, wanted: str):
    if wanted in df.columns:
        return wanted
    low = {c.lower(): c for c in df.columns}
    return low.get(wanted.lower(), None)


def _get_fd_series(df: pd.DataFrame, fd_col: str) -> tuple[np.ndarray, str]:
    col = _find_column(df, fd_col)
    if col is None:
        for alt in ["fd", "FD", "FramewiseDisplacement", "framewiseDisplacement"]:
            col = _find_column(df, alt)
            if col is not None:
                break
    if col is None:
        raise ValueError(f"Missing FD column '{fd_col}' (and common alternatives) in confounds table.")
    fd = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return fd, col


def _get_nonsteady_mask(df: pd.DataFrame, nonsteady_prefix: str, n_trs: int, confounds_path: str):
    cols = [c for c in df.columns if (c == nonsteady_prefix) or str(c).startswith(nonsteady_prefix)]
    if not cols:
        print(f"WARN: no columns matching '{nonsteady_prefix}' in {confounds_path}; skipping nonsteady censoring", flush=True)
        return np.zeros(n_trs, dtype=bool), []
    arr = df[cols].fillna(0).to_numpy()
    mask = (arr != 0).any(axis=1)
    if mask.shape[0] != n_trs:
        mask = np.resize(mask.astype(bool), n_trs)
    return mask.astype(bool), cols


def compute_motion_qc(confounds_path: str, mean_fd_cutoff: float, vol_fd_cutoff: float, fd_col: str, nonsteady_col: str):
    df = pd.read_csv(confounds_path, sep="\t")
    fd, fd_used_col = _get_fd_series(df, fd_col)

    mean_fd = float(np.mean(fd))
    vol_bad = fd > float(vol_fd_cutoff)

    nonsteady, nonsteady_cols_found = _get_nonsteady_mask(
        df, nonsteady_col, n_trs=len(fd), confounds_path=confounds_path
    )

    keep = ~(vol_bad | nonsteady)

    n_total = int(len(fd))
    n_keep = int(np.sum(keep))
    pct_fd_gt = float(np.mean(vol_bad) * 100.0)

    drop_run = mean_fd > float(mean_fd_cutoff)

    qc = {
        "fd_col_used": fd_used_col,
        "nonsteady_prefix": nonsteady_col,
        "nonsteady_cols_found": nonsteady_cols_found,
        "mean_fd": mean_fd,
        "mean_fd_cutoff": float(mean_fd_cutoff),
        "vol_fd_cutoff": float(vol_fd_cutoff),
        "pct_vol_fd_gt_cutoff": pct_fd_gt,
        "n_trs_total": n_total,
        "n_trs_keep": n_keep,
        "pct_trs_keep": float(n_keep / max(1, n_total) * 100.0),
        "drop_run_mean_fd": bool(drop_run),
    }
    return keep, qc


def _resample_atlas_to_bold(atlas_img: nib.Nifti1Image, bold_img: nib.Nifti1Image):
    """
    Resample label atlas to match bold spatial grid (nearest neighbor).
    Requires scipy (used by nibabel.processing.resample_from_to).
    """
    if atlas_img.shape == bold_img.shape[:3] and np.allclose(atlas_img.affine, bold_img.affine):
        return atlas_img, {"atlas_resampled": False}

    try:
        from nibabel.processing import resample_from_to
    except Exception as e:
        raise RuntimeError(
            "Atlas/BOLD grid mismatch and nibabel.processing import failed. "
            "Install scipy in the venv: pip install scipy"
        ) from e

    target = (bold_img.shape[:3], bold_img.affine)
    atlas_rs = resample_from_to(atlas_img, target, order=0)  # nearest neighbor
    meta = {
        "atlas_resampled": True,
        "atlas_resample_method": "nibabel.processing.resample_from_to(order=0)",
        "atlas_shape_orig": tuple(atlas_img.shape),
        "atlas_shape_new": tuple(atlas_rs.shape),
    }
    return atlas_rs, meta


def extract_timeseries(bold_path: str, atlas_path: str, keep_mask: np.ndarray):
    bold_img = nib.load(bold_path)
    atlas_img = nib.load(atlas_path)

    # resample atlas if needed
    atlas_img, rs_meta = _resample_atlas_to_bold(atlas_img, bold_img)

    bold = np.asanyarray(bold_img.dataobj).astype(np.float32)  # X,Y,Z,T
    atlas = np.asanyarray(atlas_img.dataobj)
    atlas = np.rint(atlas).astype(np.int32)

    if bold.ndim != 4:
        raise ValueError(f"BOLD should be 4D but got shape {bold.shape} from {bold_path}")
    if atlas.shape != bold.shape[:3]:
        raise ValueError(f"After resampling, atlas shape {atlas.shape} still != BOLD spatial shape {bold.shape[:3]}")

    T = bold.shape[3]
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.shape[0] != T:
        raise ValueError(f"keep_mask length {keep_mask.shape[0]} != T={T}")

    labels = np.unique(atlas)
    labels = labels[labels != 0]
    labels = np.sort(labels)
    n_rois = int(labels.size)
    if n_rois == 0:
        raise ValueError("Atlas has no non-zero labels (ROIs).")

    keep_idx = np.where(keep_mask)[0]
    ts = np.zeros((int(keep_idx.size), n_rois), dtype=np.float32)

    bold_2d = bold.reshape(-1, T)  # voxels x T
    atlas_flat = atlas.reshape(-1)

    for j, lab in enumerate(labels):
        vox = np.where(atlas_flat == lab)[0]
        if vox.size == 0:
            continue
        roi_data = bold_2d[vox][:, keep_idx]  # voxels x kept_T
        ts[:, j] = roi_data.mean(axis=0)

    # standardize each ROI
    mu = ts.mean(axis=0, keepdims=True)
    sd = ts.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    ts = (ts - mu) / sd

    meta = {"ts_method": "nibabel_numpy_labels_mean", "n_rois": n_rois}
    meta.update(rs_meta)
    return ts, meta


def compute_connectome(ts: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(ts.T).astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir)
    meta_dir = Path(args.meta_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    out_mat = out_dir / f"{args.out_prefix}.npy"
    out_z = out_dir / f"{args.out_prefix}_z.npy"
    out_meta = meta_dir / f"{args.out_prefix}.json"

    if (not args.overwrite) and out_mat.exists() and out_meta.exists():
        print(f"SKIP: outputs exist for {args.out_prefix} (use --overwrite to force).", flush=True)
        return 0

    keep_mask, qc = compute_motion_qc(
        confounds_path=args.confounds,
        mean_fd_cutoff=args.mean_fd_cutoff,
        vol_fd_cutoff=args.vol_fd_cutoff,
        fd_col=args.fd_col,
        nonsteady_col=args.nonsteady_col,
    )

    qc["bold"] = os.path.abspath(args.bold)
    qc["confounds"] = os.path.abspath(args.confounds)
    qc["atlas"] = os.path.abspath(args.atlas)

    if qc["drop_run_mean_fd"]:
        qc["status"] = "excluded"
        qc["reason"] = f"mean_fd>{args.mean_fd_cutoff}"
        out_meta.write_text(json.dumps(qc, indent=2, sort_keys=True))
        print(f"EXCLUDE: {args.out_prefix} mean_fd={qc['mean_fd']:.4f} > {args.mean_fd_cutoff}", flush=True)
        return 0

    if qc["n_trs_keep"] < int(args.min_trs):
        qc["status"] = "excluded"
        qc["reason"] = f"n_trs_keep<{args.min_trs}"
        out_meta.write_text(json.dumps(qc, indent=2, sort_keys=True))
        print(f"EXCLUDE: {args.out_prefix} only {qc['n_trs_keep']} TRs kept (<{args.min_trs})", flush=True)
        return 0

    ts, ts_meta = extract_timeseries(args.bold, args.atlas, keep_mask)
    qc.update(ts_meta)

    corr = compute_connectome(ts)
    np.save(out_mat, corr)

    if args.fisher_z:
        z = np.arctanh(np.clip(corr, -0.999999, 0.999999)).astype(np.float32)
        np.save(out_z, z)

    qc["status"] = "ok"
    out_meta.write_text(json.dumps(qc, indent=2, sort_keys=True))
    print(f"WROTE: {out_mat} and {out_meta}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
