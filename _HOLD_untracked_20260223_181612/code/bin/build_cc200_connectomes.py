#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _nonsteady_mask(df):
    cols = [c for c in df.columns if c.startswith("non_steady_state_outlier")]
    if not cols:
        return np.zeros(len(df), dtype=bool)
    return (df[cols].fillna(0).sum(axis=1) > 0).to_numpy()


def _expand_mask(mask, n_before=1, n_after=2):
    # expand True mask by n_before/n_after
    idx = np.where(mask)[0]
    out = mask.copy()
    for i in idx:
        lo = max(0, i - n_before)
        hi = min(len(mask) - 1, i + n_after)
        out[lo:hi + 1] = True
    return out


def _select_confounds(df):
    # Handle old/new naming
    motion_sets = [
        ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"],
        ["X", "Y", "Z", "RotX", "RotY", "RotZ"],
    ]
    motion = []
    for s in motion_sets:
        if all(c in df.columns for c in s):
            motion = s
            break
    if not motion:
        # fallback: take whatever subset exists of common names
        for c in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "X", "Y", "Z", "RotX", "RotY", "RotZ"]:
            if c in df.columns:
                motion.append(c)

    cols = []
    cols += motion

    # derivatives if present
    for c in list(motion):
        d = c + "_derivative1"
        if d in df.columns:
            cols.append(d)

    # tissue/global signals if present
    for c in ["csf", "white_matter", "global_signal", "CSF", "WhiteMatter", "GlobalSignal"]:
        if c in df.columns:
            cols.append(c)

    conf = df[cols].copy() if cols else pd.DataFrame(index=df.index)

    # add squares (24-ish)
    if conf.shape[1] > 0:
        conf2 = conf ** 2
        conf2.columns = [f"{c}_power2" for c in conf.columns]
        conf = pd.concat([conf, conf2], axis=1)

    # add first 5 aCompCor if present
    acomp = sorted([c for c in df.columns if c.startswith("a_comp_cor_")])[:5]
    if acomp:
        conf = pd.concat([conf, df[acomp]], axis=1)

    conf = conf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return conf


def _safe_corr(ts):
    # ts: (T, R)
    ts = np.asarray(ts, dtype=np.float32)
    # avoid NaNs from constant ROIs
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.corrcoef(ts, rowvar=False)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(r, 1.0)
    return r.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bold", required=True)
    ap.add_argument("--confounds", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--dataset", required=True)      # ABIDE1 / ABIDE2
    ap.add_argument("--run-id", required=True)       # basename prefix
    ap.add_argument("--out-root", required=True)     # repo connectomes/CC200
    ap.add_argument("--fd", nargs="+", type=float, default=[0.2, 0.3])
    ap.add_argument("--min-kept", type=int, default=50)
    ap.add_argument("--censor-before", type=int, default=1)
    ap.add_argument("--censor-after", type=int, default=2)
    args = ap.parse_args()

    bold = Path(args.bold)
    confp = Path(args.confounds)
    atlas = Path(args.atlas)
    out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    # Lazy imports so we can still write an error JSON if nilearn/nibabel missing
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
        try:
            from nilearn.maskers import NiftiLabelsMasker
        except Exception:
            from nilearn.input_data import NiftiLabelsMasker
    except Exception as e:
        err = {"error": f"Missing python deps (nibabel/nilearn): {e}"}
        (out_root / f"{args.dataset}_{args.run_id}.error.json").write_text(json.dumps(err, indent=2))
        raise

    # Load confounds
    df = pd.read_csv(confp, sep="\t")
    n_vols_conf = len(df)

    fd_col = _find_col(df, ["framewise_displacement", "FramewiseDisplacement", "FD"])
    fd = df[fd_col].fillna(0).to_numpy() if fd_col else np.zeros(n_vols_conf, dtype=float)

    nss = _nonsteady_mask(df)
    confounds = _select_confounds(df)

    # Load images
    img = nib.load(str(bold))
    atlas_img = nib.load(str(atlas))

    # Match confounds length to image length if needed
    n_vols_img = img.shape[3] if img.ndim == 4 else 1
    n_vols = min(n_vols_img, n_vols_conf)
    if n_vols_img != n_vols_conf:
        # truncate all time series-like vectors to shared length
        df = df.iloc[:n_vols].copy()
        fd = fd[:n_vols]
        nss = nss[:n_vols]
        confounds = confounds.iloc[:n_vols].copy()

    # --- CRITICAL FIX FOR ARC + LABEL ATLASES ---
    # Resample atlas to BOLD grid using nearest-neighbor (order=0)
    # Use first volume as 3D reference grid
    if img.ndim == 4:
        ref_data = np.asanyarray(img.dataobj[..., 0])
        ref_hdr = img.header.copy()
        ref_hdr.set_data_shape(img.shape[:3])
        bold_ref = nib.Nifti1Image(ref_data, img.affine, ref_hdr)
    else:
        bold_ref = img

    atlas_rs = resample_from_to(atlas_img, (bold_ref.shape[:3], bold_ref.affine), order=0)
    atlas_rs = nib.Nifti1Image(atlas_rs.get_fdata().astype(np.int16), atlas_rs.affine, atlas_rs.header)

    labels = np.unique(atlas_rs.get_fdata().astype(int))
    labels = labels[labels > 0]
    n_labels = int(len(labels))

    masker = NiftiLabelsMasker(
        labels_img=atlas_rs,
        standardize=True,
        detrend=True,
        memory=None,
        verbose=0,
    )

    for thr in args.fd:
        censor = (fd > thr) | nss
        censor = _expand_mask(censor, n_before=args.censor_before, n_after=args.censor_after)
        keep_idx = np.where(~censor)[0]

        meta = {
            "dataset": args.dataset,
            "run_id": args.run_id,
            "fd_threshold": float(thr),
            "n_vols_total": int(n_vols),
            "n_vols_kept": int(len(keep_idx)),
            "mean_fd": float(np.mean(fd)) if n_vols else None,
            "fd_col": fd_col,
            "censor_before": int(args.censor_before),
            "censor_after": int(args.censor_after),
            "atlas_n_labels": n_labels,
        }

        out_dir = out_root / args.dataset / f"fd{thr:.1f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / f"{args.run_id}.npz"
        out_json = out_dir / f"{args.run_id}.json"

        if len(keep_idx) < args.min_kept:
            meta["status"] = "excluded_too_few_volumes"
            out_json.write_text(json.dumps(meta, indent=2))
            continue

        if n_labels != 200:
            meta["status"] = "excluded_wrong_roi_count"
            out_json.write_text(json.dumps(meta, indent=2))
            continue

        ts = masker.fit_transform(img, confounds=confounds, sample_mask=keep_idx)

        ts = np.asarray(ts, dtype=np.float32)
        r = _safe_corr(ts)
        z = np.arctanh(np.clip(r, -0.999999, 0.999999)).astype(np.float32)

        meta["status"] = "ok"
        meta["n_rois"] = int(r.shape[0])

        np.savez_compressed(out_npz, r=r, z=z)
        out_json.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
