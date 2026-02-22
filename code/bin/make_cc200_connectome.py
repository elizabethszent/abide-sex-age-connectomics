#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean


def read_nth_line(path: Path, n_1based: int) -> str:
    with path.open("r") as f:
        for i, line in enumerate(f, start=1):
            if i == n_1based:
                return line.strip()
    raise IndexError(f"Index {n_1based} out of range for {path}")


def aws_cp_no_sign(s3_url: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", s3_url, str(dst)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        return False
    return True


def pick_confounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reasonable, stable confound set for old fMRIPrep (1.x):
      - 36P-ish: motion + (deriv, sq, derivsq) + tissue signals (+same expansions)
      - plus aCompCor/tCompCor (first 5 each if present)
    """
    base = [
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z",
        "global_signal", "white_matter", "csf",
    ]
    suffixes = ["", "_derivative1", "_power2", "_derivative1_power2"]

    cols = []
    for b in base:
        for s in suffixes:
            c = b + s
            if c in df.columns:
                cols.append(c)

    # add compcor components if present
    acomp = sorted([c for c in df.columns if re.match(r"^a_comp_cor_\d+$", c)])
    tcomp = sorted([c for c in df.columns if re.match(r"^t_comp_cor_\d+$", c)])

    cols.extend(acomp[:5])
    cols.extend(tcomp[:5])

    cols = [c for c in cols if c in df.columns]
    conf = df[cols].copy()

    # fill NaNs conservatively
    conf = conf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return conf


def build_keep_mask(df: pd.DataFrame, fd_thr: float) -> np.ndarray:
    n = len(df)
    keep = np.ones(n, dtype=bool)

    # drop non-steady state volumes if present
    nss_cols = [c for c in df.columns if c.startswith("non_steady_state_outlier")]
    for c in nss_cols:
        keep &= (df[c].fillna(0).astype(int).values == 0)

    # censor by FD threshold if available
    if "framewise_displacement" in df.columns:
        fd = df["framewise_displacement"].fillna(0.0).astype(float).values
        keep &= (fd <= float(fd_thr))

    return keep


def corr_and_z(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # ts shape: (T, 200)
    r = np.corrcoef(ts, rowvar=False)
    r = np.clip(r, -0.999999, 0.999999)
    np.fill_diagonal(r, 1.0)
    z = np.arctanh(r)
    np.fill_diagonal(z, 0.0)
    return r.astype(np.float32), z.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runlist", required=True, type=Path)
    ap.add_argument("--index", required=True, type=int, help="1-based line index into runlist")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--atlas", required=True, type=Path)
    ap.add_argument("--bucket", default="fcp-indi")
    ap.add_argument("--tmpdir", default=None)
    ap.add_argument("--fd", nargs="+", type=float, default=[0.2, 0.3])
    ap.add_argument("--min-keep", type=int, default=30)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    qc_dir = outdir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    prefix = read_nth_line(args.runlist, args.index)  # e.g. data/Projects/.../sub-XXXX..._run-1
    run_id = prefix.split("/func/")[-1] if "/func/" in prefix else Path(prefix).name

    # where to download
    tmp_root = Path(args.tmpdir) if args.tmpdir else Path("/tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    dl_dir = tmp_root / f"cc200_{run_id}"
    dl_dir.mkdir(parents=True, exist_ok=True)

    # S3 objects
    preproc_key = f"{prefix}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    conf_key = f"{prefix}_desc-confounds_regressors.tsv"
    mask_key = f"{prefix}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

    preproc_url = f"s3://{args.bucket}/{preproc_key}"
    conf_url = f"s3://{args.bucket}/{conf_key}"
    mask_url = f"s3://{args.bucket}/{mask_key}"

    preproc_path = dl_dir / "preproc_bold.nii.gz"
    conf_path = dl_dir / "confounds.tsv"
    mask_path = dl_dir / "brain_mask.nii.gz"

    qc = {
        "run_id": run_id,
        "prefix": prefix,
        "s3": {"preproc": preproc_url, "confounds": conf_url, "mask": mask_url},
        "fd_thresholds": args.fd,
        "status": "started",
    }

    # outputs (two FD versions)
    out_files = {thr: outdir / f"{run_id}_atlas-CC200_fd-{thr:.1f}_connectome.npz" for thr in args.fd}
    qc_file = qc_dir / f"{run_id}.json"

    # Skip if already done
    if (not args.overwrite) and all(p.exists() for p in out_files.values()) and qc_file.exists():
        return

    # download required inputs
    if not aws_cp_no_sign(preproc_url, preproc_path):
        qc["status"] = "fail_download_preproc"
        qc_file.write_text(json.dumps(qc, indent=2) + "\n")
        return
    if not aws_cp_no_sign(conf_url, conf_path):
        qc["status"] = "fail_download_confounds"
        qc_file.write_text(json.dumps(qc, indent=2) + "\n")
        return
    # mask is nice-to-have; if missing, masker can still work but will be less clean
    have_mask = aws_cp_no_sign(mask_url, mask_path)
    qc["have_mask"] = bool(have_mask)

    # load data
    img = nib.load(str(preproc_path))
    n_vols = img.shape[3] if img.ndim == 4 else 0
    tr = None
    try:
        tr = float(img.header.get_zooms()[3])
        if not np.isfinite(tr) or tr <= 0:
            tr = None
    except Exception:
        tr = None

    df = pd.read_csv(conf_path, sep="\t")
    qc["n_volumes"] = int(n_vols)
    qc["confounds_rows"] = int(len(df))
    qc["tr"] = tr

    if n_vols == 0 or len(df) == 0 or abs(len(df) - n_vols) > 2:
        qc["status"] = "fail_bad_dimensions"
        qc_file.write_text(json.dumps(qc, indent=2) + "\n")
        return

    # align confounds length to data (rare off-by-1 issues)
    m = min(n_vols, len(df))
    if len(df) != m:
        df = df.iloc[:m].reset_index(drop=True)
        img = nib.funcs.four_to_three(img)  # list-like proxy; below we won't index image anyway
        # We’ll handle by trimming timeseries after extraction if needed.

    # set up masker
    atlas_img = nib.load(str(args.atlas))
    mask_img = nib.load(str(mask_path)) if have_mask else None

    # Extract raw timeseries first; do regression + censoring in nilearn.signal.clean
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        mask_img=mask_img,
        resampling_target="data",
        interpolation="nearest",
        standardize=False,
        detrend=False,
    )

    raw_ts = masker.fit_transform(nib.load(str(preproc_path)))
    qc["n_rois"] = int(raw_ts.shape[1])

    if raw_ts.shape[1] != 200:
        qc["status"] = "fail_not_200_rois"
        qc["note"] = f"Expected 200 ROIs, got {raw_ts.shape[1]}"
        qc_file.write_text(json.dumps(qc, indent=2) + "\n")
        return

    conf = pick_confounds(df)
    qc["confounds_used"] = list(conf.columns)

    # If needed, trim to match raw_ts length
    T = raw_ts.shape[0]
    if len(conf) != T:
        m = min(len(conf), T)
        raw_ts = raw_ts[:m, :]
        df = df.iloc[:m].reset_index(drop=True)
        conf = conf.iloc[:m].reset_index(drop=True)
        T = m

    # Compute for each FD threshold
    qc["results"] = {}
    for thr in args.fd:
        keep = build_keep_mask(df, thr)
        kept_idx = np.where(keep)[0]
        qc["results"][str(thr)] = {
            "kept": int(len(kept_idx)),
            "dropped": int(T - len(kept_idx)),
        }

        if len(kept_idx) < args.min_keep:
            qc["results"][str(thr)]["status"] = "too_few_volumes"
            continue

        # filtering defaults (can be tuned later); if TR is unknown, disable filtering safely
        hp = 0.01 if tr else None
        lp = 0.08 if tr else None

        cleaned = clean(
            raw_ts,
            confounds=conf.values,
            detrend=True,
            standardize="zscore_sample",
            low_pass=lp,
            high_pass=hp,
            t_r=tr,
            sample_mask=kept_idx,
        )

        r, z = corr_and_z(cleaned)
        out_path = out_files[thr]
        np.savez_compressed(
            out_path,
            r=r,
            z=z,
            fd_threshold=float(thr),
            tr=(float(tr) if tr else np.nan),
            n_kept=int(len(kept_idx)),
            n_total=int(T),
        )
        qc["results"][str(thr)]["status"] = "ok"
        qc["results"][str(thr)]["out"] = str(out_path)

    qc["status"] = "ok"
    qc_file.write_text(json.dumps(qc, indent=2) + "\n")

    # clean up downloads to avoid eating shared space
    try:
        for p in dl_dir.glob("*"):
            p.unlink(missing_ok=True)
        dl_dir.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()
