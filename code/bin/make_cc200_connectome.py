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


def norm_bucket(b: str) -> str:
    b = (b or "").strip()
    if b.startswith("s3://"):
        b = b[len("s3://"):]
    return b.strip("/")


def aws_cp_no_sign(venv_python: str, bucket: str, key: str, dst: Path) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    bucket = norm_bucket(bucket)
    s3_url = f"s3://{bucket}/{key}"

    print(f"DOWNLOADING: {s3_url} -> {dst}", flush=True)
    cmd = [venv_python, "-m", "awscli", "s3", "cp",
           "--no-sign-request", "--no-progress", s3_url, str(dst)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        err = (r.stderr.strip()[-1200:] if r.stderr else "awscli failed")
        print(f"DOWNLOAD FAILED: {s3_url}", flush=True)
        return False, err

    print(f"DOWNLOADED: {dst} ({dst.stat().st_size/1e6:.1f} MB)", flush=True)
    return True, ""


def pick_confounds(df: pd.DataFrame) -> pd.DataFrame:
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
        for c in ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z","X","Y","Z","RotX","RotY","RotZ"]:
            if c in df.columns:
                motion.append(c)

    cols = list(motion)

    for c in list(motion):
        d = c + "_derivative1"
        if d in df.columns:
            cols.append(d)

    for c in ["csf","white_matter","global_signal","CSF","WhiteMatter","GlobalSignal"]:
        if c in df.columns:
            cols.append(c)

    conf = df[cols].copy() if cols else pd.DataFrame(index=df.index)

    if conf.shape[1] > 0:
        conf2 = conf ** 2
        conf2.columns = [f"{c}_power2" for c in conf.columns]
        conf = pd.concat([conf, conf2], axis=1)

    acomp = sorted([c for c in df.columns if c.startswith("a_comp_cor_")])[:5]
    if acomp:
        conf = pd.concat([conf, df[acomp]], axis=1)

    conf = conf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return conf


def nonsteady_mask(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("non_steady_state_outlier")]
    if not cols:
        return np.zeros(len(df), dtype=bool)
    return (df[cols].fillna(0).sum(axis=1) > 0).to_numpy()


def expand_mask(mask: np.ndarray, n_before=1, n_after=2) -> np.ndarray:
    idx = np.where(mask)[0]
    out = mask.copy()
    for i in idx:
        lo = max(0, i - n_before)
        hi = min(len(mask) - 1, i + n_after)
        out[lo:hi + 1] = True
    return out


def corr_and_z(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.corrcoef(ts, rowvar=False)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.clip(r, -0.999999, 0.999999)
    np.fill_diagonal(r, 1.0)

    z = np.arctanh(r)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(z, 0.0)
    return r.astype(np.float32), z.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runlist", required=True, type=Path)
    ap.add_argument("--index", required=True, type=int, help="1-based line index into runlist")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--atlas", required=True, type=Path)
    ap.add_argument("--bucket", default="fcp-indi")  # just "fcp-indi" (no s3://)
    ap.add_argument("--tmpdir", default=None)
    ap.add_argument("--fd", nargs="+", type=float, default=[0.2, 0.3])
    ap.add_argument("--min-keep", type=int, default=50)
    ap.add_argument("--censor-before", type=int, default=1)
    ap.add_argument("--censor-after", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    qc_dir = outdir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    prefix = read_nth_line(args.runlist, args.index)
    run_id = prefix.split("/func/")[-1] if "/func/" in prefix else Path(prefix).name

    tmp_root = Path(args.tmpdir) if args.tmpdir else Path("/tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    dl_dir = tmp_root / f"cc200_{run_id}"
    dl_dir.mkdir(parents=True, exist_ok=True)

    preproc_key = f"{prefix}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    conf_key    = f"{prefix}_desc-confounds_regressors.tsv"
    mask_key    = f"{prefix}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

    preproc_path = dl_dir / "preproc_bold.nii.gz"
    conf_path    = dl_dir / "confounds.tsv"
    mask_path    = dl_dir / "brain_mask.nii.gz"

    venv_python = os.environ.get("VENV_PY") or os.path.realpath(__import__("sys").executable)

    qc = {
        "run_id": run_id,
        "prefix": prefix,
        "bucket": norm_bucket(args.bucket),
        "keys": {"preproc": preproc_key, "confounds": conf_key, "mask": mask_key},
        "fd_thresholds": args.fd,
        "status": "started",
    }

    out_files = {thr: outdir / f"{run_id}_atlas-CC200_fd-{thr:.1f}_connectome.npz" for thr in args.fd}
    qc_file = qc_dir / f"{run_id}.json"

    if (not args.overwrite) and all(p.exists() for p in out_files.values()) and qc_file.exists():
        return

    ok, err = aws_cp_no_sign(venv_python, args.bucket, preproc_key, preproc_path)
    if not ok:
        qc["status"] = "fail_download_preproc"
        qc["aws_err"] = err
        qc_file.write_text(json.dumps(qc, indent=2) + "\n")
        return

    ok, err = aws_cp_no_sign(venv_python, args.bucket, conf_key, conf_path)
    if not ok:
        qc["status"] = "fail_download_confounds"
        qc["aws_err"] = err
        qc_file.write_text(json.dumps(qc, indent=2) + "\n")
        return

    ok_mask, _ = aws_cp_no_sign(venv_python, args.bucket, mask_key, mask_path)
    qc["have_mask"] = bool(ok_mask)

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

    m = min(n_vols, len(df))
    if len(df) != m:
        df = df.iloc[:m].reset_index(drop=True)

    atlas_img = nib.load(str(args.atlas))
    mask_img = nib.load(str(mask_path)) if ok_mask else None

    # IMPORTANT: no interpolation= (older nilearn doesn't accept it)
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        mask_img=mask_img,
        resampling_target="data",
        standardize=False,
        detrend=False,
        verbose=0,
    )

    raw_ts = masker.fit_transform(img)
    qc["n_rois"] = int(raw_ts.shape[1])

    if raw_ts.shape[1] != 200:
        qc["status"] = "fail_not_200_rois"
        qc["note"] = f"Expected 200 ROIs, got {raw_ts.shape[1]}"
        qc_file.write_text(json.dumps(qc, indent=2) + "\n")
        return

    conf = pick_confounds(df)
    qc["confounds_used"] = list(conf.columns)

    T = raw_ts.shape[0]
    if len(conf) != T:
        mm = min(len(conf), T)
        raw_ts = raw_ts[:mm, :]
        df = df.iloc[:mm].reset_index(drop=True)
        conf = conf.iloc[:mm].reset_index(drop=True)
        T = mm

    fd = df["framewise_displacement"].fillna(0.0).to_numpy() if "framewise_displacement" in df.columns else np.zeros(T)
    nss = nonsteady_mask(df)

    qc["results"] = {}
    for thr in args.fd:
        censor = (fd > thr) | nss
        censor = expand_mask(censor, n_before=args.censor_before, n_after=args.censor_after)
        kept_idx = np.where(~censor)[0]

        qc["results"][str(thr)] = {"kept": int(len(kept_idx)), "dropped": int(T - len(kept_idx))}
        if len(kept_idx) < args.min_keep:
            qc["results"][str(thr)]["status"] = "too_few_volumes"
            continue

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

    # cleanup local tmp
    try:
        for p in dl_dir.glob("*"):
            p.unlink()
        dl_dir.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()
