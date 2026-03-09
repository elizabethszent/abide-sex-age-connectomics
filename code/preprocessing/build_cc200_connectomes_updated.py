"""
Starting from existing fMRIPrep derivatives, it builds CC200 ROI-to-ROI
functional connectomes using a denoising strategy that is explicit,
reviewable, and closer to current best practice than a raw ROI-correlation
pipeline.

Per run, it:
  - reads fMRIPrep preprocessed BOLD, brain mask, and confounds TSV/JSON
  - computes motion QC from framewise displacement (FD)
  - censors non-steady-state and high-motion volumes (default FD > 0.5 mm)
  - optionally removes short surviving segments after censoring
  - excludes the run if mean FD exceeds the requested subject-level cutoff,
     or if post-scrub duration is too short
  - extracts raw CC200 ROI mean time series
  - denoises the ROI time series with a 32-parameter model by default:
       - 6 motion parameters
       - motion derivatives
       - motion quadratic terms
       - motion derivative quadratic terms
       - WM and CSF signals
       - WM/CSF derivatives
       - WM/CSF quadratic terms
       - WM/CSF derivative quadratic terms
     Optionally, global signal and its expansion terms can be added,
     producing a 36-parameter model.
  - applies joint detrending, Butterworth band-pass filtering, nuisance
     regression, standardization, and scrubbing via nilearn.signal.clean
  - computes Pearson correlation connectomes and optional Fisher-z matrices
  - writes outputs into dataset/cutoff-specific folders:
       ABIDE1/fd_0p2, ABIDE2/fd_0p2, ABIDE12/fd_0p2,
       ABIDE1/fd_0p3, ABIDE2/fd_0p3, ABIDE12/fd_0p3


- Uses fMRIPrep confounds explicitly rather than only for QC.
- Uses a widely used 32P nuisance model by default (no GSR), with GSR as an
  explicit sensitivity option.
- Uses simultaneous cleaning logic from nilearn.signal.clean to avoid naïve,
  artifact-reintroducing modular filtering/regression order.
- Enforces post-scrub minimum duration in seconds rather than only a small
  number of retained TRs.
- Records all choices in subject-level JSON sidecars for auditability.

Required manifest columns
-------------------------
dataset,out_prefix,bold,confounds,atlas,brain_mask,tr

`dataset` must be one of: ABIDE1, ABIDE2

Dependencies
------------
python >= 3.10
numpy, pandas, nibabel, scipy, nilearn

Example
-------
python build_cc200_connectomes_updated.py \
  --manifest connectome_manifest.csv \
  --out-root results/connectomes\
  --subject-fd-cutoffs 0.2 0.3 \
  --vol-fd-cutoff 0.5 \
  --min-time-sec 240 \
  --min-segment-trs 5 \
  --low-pass 0.08 \
  --high-pass 0.01 \
  --fisher-z

Optional GSR sensitivity analysis:
python build_cc200_connectomes_pubgrade.py \
  --manifest connectome_manifest.csv \
  --out-root results/connectomes_pubgrade_gsr \
  --subject-fd-cutoffs 0.2 0.3 \
  --vol-fd-cutoff 0.5 \
  --min-time-sec 240 \
  --min-segment-trs 5 \
  --low-pass 0.08 \
  --high-pass 0.01 \
  --gsr \
  --fisher-z
"""


import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from nilearn.signal import clean



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="CSV with dataset,out_prefix,bold,confounds,atlas,brain_mask,tr")
    p.add_argument("--out-root", required=True, help="Output root directory")

    p.add_argument(
        "--subject-fd-cutoffs",
        nargs="+",
        type=float,
        default=[0.2, 0.3],
        help="Subject-level mean FD exclusion thresholds to run as separate outputs",
    )
    p.add_argument("--vol-fd-cutoff", type=float, default=0.5, help="Volume censoring threshold on FD in mm")
    p.add_argument("--fd-col", default="framewise_displacement")
    p.add_argument("--nonsteady-prefix", default="non_steady_state_outlier")

    p.add_argument("--min-time-sec", type=float, default=240.0, help="Minimum retained post-scrub time in seconds")
    p.add_argument("--min-segment-trs", type=int, default=5, help="Drop surviving segments shorter than this many TRs")
    p.add_argument(
        "--min-pct-keep",
        type=float,
        default=0.0,
        help="Optional minimum percent of total frames retained after censoring (0 disables)",
    )

    p.add_argument("--low-pass", type=float, default=0.08)
    p.add_argument("--high-pass", type=float, default=0.01)
    p.add_argument(
        "--gsr",
        action="store_true",
        help="Add global signal expansion terms, producing a 36-parameter model instead of 32P",
    )
    p.add_argument(
        "--fisher-z",
        action="store_true",
        help="Also save Fisher-z transformed connectomes with zeroed diagonal",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()



def as_abs(pathlike: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.fspath(pathlike))


def cutoff_tag(x: float) -> str:
    return str(x).replace(".", "p")


def load_manifest(path: str | os.PathLike[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["dataset", "out_prefix", "bold", "confounds", "atlas", "brain_mask", "tr"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    df["dataset"] = df["dataset"].astype(str).str.upper()
    bad = sorted(set(df[~df["dataset"].isin(["ABIDE1", "ABIDE2"])]["dataset"].tolist()))
    if bad:
        raise ValueError(f"Manifest dataset values must be ABIDE1 or ABIDE2. Found: {bad}")
    return df


def find_col(df: pd.DataFrame, wanted: str, alternatives: Iterable[str] = ()) -> str | None:
    low = {str(c).lower(): c for c in df.columns}
    if wanted.lower() in low:
        return low[wanted.lower()]
    for alt in alternatives:
        if alt.lower() in low:
            return low[alt.lower()]
    return None


def robust_numeric(series: pd.Series, fill: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(fill).to_numpy(dtype=float)


def first_difference(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("first_difference expects 1D array")
    out = np.empty_like(x)
    out[0] = 0.0
    out[1:] = np.diff(x)
    return out


def expand_terms(x: np.ndarray) -> dict[str, np.ndarray]:
    d = first_difference(x)
    return {
        "orig": x,
        "deriv": d,
        "orig_sq": x * x,
        "deriv_sq": d * d,
    }



@dataclass
class QCResult:
    keep_mask: np.ndarray
    fd: np.ndarray
    mean_fd: float
    max_fd: float
    n_total: int
    n_keep: int
    pct_keep: float
    n_censored_fd: int
    n_censored_nonsteady: int
    n_censored_short_segments: int
    kept_time_sec: float
    drop_run: bool
    drop_reason: str | None
    fd_col_used: str
    nonsteady_cols_found: list[str]


def build_initial_keep_mask(
    confounds_df: pd.DataFrame,
    fd_col: str,
    nonsteady_prefix: str,
    vol_fd_cutoff: float,
    tr: float,
    min_segment_trs: int,
    min_time_sec: float,
    subject_fd_cutoff: float,
    min_pct_keep: float,
) -> QCResult:
    fd_used = find_col(
        confounds_df,
        fd_col,
        alternatives=["fd", "framedisplacement", "framewisedisplacement", "FramewiseDisplacement"],
    )
    if fd_used is None:
        raise ValueError(f"Could not find FD column matching '{fd_col}'")

    fd = robust_numeric(confounds_df[fd_used], fill=0.0)
    if len(fd) == 0:
        raise ValueError("Confounds table is empty")

    n_total = int(len(fd))
    mean_fd = float(np.mean(fd))
    max_fd = float(np.max(fd))

    nonsteady_cols = [c for c in confounds_df.columns if str(c).startswith(nonsteady_prefix)]
    if nonsteady_cols:
        nonsteady_arr = confounds_df[nonsteady_cols].fillna(0).to_numpy()
        nonsteady = (nonsteady_arr != 0).any(axis=1)
    else:
        nonsteady = np.zeros(n_total, dtype=bool)

    high_motion = fd > float(vol_fd_cutoff)
    keep = ~(high_motion | nonsteady)

    #remove short surviving islands after censoring. e.g., < 5 consecutive TRs.
    removed_short = np.zeros_like(keep, dtype=bool)
    if min_segment_trs > 1 and np.any(keep):
        start = None
        for i, flag in enumerate(keep):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                seg_len = i - start
                if seg_len < min_segment_trs:
                    removed_short[start:i] = True
                start = None
        if start is not None:
            seg_len = len(keep) - start
            if seg_len < min_segment_trs:
                removed_short[start:len(keep)] = True
        keep = keep & (~removed_short)

    n_keep = int(np.sum(keep))
    pct_keep = float(100.0 * n_keep / max(1, n_total))
    kept_time_sec = float(n_keep * tr)

    reason = None
    if mean_fd > float(subject_fd_cutoff):
        reason = f"mean_fd>{subject_fd_cutoff}"
    elif kept_time_sec < float(min_time_sec):
        reason = f"kept_time_sec<{min_time_sec}"
    elif min_pct_keep > 0 and pct_keep < float(min_pct_keep):
        reason = f"pct_keep<{min_pct_keep}"

    return QCResult(
        keep_mask=keep,
        fd=fd,
        mean_fd=mean_fd,
        max_fd=max_fd,
        n_total=n_total,
        n_keep=n_keep,
        pct_keep=pct_keep,
        n_censored_fd=int(np.sum(high_motion)),
        n_censored_nonsteady=int(np.sum(nonsteady)),
        n_censored_short_segments=int(np.sum(removed_short)),
        kept_time_sec=kept_time_sec,
        drop_run=reason is not None,
        drop_reason=reason,
        fd_col_used=str(fd_used),
        nonsteady_cols_found=[str(c) for c in nonsteady_cols],
    )



def build_32p_or_36p_confounds(df: pd.DataFrame, gsr: bool = False) -> tuple[pd.DataFrame, list[str]]:
    motion_cols = []
    for name, alts in [
        ("trans_x", []),
        ("trans_y", []),
        ("trans_z", []),
        ("rot_x", []),
        ("rot_y", []),
        ("rot_z", []),
    ]:
        col = find_col(df, name, alts)
        if col is None:
            raise ValueError(f"Missing required motion column: {name}")
        motion_cols.append(col)

    wm_col = find_col(df, "white_matter", ["whitematter", "WhiteMatter"])
    csf_col = find_col(df, "csf", ["CSF"])
    if wm_col is None or csf_col is None:
        raise ValueError("Confounds TSV must contain white_matter and csf for this pipeline")

    selected_names: list[str] = []
    data: dict[str, np.ndarray] = {}

    # 24 motion regressors.
    for base_col in motion_cols:
        x = robust_numeric(df[base_col], fill=0.0)
        terms = expand_terms(x)
        for suffix, arr in terms.items():
            name = f"{base_col}__{suffix}"
            data[name] = arr
            selected_names.append(name)

    # 8 tissue regressors. WM + CSF expansions.
    for base_col in [wm_col, csf_col]:
        x = robust_numeric(df[base_col], fill=0.0)
        terms = expand_terms(x)
        for suffix, arr in terms.items():
            name = f"{base_col}__{suffix}"
            data[name] = arr
            selected_names.append(name)

    #optional. 4 GSR regressors.
    if gsr:
        gs_col = find_col(df, "global_signal", ["GlobalSignal"])
        if gs_col is None:
            raise ValueError("--gsr requested but global_signal column not found")
        x = robust_numeric(df[gs_col], fill=0.0)
        terms = expand_terms(x)
        for suffix, arr in terms.items():
            name = f"{gs_col}__{suffix}"
            data[name] = arr
            selected_names.append(name)

    conf = pd.DataFrame(data)
    conf = conf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return conf, selected_names



#spatial handling and ROI extraction
def resample_label_img_to_bold(atlas_img: nib.Nifti1Image, bold_img: nib.Nifti1Image) -> tuple[nib.Nifti1Image, dict]:
    if atlas_img.shape == bold_img.shape[:3] and np.allclose(atlas_img.affine, bold_img.affine):
        return atlas_img, {"atlas_resampled": False}
    target = (bold_img.shape[:3], bold_img.affine)
    rs = resample_from_to(atlas_img, target, order=0)
    return rs, {
        "atlas_resampled": True,
        "atlas_shape_orig": tuple(int(x) for x in atlas_img.shape),
        "atlas_shape_new": tuple(int(x) for x in rs.shape),
        "atlas_resample_order": 0,
    }


def resample_mask_to_bold(mask_img: nib.Nifti1Image, bold_img: nib.Nifti1Image) -> tuple[nib.Nifti1Image, dict]:
    if mask_img.shape == bold_img.shape[:3] and np.allclose(mask_img.affine, bold_img.affine):
        return mask_img, {"brain_mask_resampled": False}
    target = (bold_img.shape[:3], bold_img.affine)
    rs = resample_from_to(mask_img, target, order=0)
    return rs, {
        "brain_mask_resampled": True,
        "brain_mask_shape_orig": tuple(int(x) for x in mask_img.shape),
        "brain_mask_shape_new": tuple(int(x) for x in rs.shape),
        "brain_mask_resample_order": 0,
    }


def extract_raw_roi_timeseries(
    bold_path: str,
    atlas_path: str,
    brain_mask_path: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    bold_img = nib.load(bold_path)
    atlas_img = nib.load(atlas_path)
    brain_mask_img = nib.load(brain_mask_path)

    if len(bold_img.shape) != 4:
        raise ValueError(f"BOLD must be 4D. Got shape={bold_img.shape}")

    atlas_img, atlas_meta = resample_label_img_to_bold(atlas_img, bold_img)
    brain_mask_img, mask_meta = resample_mask_to_bold(brain_mask_img, bold_img)

    bold = np.asanyarray(bold_img.dataobj).astype(np.float32)
    atlas = np.rint(np.asanyarray(atlas_img.dataobj)).astype(np.int32)
    brain_mask = np.asanyarray(brain_mask_img.dataobj) > 0

    if atlas.shape != bold.shape[:3]:
        raise ValueError(f"Atlas shape {atlas.shape} does not match BOLD spatial shape {bold.shape[:3]}")
    if brain_mask.shape != bold.shape[:3]:
        raise ValueError(f"Brain mask shape {brain_mask.shape} does not match BOLD spatial shape {bold.shape[:3]}")

    labels = np.unique(atlas)
    labels = labels[labels > 0]
    labels = np.sort(labels)
    if labels.size == 0:
        raise ValueError("No non-zero atlas labels found")

    T = int(bold.shape[3])
    bold_2d = bold.reshape(-1, T)
    atlas_flat = atlas.reshape(-1)
    mask_flat = brain_mask.reshape(-1)

    ts = np.zeros((T, int(labels.size)), dtype=np.float32)
    dropped_labels = []
    roi_voxel_counts = {}

    for j, lab in enumerate(labels):
        vox = np.where((atlas_flat == lab) & mask_flat)[0]
        roi_voxel_counts[int(lab)] = int(vox.size)
        if vox.size == 0:
            dropped_labels.append(int(lab))
            continue
        ts[:, j] = bold_2d[vox].mean(axis=0)

    meta = {
        "n_rois": int(labels.size),
        "labels": [int(x) for x in labels.tolist()],
        "roi_voxel_counts": roi_voxel_counts,
        "rois_with_zero_voxels_after_masking": dropped_labels,
    }
    meta.update(atlas_meta)
    meta.update(mask_meta)
    return ts, labels.astype(np.int32), meta



#cleaning and connectomes
def clean_timeseries(
    raw_ts: np.ndarray,
    confounds_df: pd.DataFrame,
    keep_mask: np.ndarray,
    tr: float,
    low_pass: float,
    high_pass: float,
) -> np.ndarray:
    if raw_ts.ndim != 2:
        raise ValueError("raw_ts must have shape (time, rois)")
    if len(confounds_df) != raw_ts.shape[0]:
        raise ValueError(f"Confounds rows ({len(confounds_df)}) != BOLD timepoints ({raw_ts.shape[0]})")
    if keep_mask.shape[0] != raw_ts.shape[0]:
        raise ValueError("keep_mask length mismatch")
    keep_idx = np.where(keep_mask)[0]
    if keep_idx.size == 0:
        raise ValueError("No timepoints retained after censoring")

    cleaned = clean(
        raw_ts,
        detrend=True,
        standardize="zscore_sample",
        sample_mask=keep_idx,
        confounds=confounds_df,
        standardize_confounds=True,
        filter="butterworth",
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=tr,
        ensure_finite=True,
    )
    return np.asarray(cleaned, dtype=np.float32)


def compute_connectome(cleaned_ts: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(cleaned_ts.T).astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def fisher_z(corr: np.ndarray) -> np.ndarray:
    z = np.arctanh(np.clip(corr, -0.999999, 0.999999)).astype(np.float32)
    np.fill_diagonal(z, 0.0)
    return z



def get_output_targets(out_root: Path, dataset: str, subject_fd_cutoff: float, out_prefix: str) -> list[dict[str, Path]]:
    cut_tag = f"fd_{cutoff_tag(subject_fd_cutoff)}"
    targets = []
    for ds in [dataset, "ABIDE12"]:
        base = out_root / ds / cut_tag
        targets.append(
            {
                "dataset_bucket": base,
                "mat": base / "matrices" / f"{out_prefix}.npy",
                "zmat": base / "matrices_z" / f"{out_prefix}_z.npy",
                "meta": base / "metadata" / f"{out_prefix}.json",
            }
        )
    return targets


def ensure_parent_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)



#main

def process_one(
    row: pd.Series,
    subject_fd_cutoff: float,
    out_root: Path,
    vol_fd_cutoff: float,
    fd_col: str,
    nonsteady_prefix: str,
    min_time_sec: float,
    min_segment_trs: int,
    min_pct_keep: float,
    low_pass: float,
    high_pass: float,
    gsr: bool,
    save_fisher_z: bool,
    overwrite: bool,
) -> None:
    dataset = str(row["dataset"]).upper()
    out_prefix = str(row["out_prefix"])
    tr = float(row["tr"])

    targets = get_output_targets(out_root, dataset, subject_fd_cutoff, out_prefix)
    if not overwrite and all(t["meta"].exists() and t["mat"].exists() for t in targets):
        print(f"SKIP [{dataset} fd<{subject_fd_cutoff}] {out_prefix}: outputs already exist")
        return

    confounds_path = as_abs(row["confounds"])
    confounds_df = pd.read_csv(confounds_path, sep="\t")

    qc = build_initial_keep_mask(
        confounds_df=confounds_df,
        fd_col=fd_col,
        nonsteady_prefix=nonsteady_prefix,
        vol_fd_cutoff=vol_fd_cutoff,
        tr=tr,
        min_segment_trs=min_segment_trs,
        min_time_sec=min_time_sec,
        subject_fd_cutoff=subject_fd_cutoff,
        min_pct_keep=min_pct_keep,
    )

    pipeline_name = "32P_noGSR_scrub_butterworth" if not gsr else "36P_GSR_scrub_butterworth"
    meta: dict = {
        "status": "excluded" if qc.drop_run else "ok",
        "reason": qc.drop_reason,
        "dataset": dataset,
        "out_prefix": out_prefix,
        "bold": as_abs(row["bold"]),
        "confounds": confounds_path,
        "atlas": as_abs(row["atlas"]),
        "brain_mask": as_abs(row["brain_mask"]),
        "tr": tr,
        "pipeline": pipeline_name,
        "fd_col_used": qc.fd_col_used,
        "subject_fd_cutoff_mean_fd": float(subject_fd_cutoff),
        "volume_fd_cutoff": float(vol_fd_cutoff),
        "nonsteady_prefix": nonsteady_prefix,
        "nonsteady_cols_found": qc.nonsteady_cols_found,
        "mean_fd": qc.mean_fd,
        "max_fd": qc.max_fd,
        "n_trs_total": qc.n_total,
        "n_trs_keep": qc.n_keep,
        "pct_trs_keep": qc.pct_keep,
        "n_censored_fd": qc.n_censored_fd,
        "n_censored_nonsteady": qc.n_censored_nonsteady,
        "n_censored_short_segments": qc.n_censored_short_segments,
        "kept_time_sec": qc.kept_time_sec,
        "min_time_sec_required": float(min_time_sec),
        "min_segment_trs": int(min_segment_trs),
        "min_pct_keep_required": float(min_pct_keep),
        "low_pass_hz": float(low_pass),
        "high_pass_hz": float(high_pass),
        "gsr": bool(gsr),
    }

    if qc.drop_run:
        for t in targets:
            ensure_parent_dirs([t["meta"]])
            t["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True))
        print(f"EXCLUDE [{dataset} fd<{subject_fd_cutoff}] {out_prefix}: {qc.drop_reason}; mean_fd={qc.mean_fd:.4f}, kept={qc.kept_time_sec:.1f}s")
        return

    nuisance_df, nuisance_names = build_32p_or_36p_confounds(confounds_df, gsr=gsr)
    raw_ts, labels, ts_meta = extract_raw_roi_timeseries(
        bold_path=as_abs(row["bold"]),
        atlas_path=as_abs(row["atlas"]),
        brain_mask_path=as_abs(row["brain_mask"]),
    )
    cleaned_ts = clean_timeseries(
        raw_ts=raw_ts,
        confounds_df=nuisance_df,
        keep_mask=qc.keep_mask,
        tr=tr,
        low_pass=low_pass,
        high_pass=high_pass,
    )
    corr = compute_connectome(cleaned_ts)
    z = fisher_z(corr) if save_fisher_z else None

    meta.update(ts_meta)
    meta.update(
        {
            "confound_columns_used": nuisance_names,
            "n_confounds": int(len(nuisance_names)),
            "n_rois": int(corr.shape[0]),
            "cleaned_n_timepoints": int(cleaned_ts.shape[0]),
            "sample_mask_keep_indices": np.where(qc.keep_mask)[0].tolist(),
            "matrix_shape": [int(x) for x in corr.shape],
        }
    )

    for t in targets:
        ensure_parent_dirs([t["mat"], t["meta"], t["zmat"]])
        np.save(t["mat"], corr)
        if save_fisher_z and z is not None:
            np.save(t["zmat"], z)
        t["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True))

    print(
        f"WROTE [{dataset} fd<{subject_fd_cutoff}] {out_prefix}: "
        f"mean_fd={qc.mean_fd:.4f}, kept={qc.kept_time_sec:.1f}s, rois={corr.shape[0]}, confounds={len(nuisance_names)}"
    )


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for cutoff in args.subject_fd_cutoffs:
        for _, row in manifest.iterrows():
            try:
                process_one(
                    row=row,
                    subject_fd_cutoff=float(cutoff),
                    out_root=out_root,
                    vol_fd_cutoff=float(args.vol_fd_cutoff),
                    fd_col=str(args.fd_col),
                    nonsteady_prefix=str(args.nonsteady_prefix),
                    min_time_sec=float(args.min_time_sec),
                    min_segment_trs=int(args.min_segment_trs),
                    min_pct_keep=float(args.min_pct_keep),
                    low_pass=float(args.low_pass),
                    high_pass=float(args.high_pass),
                    gsr=bool(args.gsr),
                    save_fisher_z=bool(args.fisher_z),
                    overwrite=bool(args.overwrite),
                )
            except Exception as e:
                dataset = str(row.get("dataset", "UNKNOWN"))
                out_prefix = str(row.get("out_prefix", "UNKNOWN"))
                print(f"ERROR [{dataset} fd<{cutoff}] {out_prefix}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
