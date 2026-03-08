#!/usr/bin/env python3
"""
Build the frozen connectome master manifest by scanning a stable local fMRIPrep
subset tree created under:
  <deriv_root>/<DATASET>/<SITE>/<SUBJECT>[/<SESSION>]/func/

The manifest includes exact local file paths and the TR read from the NIfTI
header (with JSON fallback when possible).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib

SPACE = "MNI152NLin2009cAsym"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-root", required=True, help="Root like /work/.../abide_data/derivatives/fmriprep")
    p.add_argument("--atlas", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--space", default=SPACE)
    p.add_argument("--datasets", nargs="*", default=["ABIDE1", "ABIDE2"])
    p.add_argument("--skip-incomplete", action="store_true")
    p.add_argument("--missing-report", default="")
    return p.parse_args()


RUNID_RE = re.compile(r"(run-[A-Za-z0-9]+)")


def candidate_match(func_dir: Path, run_base: str, space: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    mask = None
    for name in [
        f"{run_base}_space-{space}_desc-brain_mask.nii.gz",
        f"{run_base}_desc-brain_mask.nii.gz",
    ]:
        p = func_dir / name
        if p.exists():
            mask = p
            break

    conf = None
    for name in [
        f"{run_base}_desc-confounds_timeseries.tsv",
        f"{run_base}_desc-confounds_regressors.tsv",
    ]:
        p = func_dir / name
        if p.exists():
            conf = p
            break

    conf_json = None
    for name in [
        f"{run_base}_desc-confounds_timeseries.json",
        f"{run_base}_desc-confounds_regressors.json",
    ]:
        p = func_dir / name
        if p.exists():
            conf_json = p
            break

    return mask, conf, conf_json


def get_tr(bold_nii: Path) -> float:
    try:
        img = nib.load(str(bold_nii))
        zooms = img.header.get_zooms()
        if len(zooms) >= 4:
            return float(zooms[3])
    except Exception:
        pass

    json_sidecar = bold_nii.with_suffix("")
    json_sidecar = Path(str(json_sidecar).replace(".nii", ".json"))
    if json_sidecar.exists():
        with json_sidecar.open() as f:
            data = json.load(f)
        if "RepetitionTime" in data:
            return float(data["RepetitionTime"])

    raise RuntimeError(f"Could not determine TR for {bold_nii}")


def source_uri_for(dataset: str, subject_id: str, session: str) -> str:
    if dataset == "ABIDE1":
        return f"s3://fcp-indi/data/Projects/ABIDE/Outputs/fmriprep/fmriprep/{subject_id}/func/"
    if session:
        return f"s3://fcp-indi/data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/{subject_id}/{session}/func/"
    return f"s3://fcp-indi/data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/{subject_id}/func/"


def main() -> int:
    args = parse_args()
    deriv_root = Path(args.deriv_root)
    atlas = Path(args.atlas).resolve()
    out_csv = Path(args.out_csv)
    missing_rows: List[Dict[str, str]] = []
    rows: List[Dict[str, str]] = []

    for dataset in args.datasets:
        ds_root = deriv_root / dataset
        if not ds_root.exists():
            print(f"WARN: dataset root missing: {ds_root}", file=sys.stderr)
            continue
        pattern = f"**/*_space-{args.space}_desc-preproc_bold.nii.gz"
        for bold in sorted(ds_root.glob(pattern)):
            rel = bold.relative_to(ds_root)
            parts = rel.parts
            if len(parts) < 4:
                print(f"WARN: unexpected path layout, skipping {bold}", file=sys.stderr)
                continue
            site = parts[0]
            subject_id = parts[1]
            idx = 2
            session = ""
            if parts[idx].startswith("ses-"):
                session = parts[idx]
                idx += 1
            if parts[idx] != "func":
                print(f"WARN: expected func directory, skipping {bold}", file=sys.stderr)
                continue
            func_dir = bold.parent
            suffix = f"_space-{args.space}_desc-preproc_bold.nii.gz"
            run_base = bold.name[: -len(suffix)]
            run_match = RUNID_RE.search(run_base)
            run_id = run_match.group(1) if run_match else "run-1"
            out_prefix = f"{site}_{subject_id}"
            if session:
                out_prefix += f"_{session}"
            out_prefix += f"_{run_id}"

            brain_mask, confounds, confounds_json = candidate_match(func_dir, run_base, args.space)
            missing = []
            if brain_mask is None:
                missing.append("brain_mask")
            if confounds is None:
                missing.append("confounds")
            try:
                tr = get_tr(bold)
            except Exception as e:
                tr = None
                missing.append(f"tr:{e}")

            if missing:
                missing_rows.append({
                    "dataset": dataset,
                    "site": site,
                    "subject_id": subject_id,
                    "session": session,
                    "run_id": run_id,
                    "bold": str(bold.resolve()),
                    "missing": ";".join(missing),
                })
                if args.skip_incomplete:
                    continue

            rows.append({
                "dataset": dataset,
                "site": site,
                "subject_id": subject_id,
                "session": session,
                "run_id": run_id,
                "out_prefix": out_prefix,
                "bold": str(bold.resolve()),
                "brain_mask": str(brain_mask.resolve()) if brain_mask else "",
                "confounds": str(confounds.resolve()) if confounds else "",
                "confounds_json": str(confounds_json.resolve()) if confounds_json else "",
                "atlas": str(atlas),
                "tr": f"{tr:.6f}" if tr is not None else "",
                "source_kind": "local",
                "source_uri": source_uri_for(dataset, subject_id, session),
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "site", "subject_id", "session", "run_id", "out_prefix",
        "bold", "brain_mask", "confounds", "confounds_json", "atlas", "tr",
        "source_kind", "source_uri",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote master manifest: {out_csv} ({len(rows)} rows)")

    if missing_rows:
        missing_report = Path(args.missing_report) if args.missing_report else out_csv.with_name(out_csv.stem + "_missing.csv")
        with missing_report.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "site", "subject_id", "session", "run_id", "bold", "missing"])
            writer.writeheader()
            writer.writerows(missing_rows)
        print(f"Wrote missing report: {missing_report} ({len(missing_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
