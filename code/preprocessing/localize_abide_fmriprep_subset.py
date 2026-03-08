#!/usr/bin/env python3
"""
Localize the minimum fMRIPrep derivatives needed for CC200 connectome building
from ABIDE/ABIDE2 public S3 into a stable local derivatives tree.

Local layout produced:
  <out_root>/<DATASET>/<SITE>/<SUBJECT>[/<SESSION>]/func/<files>

Required downloads per run:
  - *_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
  - matching brain mask (space-qualified preferred, fallback unqualified)
  - matching confounds TSV (timeseries preferred, fallback regressors)
Optional:
  - matching confounds JSON
  - preproc BOLD JSON

This script is intentionally conservative and writes an index CSV describing what
was localized and what is missing.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

ABIDE1_S3_ROOT = "s3://fcp-indi/data/Projects/ABIDE/Outputs/fmriprep/fmriprep"
ABIDE2_S3_ROOT = "s3://fcp-indi/data/Projects/ABIDE2/Outputs/fmriprep/fmriprep"
SPACE = "MNI152NLin2009cAsym"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["ABIDE1", "ABIDE2"])
    p.add_argument("--runlist", required=True, help="Text file with one run prefix per line")
    p.add_argument("--out-root", required=True, help="Root like /work/.../abide_data/derivatives/fmriprep")
    p.add_argument("--phenotypes", required=True, help="Combined phenotype CSV used to map subject -> site")
    p.add_argument("--aws-python", default="", help="Path to python that can run 'python -m awscli'")
    p.add_argument("--space", default=SPACE)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--index-csv", default="", help="Where to write localization index CSV")
    return p.parse_args()


def aws_base_cmd(aws_python: str) -> List[str]:
    if aws_python:
        return [aws_python, "-m", "awscli"]
    return ["aws"]


def run_cmd(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


def s3_exists(base_cmd: List[str], s3_uri: str) -> bool:
    proc = run_cmd(base_cmd + ["s3", "ls", s3_uri, "--no-sign-request"])
    return proc.returncode == 0 and bool(proc.stdout.strip())


def s3_copy(base_cmd: List[str], src: str, dst: Path, dry_run: bool = False) -> bool:
    if dry_run:
        print(f"DRYRUN cp {src} -> {dst}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = run_cmd(base_cmd + ["s3", "cp", src, str(dst), "--no-sign-request", "--no-progress"])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return False
    return True


def normalize_subject(value: str) -> str:
    s = str(value).strip()
    m = re.search(r"sub-([A-Za-z0-9]+)", s)
    if m:
        return f"sub-{m.group(1)}"
    s = s.replace(" ", "")
    if s:
        return f"sub-{s.removeprefix('sub-')}"
    return s


def load_site_map(phenotypes_csv: str) -> Dict[str, str]:
    if pd is None:
        raise RuntimeError("pandas is required for phenotype parsing")
    df = pd.read_csv(phenotypes_csv)
    cols = {c.lower(): c for c in df.columns}

    subj_candidates = [
        "file_id", "sub_id", "subject_id", "participant_id", "subject", "subid", "sub"
    ]
    site_candidates = ["site_id", "site", "site_name", "scanner_site"]

    subj_col = next((cols[c] for c in subj_candidates if c in cols), None)
    site_col = next((cols[c] for c in site_candidates if c in cols), None)
    if subj_col is None or site_col is None:
        raise ValueError(
            f"Could not identify subject/site columns in {phenotypes_csv}. Columns were: {list(df.columns)}"
        )

    site_map: Dict[str, str] = {}
    for _, row in df[[subj_col, site_col]].dropna().iterrows():
        subj = normalize_subject(row[subj_col])
        site = str(row[site_col]).strip().replace(" ", "_")
        if subj and site and subj not in site_map:
            site_map[subj] = site
        raw = str(row[subj_col]).strip()
        if raw and raw not in site_map:
            site_map[raw] = site
        no_prefix = subj.removeprefix("sub-")
        if no_prefix and no_prefix not in site_map:
            site_map[no_prefix] = site
    return site_map


RUN_RE = re.compile(r"(sub-[A-Za-z0-9]+(?:_ses-[A-Za-z0-9]+)?_task-rest_run-[A-Za-z0-9]+)")
SUBJ_RE = re.compile(r"(sub-[A-Za-z0-9]+)")
SES_RE = re.compile(r"(ses-[A-Za-z0-9]+)")
RUNID_RE = re.compile(r"(run-[A-Za-z0-9]+)")


def parse_runlist_line(line: str, dataset: str) -> Tuple[str, str, Optional[str], str, Optional[str]]:
    """
    Returns (s3_dir, run_base, session, subject_id, site_hint)
    """
    line = line.strip().strip("/")
    if not line:
        raise ValueError("Empty runlist line")

    site_hint: Optional[str] = None

    if "/func/" in line:
        key_prefix = line
        if not key_prefix.startswith("s3://"):
            key_prefix = f"s3://fcp-indi/{key_prefix}"
        run_base = Path(key_prefix).name
        s3_dir = str(Path(key_prefix).parent)
        m_site = re.match(r"([A-Za-z0-9_]+)_sub-", run_base)
        if m_site:
            site_hint = m_site.group(1)
    else:
        m = RUN_RE.search(line)
        if not m:
            raise ValueError(f"Could not parse run base from line: {line}")
        run_base = m.group(1)
        pre = line[: m.start()].rstrip("_")
        if pre:
            site_hint = pre
        subj = SUBJ_RE.search(run_base)
        if not subj:
            raise ValueError(f"Could not parse subject from run base: {run_base}")
        subject_id = subj.group(1)
        ses = SES_RE.search(run_base)
        root = ABIDE1_S3_ROOT if dataset == "ABIDE1" else ABIDE2_S3_ROOT
        if ses:
            s3_dir = f"{root}/{subject_id}/{ses.group(1)}/func"
        else:
            s3_dir = f"{root}/{subject_id}/func"

    subj = SUBJ_RE.search(run_base)
    if not subj:
        raise ValueError(f"Could not parse subject from run base: {run_base}")
    subject_id = subj.group(1)
    ses = SES_RE.search(run_base)
    session = ses.group(1) if ses else None
    return s3_dir, run_base, session, subject_id, site_hint


def first_existing_s3(base_cmd: List[str], candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if s3_exists(base_cmd, c):
            return c
    return None


def local_dst_func_dir(out_root: Path, dataset: str, site: str, subject_id: str, session: Optional[str]) -> Path:
    p = out_root / dataset / site / subject_id
    if session:
        p = p / session
    return p / "func"


def localize_one(
    base_cmd: List[str],
    dataset: str,
    site_map: Dict[str, str],
    out_root: Path,
    space: str,
    line: str,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, str]:
    s3_dir, run_base, session, subject_id, site_hint = parse_runlist_line(line, dataset)
    run_id_match = RUNID_RE.search(run_base)
    run_id = run_id_match.group(1) if run_id_match else "run-1"
    site = site_hint or site_map.get(subject_id) or site_map.get(subject_id.removeprefix("sub-")) or "UNKNOWN"
    site = site.replace(" ", "_")
    dst_dir = local_dst_func_dir(out_root, dataset, site, subject_id, session)

    bold_name = f"{run_base}_space-{space}_desc-preproc_bold.nii.gz"
    bold_json_name = f"{run_base}_space-{space}_desc-preproc_bold.json"
    mask_candidates = [
        f"{run_base}_space-{space}_desc-brain_mask.nii.gz",
        f"{run_base}_desc-brain_mask.nii.gz",
    ]
    conf_tsv_candidates = [
        f"{run_base}_desc-confounds_timeseries.tsv",
        f"{run_base}_desc-confounds_regressors.tsv",
    ]
    conf_json_candidates = [
        f"{run_base}_desc-confounds_timeseries.json",
        f"{run_base}_desc-confounds_regressors.json",
    ]

    bold_src = f"{s3_dir}/{bold_name}"
    if not s3_exists(base_cmd, bold_src):
        raise FileNotFoundError(f"Missing required BOLD on S3: {bold_src}")

    mask_src = first_existing_s3(base_cmd, (f"{s3_dir}/{n}" for n in mask_candidates))
    conf_src = first_existing_s3(base_cmd, (f"{s3_dir}/{n}" for n in conf_tsv_candidates))
    conf_json_src = first_existing_s3(base_cmd, (f"{s3_dir}/{n}" for n in conf_json_candidates))
    bold_json_src = first_existing_s3(base_cmd, (f"{s3_dir}/{bold_json_name}",))

    if mask_src is None:
        raise FileNotFoundError(f"Missing required brain mask for run base {run_base} under {s3_dir}")
    if conf_src is None:
        raise FileNotFoundError(f"Missing required confounds TSV for run base {run_base} under {s3_dir}")

    targets = [
        (bold_src, dst_dir / Path(bold_src).name, True),
        (mask_src, dst_dir / Path(mask_src).name, True),
        (conf_src, dst_dir / Path(conf_src).name, True),
        (conf_json_src, dst_dir / Path(conf_json_src).name, False) if conf_json_src else None,
        (bold_json_src, dst_dir / Path(bold_json_src).name, False) if bold_json_src else None,
    ]

    for item in targets:
        if item is None:
            continue
        src, dst, required = item
        if dst.exists() and not overwrite:
            continue
        ok = s3_copy(base_cmd, src, dst, dry_run=dry_run)
        if required and not ok:
            raise RuntimeError(f"Failed copying required file {src} -> {dst}")

    out_prefix = f"{site}_{subject_id}"
    if session:
        out_prefix += f"_{session}"
    out_prefix += f"_{run_id}"

    return {
        "dataset": dataset,
        "site": site,
        "subject_id": subject_id,
        "session": session or "",
        "run_id": run_id,
        "run_base": run_base,
        "out_prefix": out_prefix,
        "local_func_dir": str(dst_dir),
        "bold": str(dst_dir / Path(bold_src).name),
        "brain_mask": str(dst_dir / Path(mask_src).name),
        "confounds": str(dst_dir / Path(conf_src).name),
        "confounds_json": str(dst_dir / Path(conf_json_src).name) if conf_json_src else "",
        "bold_json": str(dst_dir / Path(bold_json_src).name) if bold_json_src else "",
        "source_kind": "local",
        "source_uri": s3_dir + "/",
        "status": "ok",
    }


def main() -> int:
    args = parse_args()
    base_cmd = aws_base_cmd(args.aws_python)
    site_map = load_site_map(args.phenotypes)
    out_root = Path(args.out_root)
    runlist = Path(args.runlist)

    rows: List[Dict[str, str]] = []
    with runlist.open() as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip().strip("\r")
            if not line:
                continue
            try:
                row = localize_one(
                    base_cmd=base_cmd,
                    dataset=args.dataset,
                    site_map=site_map,
                    out_root=out_root,
                    space=args.space,
                    line=line,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
                print(f"[{i}] OK  {row['out_prefix']}")
            except Exception as e:
                print(f"[{i}] ERR {line} :: {e}", file=sys.stderr)
                row = {
                    "dataset": args.dataset,
                    "site": "",
                    "subject_id": "",
                    "session": "",
                    "run_id": "",
                    "run_base": "",
                    "out_prefix": "",
                    "local_func_dir": "",
                    "bold": "",
                    "brain_mask": "",
                    "confounds": "",
                    "confounds_json": "",
                    "bold_json": "",
                    "source_kind": "",
                    "source_uri": line,
                    "status": f"error: {e}",
                }
            rows.append(row)

    index_csv = Path(args.index_csv) if args.index_csv else out_root / f"localized_{args.dataset.lower()}_index.csv"
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "site", "subject_id", "session", "run_id", "run_base", "out_prefix",
        "local_func_dir", "bold", "brain_mask", "confounds", "confounds_json", "bold_json",
        "source_kind", "source_uri", "status",
    ]
    with index_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote index: {index_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
