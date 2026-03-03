#!/bin/bash
#SBATCH --job-name=cc200_connectomes_from_s3_array
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

DATASET="${1:?Usage: sbatch --array=1-N this.sbatch <abide1|abide2>}"

U="${SLURM_JOB_USER:-${USER:-}}"
[[ -n "$U" ]] || { echo "ERROR: SLURM_JOB_USER/USER not set" >&2; exit 2; }

PBASE="/work/ioannou_lab/$U/abide_data"
REPO="/work/ioannou_lab/$U/abide-sex-age-connectomics"

VENV="$PBASE/code/.venv"
PY="$VENV/bin/python"
AWSCLI=("$PY" "-m" "awscli")
export AWS_EC2_METADATA_DISABLED=true

SCRIPT="$REPO/code/bin/make_cc200_connectome.py"
ATLAS="$REPO/atlases/cc200/cc200_roi_atlas.nii.gz"

[[ -x "$PY" ]] || { echo "ERROR: python not executable at $PY" >&2; exit 2; }
[[ -f "$ATLAS" ]] || { echo "ERROR: atlas not found at $ATLAS" >&2; exit 2; }
[[ -f "$SCRIPT" ]] || { echo "ERROR: script not found at $SCRIPT" >&2; exit 2; }

ROOT="$REPO/outputs/cc200"

if [[ "$DATASET" == "abide1" ]]; then
  RUNLIST="$REPO/subjects/abide1_valid_run_prefixes.txt"
  S3_PREFIX="s3://fcp-indi/data/Projects/ABIDE/Outputs/fmriprep/fmriprep"
  TAG="abide1"
elif [[ "$DATASET" == "abide2" ]]; then
  RUNLIST="$REPO/subjects/abide2_valid_run_prefixes.txt"
  S3_PREFIX="s3://fcp-indi/data/Projects/ABIDE2/Outputs/fmriprep/fmriprep"
  TAG="abide2"
else
  echo "ERROR: DATASET must be abide1 or abide2" >&2
  exit 2
fi

OUT_DIR="$ROOT/$TAG/runs/npy"
META_DIR="$ROOT/$TAG/runs/meta"
mkdir -p "$OUT_DIR" "$META_DIR" "$ROOT/$TAG/logs" "$PBASE/tmp"

line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUNLIST" | tr -d '\r' | xargs)"
if [[ -z "$line" ]]; then
  echo "WARN: empty line at array index $SLURM_ARRAY_TASK_ID"
  exit 0
fi

# Resolve S3 path + run_id
if [[ "$line" == *"/func/"* ]]; then
  # New format: already a key prefix (maybe without s3://)
  key_prefix="${line#/}"
  if [[ "$key_prefix" != s3://* ]]; then
    key_prefix="s3://fcp-indi/$key_prefix"
  fi
  run="$(basename "$key_prefix")"
  s3_dir="$(dirname "$key_prefix")"
else
  # Legacy format: parse run like sub-0050002_task-rest_run-1 (possibly with SITE_ prefix)
  if [[ "$line" =~ (sub-[0-9]+_task-rest_run-[0-9]+) ]]; then
    run="${BASH_REMATCH[1]}"
  else
    echo "ERROR: could not parse run from line: [$line]" >&2
    exit 2
  fi
  subj="${run%%_*}"
  s3_dir="$S3_PREFIX/$subj/func"
fi

BOLD_KEY="$s3_dir/${run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
CONF_KEY="$s3_dir/${run}_desc-confounds_regressors.tsv"

# Per-task scratch to avoid collisions
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_JOB_ID:-0}"
WORKROOT="/tmp/$U/${JOB_ID}/${SLURM_JOB_NAME}_${JOB_ID}_${TASK_ID}"
mkdir -p "$WORKROOT"

BOLD_LOCAL="$WORKROOT/preproc_bold.nii.gz"
CONF_LOCAL="$WORKROOT/confounds.tsv"

"${AWSCLI[@]}" s3 cp "$BOLD_KEY" "$BOLD_LOCAL" --no-sign-request --no-progress
"${AWSCLI[@]}" s3 cp "$CONF_KEY" "$CONF_LOCAL" --no-sign-request --no-progress

OW_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OW_FLAG=(--overwrite)
fi

"$PY" "$SCRIPT" \
  --bold "$BOLD_LOCAL" \
  --confounds "$CONF_LOCAL" \
  --atlas "$ATLAS" \
  --out-dir "$OUT_DIR" \
  --meta-dir "$META_DIR" \
  --out-prefix "$run" \
  --mean-fd-cutoff 0.2 \
  --vol-fd-cutoff 0.5 \
  "${OW_FLAG[@]}"

rm -rf "$WORKROOT"
