#!/bin/bash
#SBATCH --job-name=cc200_connectomes_from_s3_array
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/work/ioannou_lab/%u/abide_data/logs/%x_%A_%a.out
#SBATCH --error=/work/ioannou_lab/%u/abide_data/logs/%x_%A_%a.err

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

if [[ "$DATASET" == "abide1" ]]; then
  RUNLIST="$REPO/subjects/abide1_valid_run_prefixes.txt"
  S3_PREFIX="s3://fcp-indi/data/Projects/ABIDE/Outputs/fmriprep/fmriprep"
  OUT_DIR="$REPO/connectomes/CC200/ABIDE1"
  META_DIR="$REPO/connectomes/CC200/ABIDE1_meta"
elif [[ "$DATASET" == "abide2" ]]; then
  RUNLIST="$REPO/subjects/abide2_valid_run_prefixes.txt"
  S3_PREFIX="s3://fcp-indi/data/Projects/ABIDE2/Outputs/fmriprep/fmriprep"
  OUT_DIR="$REPO/connectomes/CC200/ABIDE2"
  META_DIR="$REPO/connectomes/CC200/ABIDE2_meta"
else
  echo "ERROR: DATASET must be abide1 or abide2" >&2
  exit 2
fi

mkdir -p "$PBASE/tmp" "$PBASE/logs" "$OUT_DIR" "$META_DIR"

line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUNLIST" | tr -d '\r' | xargs)"
if [[ -z "$line" ]]; then
  echo "WARN: empty line at array index $SLURM_ARRAY_TASK_ID"
  exit 0
fi

# If the runlist line already contains /func/, treat it as an S3 key prefix (no extension).
# Example:
#   data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/sub-28743/ses-1/func/sub-28743_ses-1_task-rest_run-1
if [[ "$line" == *"/func/"* ]]; then
  key_prefix="${line#/}"          # drop leading slash if present
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

BOLD_LOCAL="$PBASE/tmp/${run}_desc-preproc_bold.nii.gz"
CONF_LOCAL="$PBASE/tmp/${run}_desc-confounds_regressors.tsv"

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

rm -f "$BOLD_LOCAL" "$CONF_LOCAL"
