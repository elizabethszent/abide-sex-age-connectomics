#!/usr/bin/env bash
#SBATCH --partition=cpu2022
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ioannou_lab/%u/abide_data/logs/%x_%A_%a.out
#SBATCH --error=/work/ioannou_lab/%u/abide_data/logs/%x_%A_%a.err

set -euo pipefail

DATASET_IN="${1:?Usage: sbatch --array=... cc200_connectomes_from_s3_array.sbatch <abide1|abide2|abide12>}"
DATASET_LC="$(echo "$DATASET_IN" | tr '[:upper:]' '[:lower:]')"
DATASET_UC="$(echo "$DATASET_IN" | tr '[:lower:]' '[:upper:]')"

LAB="/work/ioannou_lab/$USER"
PBASE="$LAB/abide_data"
REPO="$LAB/abide-sex-age-connectomics"

PY="$PBASE/code/.venv/bin/python"

RUNLIST="$REPO/subjects/${DATASET_LC}_valid_run_prefixes.txt"
ATLAS="$REPO/atlases/cc200/cc200_roi_atlas.nii.gz"
OUTDIR="$REPO/connectomes/CC200/${DATASET_UC}"
BUCKET="fcp-indi"

mkdir -p "$PBASE/logs" "$OUTDIR" "$OUTDIR/qc"

echo "TAG=$DATASET_UC"
echo "RUNLIST=$RUNLIST"
echo "ATLAS=$ATLAS"
echo "OUTDIR=$OUTDIR"
echo "BUCKET=$BUCKET"
echo "TASK_ID=$SLURM_ARRAY_TASK_ID"
"$PY" -V

# fast sanity on node
"$PY" -c "import awscli,nibabel,nilearn; print('deps ok')"

WORKROOT="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
WORKROOT="$WORKROOT/cc200_${DATASET_LC}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORKROOT"
echo "WORKROOT=$WORKROOT"

export VENV_PY="$PY"

OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

"$PY" "$REPO/code/bin/make_cc200_connectome.py" \
  --runlist "$RUNLIST" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --outdir "$OUTDIR" \
  --atlas "$ATLAS" \
  --bucket "$BUCKET" \
  --tmpdir "$WORKROOT" \
  --fd 0.2 0.3 \
  --min-keep 50 \
  $OVERWRITE_FLAG

echo "DONE $DATASET_UC task=$SLURM_ARRAY_TASK_ID"
