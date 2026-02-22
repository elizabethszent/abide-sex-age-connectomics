#!/usr/bin/env bash
#SBATCH --job-name=cc200_conn
#SBATCH --partition=cpu2022
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=/work/ioannou_lab/%u/abide_data/logs/%x_%A_%a.out
#SBATCH --error=/work/ioannou_lab/%u/abide_data/logs/%x_%A_%a.err
#SBATCH --export=NONE

set -euo pipefail

LIST="${1:?Usage: sbatch --array=1-N%K cc200_connectomes_from_s3_array.sbatch <RUN_PREFIX_LIST> <TAG>}"
TAG="${2:?Usage: ... <RUN_PREFIX_LIST> <TAG>}"

LAB="/work/ioannou_lab/$USER"
PBASE="$LAB/abide_data"
REPO="$LAB/abide-sex-age-connectomics"

# tools + python deps + aws live here
source "$PBASE/code/.venv/bin/activate"

S3ROOT="s3://fcp-indi"
ATLAS="$REPO/atlases/cc200/cc200_roi_atlas.nii.gz"
SCRIPT="$REPO/code/bin/make_cc200_connectome.py"

mkdir -p "$PBASE/logs" "$REPO/connectomes/CC200/$TAG"

prefix="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")"
if [[ -z "${prefix:-}" ]]; then
  echo "[ERR] Empty prefix for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

base="${prefix##*/}"
WORKROOT="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$WORKROOT"

bold="$WORKROOT/${base}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
conf="$WORKROOT/${base}_desc-confounds_regressors.tsv"
mask="$WORKROOT/${base}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

echo "TAG=$TAG"
echo "PREFIX=$prefix"
echo "WORKROOT=$WORKROOT"

# download the two required inputs (mask optional)
aws s3 cp --no-sign-request "$S3ROOT/${prefix}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" "$bold"
aws s3 cp --no-sign-request "$S3ROOT/${prefix}_desc-confounds_regressors.tsv" "$conf"
aws s3 cp --no-sign-request "$S3ROOT/${prefix}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" "$mask" || true

OUTDIR="$REPO/connectomes/CC200/$TAG"

for FD in 0.2 0.3; do
  out="$OUTDIR/${base}_atlas-CC200_desc-connectome_fd${FD}.npz"
  if [[ -s "$out" ]]; then
    echo "SKIP exists: $out"
    continue
  fi

  # NOTE: these flags assume your python script uses argparse with these names.
  # If it doesn’t, run: python $SCRIPT -h  and adjust this call.
  python "$SCRIPT" \
    --bold "$bold" \
    --confounds "$conf" \
    --atlas "$ATLAS" \
    --out "$out" \
    --fd-threshold "$FD" \
    ${mask:+--mask "$mask"}
done

echo "DONE $TAG $base"
