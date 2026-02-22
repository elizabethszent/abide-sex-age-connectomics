#!/usr/bin/env bash
#SBATCH --job-name=fmriprep_ABIDE1
#SBATCH --partition=cpu2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/work/ioannou_lab/%u/abide/logs/%x_%A_%a.out
#SBATCH --error=/work/ioannou_lab/%u/abide/logs/%x_%A_%a.err
#SBATCH --export=NONE

set -euo pipefail

SITE="${1:?Usage: sbatch --array=... fmriprep_abide1_array.sbatch <SITE>}"
PBASE="/work/ioannou_lab/$USER/abide"

SUBLIST="$PBASE/subjects/abide1_${SITE}_subjects.txt"
BIDS_DIR="$PBASE/raw_bids/ABIDE1/$SITE"
DERIV_DIR="$PBASE/derivatives/fmriprep/ABIDE1/$SITE"
SIF="$PBASE/containers/fmriprep-25.1.4.sif"
LIC="$PBASE/code/licenses/license.txt"
TFHOME="$PBASE/atlases/templateflow"

SUB="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBLIST")"
if [[ -z "${SUB:-}" ]]; then
  echo "[ERR] Empty subject for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
LABEL="${SUB#sub-}"

WORKROOT="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$WORKROOT/work" "$WORKROOT/tmp" "$DERIV_DIR"

echo "SITE=$SITE SUB=$SUB LABEL=$LABEL"
echo "WORKROOT=$WORKROOT"

# Use shared TemplateFlow cache (no internet needed on compute nodes)
mkdir -p "$TFHOME"

/usr/bin/apptainer exec --cleanenv \
  --bind "$BIDS_DIR":/data:ro \
  --bind "$DERIV_DIR":/out \
  --bind "$WORKROOT/work":/work \
  --bind "$WORKROOT/tmp":/tmp \
  --bind "$TFHOME":/templateflow \
  --bind "$LIC":/opt/freesurfer/license.txt:ro \
  --env FS_LICENSE=/opt/freesurfer/license.txt \
  --env TEMPLATEFLOW_HOME=/templateflow \
  --env TMPDIR=/tmp \
  "$SIF" \
  fmriprep /data /out participant \
    --participant-label "$LABEL" \
    --fs-license-file /opt/freesurfer/license.txt \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --work-dir /work \
    --nthreads "$SLURM_CPUS_PER_TASK" \
    --omp-nthreads 2

echo "DONE $SITE $SUB"
