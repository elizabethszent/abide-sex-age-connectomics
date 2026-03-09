#!/bin/bash
#SBATCH --job-name=cc200_manifest
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/work/ioannou_lab/%u/abide-sex-age-connectomics/results/qc/%x_%j.out
#SBATCH --error=/work/ioannou_lab/%u/abide-sex-age-connectomics/results/qc/%x_%j.err

set -euo pipefail

MANIFEST="${1:?need manifest csv}"
OUTROOT="${2:?need output root}"

USER_NAME="${SLURM_JOB_USER:-$USER}"
REPO="/work/ioannou_lab/$USER_NAME/abide-sex-age-connectomics"
PY="$REPO/.venv_clean/bin/python"

cd "$REPO"
mkdir -p "$OUTROOT" "$REPO/results/qc"

echo "PY=$PY"
"$PY" --version
readlink -f "$PY" || true

"$PY" code/preprocessing/build_cc200_connectomes_updated.py \
  --manifest "$MANIFEST" \
  --out-root "$OUTROOT" \
  --subject-fd-cutoffs 0.2 0.3 \
  --vol-fd-cutoff 0.5 \
  --min-time-sec 240 \
  --min-segment-trs 5 \
  --low-pass 0.08 \
  --high-pass 0.01 \
  --fisher-z
