#!/usr/bin/env bash
set -euo pipefail

SITE="${1:?Usage: download_abide1_site.sh <SITE>}"
PBASE="${PBASE:-/work/ioannou_lab/$USER/abide}"
BIDS_DIR="$PBASE/raw_bids/ABIDE1/$SITE"

mkdir -p "$BIDS_DIR" "$PBASE/subjects"

echo "=== Syncing ABIDE1 site: $SITE ==="
echo "DEST=$BIDS_DIR"
aws s3 sync "s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/$SITE/" "$BIDS_DIR/" \
  --no-sign-request --only-show-errors

echo "=== Fixing/creating BIDS JSON sidecars (TR + AcquisitionDuration cleanup) ==="
python3 "$PBASE/code/bin/fix_bids_json.py" "$BIDS_DIR"

# (Optional) silence any leftover *.bak if they exist from earlier experiments
if compgen -G "$BIDS_DIR/*.bak" > /dev/null; then
  echo "Found *.bak in site root; moving them to $PBASE/logs/bak/"
  mkdir -p "$PBASE/logs/bak/$SITE"
  mv "$BIDS_DIR"/*.bak "$PBASE/logs/bak/$SITE/" || true
fi

echo "=== Writing subject list ==="
find "$BIDS_DIR" -maxdepth 1 -type d -name "sub-*" -printf "%f\n" | sort \
  > "$PBASE/subjects/abide1_${SITE}_subjects.txt"

echo "Wrote $(wc -l < "$PBASE/subjects/abide1_${SITE}_subjects.txt") subjects to:"
echo "$PBASE/subjects/abide1_${SITE}_subjects.txt"
