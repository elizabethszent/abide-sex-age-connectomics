# Pipeline freeze checklist

## 1. Freeze the canonical script
- [ ] Confirm `code/preprocessing/build_cc200_connectomes_updated.py` is the only preprocessing entrypoint used for the main analysis.
- [ ] Record its git commit hash.
- [ ] Save a copy of the frozen config as `docs/methods/preprocessing_config_frozen.yaml`.

## 2. Freeze manifests
- [ ] Create `manifests/connectome_manifest_master.csv` with one row per processed run.
- [ ] Save derived manifests:
  - [ ] `manifests/connectome_manifest_ABIDE1_fd_0p2.csv`
  - [ ] `manifests/connectome_manifest_ABIDE2_fd_0p2.csv`
  - [ ] `manifests/connectome_manifest_ABIDE12_fd_0p2.csv`
  - [ ] `manifests/connectome_manifest_ABIDE1_fd_0p3.csv`
  - [ ] `manifests/connectome_manifest_ABIDE2_fd_0p3.csv`
  - [ ] `manifests/connectome_manifest_ABIDE12_fd_0p3.csv`
- [ ] Do not edit frozen manifests by hand after analysis begins.

## 3. Freeze documentation
- [ ] Save/update `docs/methods/preprocessing_frozen.md`.
- [ ] Add the canonical config path and manifest names to the project README.
- [ ] Record the primary analysis policy: ABIDE12 pooled, FD < 0.2, no GSR.

## 4. Freeze outputs and QC
- [ ] Save per-subject metadata JSONs.
- [ ] Save QC summary tables under `results/qc/`.
- [ ] Save exclusion ledger with reason per subject/run.

## 5. Freeze provenance
- [ ] Record exact atlas path and checksum.
- [ ] Record exact Python environment / package versions.
- [ ] Record git commit hash in QC summary output.
- [ ] Create a git tag for the freeze point.

## 6. Recommended git tag
Example:
`git tag -a preprocess-freeze-v1 -m "Frozen preprocessing for ABIDE sex differences project"`
