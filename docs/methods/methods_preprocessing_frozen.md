# Frozen preprocessing and connectome-building record

## Project
Sex heterogeneity / sex differences in autism spectrum disorder using ABIDE I and ABIDE II resting-state fMRI.

## Canonical preprocessing script
- Script: `code/preprocessing/build_cc200_connectomes_updated.py`
- Frozen config: `docs/methods/preprocessing_config_frozen.yaml`
- First frozen version tag: `v1.0.0`

## Imaging inputs
The pipeline starts from fMRIPrep derivatives in MNI152NLin2009cAsym space. For each run, the required inputs are:
- preprocessed BOLD image: `*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
- brain mask: `*_desc-brain_mask.nii.gz`
- confounds table: `*_desc-confounds_timeseries.tsv` or `*_desc-confounds_regressors.tsv`
- confounds JSON: matching JSON sidecar when available
- CC200 atlas image

## Atlas
- Atlas: CC200
- Atlas path: `atlases/cc200/cc200_roi_atlas.nii.gz`
- Background label: 0
- Atlas is resampled to the BOLD grid using nearest-neighbor interpolation when needed.

## Motion QC and exclusion
### Subject-level thresholds
Two subject-level thresholds are retained:
- main threshold: mean FD < 0.2
- sensitivity threshold: mean FD < 0.3

### Volume-level censoring
Volumes are censored when:
- framewise displacement (FD) > 0.5 mm
- fMRIPrep marks them as non-steady-state outliers

### Post-scrub retention rules
A run is excluded if the retained data after censoring fail either of the following:
- minimum surviving contiguous segment length < 5 TRs
- total retained time < 240 seconds

## Denoising model
Primary model: 32-parameter no-GSR model.

Regressors included:
- 24 motion regressors (6 rigid-body motion parameters, derivatives, squares, squared derivatives)
- white matter signal
- CSF signal
- derivatives and quadratic expansions of white matter and CSF terms
- cosine drift regressors from fMRIPrep

Sensitivity model:
- 36P model with global signal regression (GSR)

## Temporal processing
- detrending: yes
- temporal filtering: band-pass 0.01 to 0.08 Hz
- standardization: yes

## ROI time-series extraction
For each CC200 parcel, the voxelwise mean signal is extracted from the retained timepoints after censoring and denoising.

## Connectivity estimation
- connectivity metric: Pearson correlation
- transform: Fisher z-transform saved in parallel
- diagonal handling: diagonal set to 0 after z-transform for downstream analyses

## Output layout
Connectomes are written under:
- `results/connectomes/ABIDE1/fd_0p2/`
- `results/connectomes/ABIDE2/fd_0p2/`
- `results/connectomes/ABIDE12/fd_0p2/`
- `results/connectomes/ABIDE1/fd_0p3/`
- `results/connectomes/ABIDE2/fd_0p3/`
- `results/connectomes/ABIDE12/fd_0p3/`

Each processed subject/run also receives a QC metadata JSON documenting:
- input file paths
- FD column used
- mean FD
- number of censored volumes
- retained TRs/time
- exclusion reason if excluded
- atlas resampling status
- denoising configuration

## Primary analysis policy
Primary inferential analyses will use:
- dataset: ABIDE12 pooled
- subject-level motion threshold: mean FD < 0.2
- site as a covariate
- age as a covariate
- mean FD as a covariate

Sensitivity analyses will evaluate:
- ABIDE1 and ABIDE2 separately
- mean FD < 0.3 threshold
- GSR vs no-GSR pipelines

## Change-control policy
After freezing, any change to preprocessing choices must:
1. create a new config version
2. update this methods record
3. regenerate manifests/QC summaries
4. be explicitly labeled as a sensitivity or revised primary pipeline
