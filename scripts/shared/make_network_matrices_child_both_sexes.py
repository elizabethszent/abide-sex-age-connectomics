import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")

CONN_DIR    = ROOT / "data/connectomes/cpac/nofilt_noglobal/cc200_z"
MODULES_CSV = ROOT / "data/parcellation/CC200_Louvain7_modules.csv"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


#cohort files: EDIT THESE TO MATCH YOUR ACTUAL FILE NAMES
COHORT_FILES = {
    #key → (input_csv, output_csv_name)
    "female_child": (
        ROOT / "data/female/child_metadata_included.csv",
        RESULTS_DIR / "female_child_network_connectivity_wide.csv",
    ),
    "male_child": (
        ROOT / "data/male/child_metadata_included.csv",
        RESULTS_DIR / "male_child_network_connectivity_wide.csv",
    ),
}

mods_df = pd.read_csv(MODULES_CSV)

if "roi" not in mods_df.columns or "module" not in mods_df.columns:
    raise ValueError("CC200_Louvain7_modules.csv must have columns: roi, module")

mods_df = mods_df.sort_values("roi")
modules = mods_df["module"].to_numpy().astype(int)
n_rois = modules.size
n_modules = modules.max()

print(f"Modules mapping loaded: {n_rois} ROIs, {n_modules} modules")
if n_rois != 200:
    print("WARNING: expected 200 ROIs for CC200, but modules file has", n_rois)

#ROI indices per module
module_indices = {
    m: np.where(modules == m)[0]
    for m in range(1, n_modules + 1)
}


def network_matrix_from_Z(Z, module_indices, n_modules):
    if Z.shape[0] != Z.shape[1]:
        raise ValueError(f"Non-square matrix: {Z.shape}")

    out = {}
    for m1 in range(1, n_modules + 1):
        idx1 = module_indices[m1]
        if len(idx1) == 0:
            continue
        for m2 in range(m1, n_modules + 1):
            idx2 = module_indices[m2]
            if len(idx2) == 0:
                continue

            block = Z[np.ix_(idx1, idx2)]

            if m1 == m2:
                #within-module: use upper triangle (excluding diagonal)
                k = block.shape[0]
                iu = np.triu_indices(k, 1)
                vals = block[iu]
            else:
                #between-module: all cross-edges
                vals = block.ravel()

            vals = vals[np.isfinite(vals)]
            mean_val = float(vals.mean()) if vals.size > 0 else np.nan

            col_name = f"net{m1}_{m2}"
            out[col_name] = mean_val
    return out


#main loop over cohorts female_child, male_child
for label, (cohort_csv, out_csv) in COHORT_FILES.items():
    print(f"\n=== Processing cohort: {label} ===")
    if not cohort_csv.exists():
        print(f"[ERROR] Cohort CSV not found: {cohort_csv}")
        continue

    cohort = pd.read_csv(cohort_csv)
    print(f"N rows in {label} cohort file: {len(cohort)}")

    required_cols = ["FILE_ID", "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd"]
    missing_cols = [c for c in required_cols if c not in cohort.columns]
    if missing_cols:
        raise ValueError(f"{cohort_csv} is missing required columns: {missing_cols}")

    rows = []
    missing = []

    for _, row in cohort.iterrows():
        fid = str(row["FILE_ID"])
        dx  = int(row["DX_GROUP"])
        age = float(row["AGE_AT_SCAN"])
        fd  = float(row["func_mean_fd"])

        npy_path = CONN_DIR / f"{fid}.npy"
        if not npy_path.exists():
            print(f"[WARN] Missing connectome for {label}: {npy_path}")
            missing.append(fid)
            continue

        Z = np.load(npy_path)
        if Z.shape[0] != n_rois:
            print(f"[WARN] {fid}: Z shape {Z.shape} != {n_rois} for {label}")
            missing.append(fid)
            continue

        net_vals = network_matrix_from_Z(Z, module_indices, n_modules)

        out_row = {
            "FILE_ID": fid,
            "DX_GROUP": dx,
            "AGE_AT_SCAN": age,
            "func_mean_fd": fd,
            "sex_group": label,  #just to remember where it came from
        }
        out_row.update(net_vals)
        rows.append(out_row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)

    print(f"Saved {label} network connectivity to:\n  {out_csv}")
    print(f"  Valid subjects: {len(df_out)}")
    if missing:
        print(f"  Subjects skipped due to issues: {len(missing)}")
