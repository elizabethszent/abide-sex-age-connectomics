# scripts/check_missing_subjects.py

import os, glob, pandas as pd

PHENO = "data/processed/Phenotypic_V1_0b_preprocessed1.csv"
TS_DIR = "data/roi_timeseries"

# Your current choice:
pipeline  = "cpac"
strategy  = "nofilt_noglobal"
atlas     = "rois_cc200"

pheno = pd.read_csv(PHENO)
fem = pheno[pheno["SEX"] == 2].copy()  # females

# all FILE_IDs present for your current choice
files = glob.glob(os.path.join(TS_DIR, pipeline, strategy, atlas, f"*_{atlas}.1D"))
present_ids = {os.path.basename(f).split(f"_{atlas}.1D")[0] for f in files}

fem["has_current"] = fem["FILE_ID"].isin(present_ids)
missing = fem[~fem["has_current"]][["FILE_ID","SITE_ID","SUB_ID","DX_GROUP","AGE_AT_SCAN"]].copy()

print(f"Females in phenotypic: {len(fem)}")
print(f"Females with {pipeline}/{strategy}/{atlas}: {fem['has_current'].sum()}")
print(f"Missing in this combo: {len(missing)}\n")

# Search for each missing ID under any pipeline/strategy/atlas
pipelines = ["cpac","ccs","dparsf","niak"]
strategies = ["nofilt_noglobal","nofilt_global","filt_noglobal","filt_global"]
atlases    = ["rois_cc200","rois_cc400","rois_ho","rois_aal","rois_dosenbach160","rois_tt","rois_ez"]

def find_any(fid):
    for p in pipelines:
        for s in strategies:
            for a in atlases:
                cand = os.path.join(TS_DIR, p, s, a, f"{fid}_{a}.1D")
                if os.path.exists(cand):
                    return cand
    return ""

missing["found_elsewhere"] = [find_any(fid) for fid in missing["FILE_ID"]]
print("Missing females and whether they exist elsewhere:\n")
print(missing.to_string(index=False))
