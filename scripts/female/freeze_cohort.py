# scripts/freeze_cohort.py
from pathlib import Path
import pandas as pd
import numpy as np

PHENOS = Path("data/processed//Phenotypic_V1_0b_preprocessed1.csv")
CONN_DIR = Path("data/connectomes/cpac/nofilt_noglobal/cc200_z")
OUT = Path("data/female"); OUT.mkdir(exist_ok=True)

phenos = pd.read_csv(PHENOS)
phenos.columns = [c.strip() for c in phenos.columns]

#ensure FILE_ID exists (it usually already does in *_preprocessed1.csv)
if "FILE_ID" not in phenos.columns:
    phenos["SUB_ID"] = phenos["SUB_ID"].astype(int).astype(str).str.zfill(7)
    phenos["FILE_ID"] = phenos["SITE_ID"].astype(str) + "_" + phenos["SUB_ID"]

#females only (ABIDE: 1=male, 2=female)
fem = phenos[phenos["SEX"] == 2].copy()

#has saved connectome?
fem["has_matrix"] = fem["FILE_ID"].apply(lambda s: (CONN_DIR / f"{s}.npy").exists())

#functional QC must be OK (your printout shows many 'fail')
fem["qc_ok"] = fem["qc_func_rater_3"].astype(str).str.upper().eq("OK")

#motion threshold (mean FD); use ≤ 0.30 to start
fem["fd_ok"] = fem["func_mean_fd"].fillna(np.inf) <= 0.30

#final include
included = fem[fem["has_matrix"] & fem["qc_ok"] & fem["fd_ok"]].copy()
excluded = fem[~(fem["has_matrix"] & fem["qc_ok"] & fem["fd_ok"])].copy()
excluded["reason"] = np.where(~excluded["has_matrix"], "no_matrix",
                       np.where(~excluded["qc_ok"], "qc_fail",
                                np.where(~excluded["fd_ok"], "high_motion", "other")))

#save
included_cols = ["FILE_ID","DX_GROUP","AGE_AT_SCAN","func_mean_fd"]
included[included_cols].to_csv(OUT / "female_metadata_included.csv", index=False)
included["FILE_ID"].to_csv(OUT / "female_subjects_included.txt", index=False, header=False)
excluded[["FILE_ID","DX_GROUP","AGE_AT_SCAN","func_mean_fd","qc_func_rater_3","reason"]] \
    .to_csv(OUT / "female_excluded_with_reasons.csv", index=False)

asd_n = (included["DX_GROUP"] == 1).sum()#1 = ASD
ctl_n = (included["DX_GROUP"] == 2).sum()#2 = Control
print("Included:", len(included), "| ASD:", asd_n, "| Control:", ctl_n)
print("Excluded:", len(excluded))
print("\nExclusion reasons:\n", excluded["reason"].value_counts(dropna=False))