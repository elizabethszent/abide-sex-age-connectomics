# scripts/freeze_cohort_male.py
from pathlib import Path
import pandas as pd
import numpy as np

PHENOS = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/Phenotypic_V1_0b_preprocessed1.csv")
CONN_DIR = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/connectomes/cpac/nofilt_noglobal/cc200_z")
OUT = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/male"); OUT.mkdir(exist_ok=True)

phenos = pd.read_csv(PHENOS)
phenos.columns = [c.strip() for c in phenos.columns]

#ensure FILE_ID exists (it usually already does in *_preprocessed1.csv)
if "FILE_ID" not in phenos.columns:
    phenos["SUB_ID"] = phenos["SUB_ID"].astype(int).astype(str).str.zfill(7)
    phenos["FILE_ID"] = phenos["SITE_ID"].astype(str) + "_" + phenos["SUB_ID"]

#males only (ABIDE: 1=male, 2=maleale)
male = phenos[phenos["SEX"] == 1].copy()

#has saved connectome?
male["has_matrix"] = male["FILE_ID"].apply(lambda s: (CONN_DIR / f"{s}.npy").exists())

#functional QC must be OK (your printout shows many 'fail')
male["qc_ok"] = male["qc_func_rater_3"].astype(str).str.upper().eq("OK")

#motion threshold (mean FD); use ≤ 0.30 to start
male["fd_ok"] = male["func_mean_fd"].fillna(np.inf) <= 0.30

#final include
included = male[male["has_matrix"] & male["qc_ok"] & male["fd_ok"]].copy()
excluded = male[~(male["has_matrix"] & male["qc_ok"] & male["fd_ok"])].copy()
excluded["reason"] = np.where(~excluded["has_matrix"], "no_matrix",
                       np.where(~excluded["qc_ok"], "qc_fail",
                                np.where(~excluded["fd_ok"], "high_motion", "other")))

#save
included_cols = ["FILE_ID","DX_GROUP","AGE_AT_SCAN","func_mean_fd"]
included[included_cols].to_csv(OUT / "male_metadata_included.csv", index=False)
included["FILE_ID"].to_csv(OUT / "male_subjects_included.txt", index=False, header=False)
excluded[["FILE_ID","DX_GROUP","AGE_AT_SCAN","func_mean_fd","qc_func_rater_3","reason"]] \
    .to_csv(OUT / "male_excluded_with_reasons.csv", index=False)

asd_n = (included["DX_GROUP"] == 1).sum()#1 = ASD
ctl_n = (included["DX_GROUP"] == 2).sum()#2 = Control
print("Included:", len(included), "| ASD:", asd_n, "| Control:", ctl_n)
print("Excluded:", len(excluded))
print("\nExclusion reasons:\n", excluded["reason"].value_counts(dropna=False))