import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

#load the per-subject weighted results you already made
W = pd.read_csv("results/weighted_clustering_subjects.csv")  


if "group" not in W.columns:
    Mgrp = pd.read_csv("data/female/metrics_merged.csv")[["FILE_ID", "DX_GROUP"]]
    Mgrp["group"] = Mgrp["DX_GROUP"].map({1: "ASD", 2: "Control"})
    W = W.merge(Mgrp[["FILE_ID", "group"]], on="FILE_ID", how="left")

#bring in covariates
Mcov = pd.read_csv("data/female/metrics_merged.csv")
cols = ["FILE_ID", "AGE_AT_SCAN", "func_mean_fd"]
if "SITE_ID" in Mcov.columns:
    cols.append("SITE_ID")
Mcov = Mcov[cols]

df = W.merge(Mcov, on="FILE_ID", how="left")
df = df.dropna(subset=["Lw_emp", "group"]).copy()
df["group"] = df["group"].astype("category")

print("\nWeighted average shortest path (sum of 1/weight) by group:")
print(df.groupby("group")["Lw_emp"].describe())

# OLS: include site fixed effects if available
if "SITE_ID" in df.columns:
    formula = "Lw_emp ~ C(group) + AGE_AT_SCAN + func_mean_fd + C(SITE_ID)"
    print("\nNote: Including site fixed effects (C(SITE_ID)).")
else:
    formula = "Lw_emp ~ C(group) + AGE_AT_SCAN + func_mean_fd"
    print("\nNote: SITE_ID not found; running model without site effects.")

m = smf.ols(formula, data=df).fit()
print("\n=== OLS (Lw_emp) ===")
print(m.summary())

#pull the ASD vs Control p-value
gp_term = [t for t in m.pvalues.index if t.startswith("C(group)")][0]
print(f"\nGroup p-value (ASD vs Control): {m.pvalues[gp_term]:.4g}")

#Plot: ASD vs Control boxplot
order = ["ASD", "Control"]
data = [df.loc[df.group == g, "Lw_emp"].values for g in order]

Path("results").mkdir(exist_ok=True, parents=True)
plt.figure(figsize=(6,4))
plt.boxplot(data, tick_labels=order, showfliers=False)
plt.ylabel("Weighted avg shortest path (sum of 1/weight)")
plt.title("Weighted path length by group")
plt.tight_layout()
plt.savefig("results/female/figs/weighted_path_groups.png", dpi=200)
plt.close()
print("Saved: results/female/figs/weighted_path_groups.png")
