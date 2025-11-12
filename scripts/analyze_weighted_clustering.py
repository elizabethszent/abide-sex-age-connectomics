# scripts/analyze_weighted_clustering.py
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path

W = pd.read_csv("results/weighted_clustering_subjects.csv")   # has FILE_ID, Cw_emp, group
M = pd.read_csv("results/female_metadata_included.csv")       # FILE_ID, DX_GROUP, AGE_AT_SCAN, func_mean_fd, (maybe SITE_ID)

# Choose merge columns that exist
need = ["FILE_ID","DX_GROUP","AGE_AT_SCAN","func_mean_fd"]
have = [c for c in need if c in M.columns]
use_site = "SITE_ID" in M.columns
cols = have + (["SITE_ID"] if use_site else [])

if not set(need).issubset(M.columns):
    missing = list(set(need) - set(M.columns))
    print(f"Note: metadata missing columns {missing}; merging with what is available: {cols}")

df = pd.merge(W, M[cols], on="FILE_ID", how="inner")

# If group isn’t present in W, map it from DX_GROUP
if "group" not in df.columns and "DX_GROUP" in df.columns:
    df["group"] = df["DX_GROUP"].map({1:"ASD", 2:"Control"}).astype("category")

print("N after merge:", len(df))
print("\nGroup means (weighted clustering):")
print(df.groupby("group")["Cw_emp"].mean())

# Build formula
formula = "Cw_emp ~ C(group) + AGE_AT_SCAN + func_mean_fd"
if use_site:
    formula += " + C(SITE_ID)"
else:
    print("Note: SITE_ID not found; running model without site fixed effects.")

# Fit OLS
model = smf.ols(formula, data=df).fit()
print("\n=== OLS (Cw_emp) ===")
print(model.summary())

# Save compact line for slides
p = model.pvalues.get("C(group)[T.Control]", float("nan"))
diff = df.loc[df.group=="Control","Cw_emp"].mean() - df.loc[df.group=="ASD","Cw_emp"].mean()
Path("results").mkdir(exist_ok=True, parents=True)
with open("results/weighted_clustering_group_test.txt","w") as f:
    f.write(f"Cw (CTL–ASD) = {diff:.4f}  |  p={p:.4f}  "
            f"{'(age, motion, site controlled)' if use_site else '(age, motion controlled)'}\n")

# Simple boxplot for the slide
plt.figure(figsize=(5,4))
df[["group","Cw_emp"]].boxplot(by="group", grid=False)
plt.suptitle("")
plt.title("Weighted clustering by group")
plt.xlabel("")
plt.ylabel("Cw")
plt.tight_layout()
plt.savefig("results/weighted_clustering_groups_box.png", dpi=200)
plt.close()

