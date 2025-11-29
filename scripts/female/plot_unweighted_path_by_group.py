import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

#load per-subject unweighted path lengths
P = pd.read_csv("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/female/unweighted_path_subjects.csv") 

#load phenotype/meta to get DX_GROUP, AGE_AT_SCAN, motion
M = pd.read_csv("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/female/metrics_merged.csv")
keep = [c for c in ["FILE_ID","DX_GROUP","AGE_AT_SCAN","func_mean_fd"] if c in M.columns]
M = M[keep].copy()

#merge; avoid accidental suffix surprises; then normalize names
df = P.merge(M, on="FILE_ID", how="left", validate="1:1", suffixes=("", "_m"))

for col in ["AGE_AT_SCAN", "func_mean_fd", "DX_GROUP"]:
    if col not in df and f"{col}_m" in df.columns:
        df[col] = df[f"{col}_m"]

#group label
if "group" not in df.columns:
    df["group"] = df["DX_GROUP"].map({1: "ASD", 2: "Control"}).astype("category")

#keep rows with everything we need
df = df.dropna(subset=["L_emp", "AGE_AT_SCAN", "func_mean_fd", "group"])

#quick summary
print("\nAverage shortest path (unweighted) by group:")
print(df.groupby("group")["L_emp"].describe())

#OLS with age & motion covariates
m = smf.ols("L_emp ~ C(group) + AGE_AT_SCAN + func_mean_fd", data=df).fit()
print("\n=== OLS (L_emp) ===")
print(m.summary())

#Boxplot
plt.figure(figsize=(7,4))
df.boxplot(column="L_emp", by="group", showfliers=False)
plt.title("Average shortest path (unweighted) by group")
plt.suptitle("")
plt.ylabel("Average shortest path (LCC)")
plt.tight_layout()
plt.savefig("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/results/female/figs/unweighted_path_groups.png", dpi=200)
plt.close()
print("\nSaved figure -> C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/results/female/figs/unweighted_path_groups.png")
