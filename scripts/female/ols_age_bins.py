import os, pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("data/female/metrics_merged.csv")

if "SITE_ID" not in df.columns:
    ph_path = os.path.join("data","processed","Phenotypic_V1_0b_preprocessed1.csv")
    ph = pd.read_csv(ph_path)
    df = df.merge(ph[["FILE_ID","SITE_ID"]].drop_duplicates(), on="FILE_ID", how="left")

if "group" not in df.columns:
    df["group"] = df["DX_GROUP"].map({1:"ASD", 2:"Control"})

df = df.dropna(subset=["SITE_ID","AGE_AT_SCAN","func_mean_fd"])
df["age_bin"] = pd.cut(df["AGE_AT_SCAN"], bins=[0,12,18,99], labels=["child","adolescent","adult"])

for b, sub in df.groupby("age_bin"):
    sub = sub.dropna()
    print(f"\n[{b}] n={len(sub)}")
    if len(sub) < 20:
        print("  too small, skipping"); 
        continue
    for y in ["mean_degree","global_clustering","global_efficiency"]:
        m = smf.ols(f"{y} ~ C(group) + AGE_AT_SCAN + func_mean_fd + C(SITE_ID)", data=sub).fit()
        try:
            p = m.pvalues["C(group)[T.Control]"]
        except KeyError:
            p = m.pvalues[[k for k in m.pvalues.index if k.startswith("C(group)")][0]]
        print(f"  {y:18s}  p={p:.4g}")

