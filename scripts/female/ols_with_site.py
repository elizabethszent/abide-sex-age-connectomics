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

for y in ["mean_degree","global_clustering","global_efficiency"]:
    m = smf.ols(f"{y} ~ C(group) + AGE_AT_SCAN + func_mean_fd + C(SITE_ID)", data=df).fit()
    print(f"\nOutcome: {y}")
    print(m.summary().tables[1])
    try:
        p = m.pvalues["C(group)[T.Control]"]
    except KeyError:
        p = m.pvalues[[k for k in m.pvalues.index if k.startswith("C(group)")][0]]
    print(f"Group (ASD vs Control) p={p:.4g}")
