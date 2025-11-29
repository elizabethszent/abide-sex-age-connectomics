import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/male/male_metrics_merged.csv")
#map labels for readability
df["group"] = df["DX_GROUP"].map({1:"ASD", 2:"Control"})
#center age & FD to help stability
df["age_c"] = df["AGE_AT_SCAN"] - df["AGE_AT_SCAN"].mean()
df["fd_c"]  = df["func_mean_fd"] - df["func_mean_fd"].mean()
#site as categorical if available
site_col = "SITE_ID" if "SITE_ID" in df.columns else None

for y in ["mean_degree","global_clustering","global_efficiency"]:
    if site_col:
        formula = f"{y} ~ C(group) + age_c + fd_c + C({site_col})"
    else:
        formula = f"{y} ~ C(group) + age_c + fd_c"
    model = smf.ols(formula, data=df).fit()
    p = model.pvalues.get("C(group)[T.Control]", float("nan"))#group effect
    print(f"\nOutcome: {y}\n{model.summary().tables[1]}")
    print(f"Group (ASD vs Control) p={p:.4g}")