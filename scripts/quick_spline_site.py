import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import bs
from pathlib import Path

mfile = Path("results/metrics_merged.csv")
meta  = Path("results/female_metadata_included.csv")

dfm = pd.read_csv(mfile)
dfm = dfm.dropna(subset=["DX_GROUP","AGE_AT_SCAN","func_mean_fd"])

#bring in SITE_ID from metadata (preferred over parsing FILE_ID)
if meta.exists():
    met = pd.read_csv(meta, dtype=str)
    keep_cols = ["FILE_ID","SITE_ID"]
    met = met[[c for c in keep_cols if c in met.columns]]
    dfm = dfm.merge(met, on="FILE_ID", how="left")

#If SITE_ID still missing, try a simple parse from FILE_ID prefix
if "SITE_ID" not in dfm.columns or dfm["SITE_ID"].isna().all():
    dfm["SITE_ID"] = dfm["FILE_ID"].str.split("_", n=1, expand=True)[0]

#types
dfm["DX_GROUP"] = dfm["DX_GROUP"].astype("category")
dfm["SITE_ID"]  = dfm["SITE_ID"].astype("category")

outcomes = ["mean_degree","global_clustering","global_efficiency"]

for y in outcomes:
    #cubic B-spline for age; include group×age spline; control for motion and site
    formula = f"{y} ~ C(DX_GROUP) * bs(AGE_AT_SCAN, df=4) + func_mean_fd + C(SITE_ID)"
    m = smf.ols(formula=formula, data=dfm).fit()
    print(f"\nOutcome: {y}")
    #test the whole interaction block (does ASD–Control difference vary with age?)
    param_index = list(m.params.index)
    inter_cols = [c for c in param_index if ('C(DX_GROUP)' in c and 'bs(' in c and ':' in c)]
    if inter_cols:
        R = np.zeros((len(inter_cols), len(param_index)))
        for i, col in enumerate(inter_cols):
            R[i, param_index.index(col)] = 1.0
        ft = m.f_test(R)
        print("  group × age-spline: F p =", float(ft.pvalue))
    #main effect of group (overall difference)
    grp_ps = [p for name, p in m.pvalues.items()
              if name.startswith("C(DX_GROUP)") and ":bs(" not in name]
    if grp_ps:
        print("  group main effect p =", float(min(grp_ps)))
