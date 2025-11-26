import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
HUB_CSV = BASE / r"results\hubs\female_hub_metrics.csv"
OUT_CSV = BASE / r"results\hubs\female_hub_stats.csv"

df = pd.read_csv(HUB_CSV)

df = df.dropna(subset=["DX_GROUP", "AGE_AT_SCAN", "func_mean_fd"])
df["group"] = df["DX_GROUP"].map({1: "ASD", 2: "Control"}).astype("category")
df["group"] = df["group"].cat.reorder_categories(["Control", "ASD"])

metrics = ["Q_fixed"] + \
          [c for c in df.columns if c.startswith("PC_med_m")] + \
          [c for c in df.columns if c.startswith("Z_med_m")]

rows = []
for col in metrics:
    formula = f"{col} ~ C(group) + AGE_AT_SCAN + func_mean_fd"
    m = smf.ols(formula, data=df).fit()
    beta = m.params.get("C(group)[T.ASD]", np.nan)
    pval = m.pvalues.get("C(group)[T.ASD]", np.nan)
    rows.append({
        "metric": col,
        "beta_ASD_vs_Control": beta,
        "p_ASD_vs_Control": pval,
        "R2": m.rsquared,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV}")

print("\nMetrics sorted by |beta|:")
print(
    out.reindex(out["beta_ASD_vs_Control"].abs()
                .sort_values(ascending=False).index)
       .head(10)
)
