# scripts/female/analyze_network_connectivity.py

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

NET_CSV = BASE / r"results\network_level\female_network_mats.csv"
OUT_CSV = BASE / r"results\network_level\female_network_stats.csv"


df = pd.read_csv(NET_CSV)

# keep only rows with all needed covariates
df = df.dropna(subset=["DX_GROUP", "AGE_AT_SCAN", "func_mean_fd"])

# ABIDE codes: 1 = ASD, 2 = Control
df["group"] = df["DX_GROUP"].map({1: "ASD", 2: "Control"}).astype("category")
# ensure Control is baseline
df["group"] = df["group"].cat.reorder_categories(["Control", "ASD"])

# all network-pair columns (net_0_0, net_0_1, ..., net_6_6)
net_cols = [c for c in df.columns if c.startswith("net_")]
print(f"Found {len(net_cols)} network-pair columns")

rows = []

for col in net_cols:
    formula = f"{col} ~ C(group) + AGE_AT_SCAN + func_mean_fd"
    m = smf.ols(formula, data=df).fit()

    beta = m.params.get("C(group)[T.ASD]", np.nan)   # ASD vs Control
    pval = m.pvalues.get("C(group)[T.ASD]", np.nan)

    rows.append({
        "metric": col,
        "beta_ASD_vs_Control": beta,
        "p_ASD_vs_Control": pval,
        "R2": m.rsquared,
    })

out = pd.DataFrame(rows)


p = out["p_ASD_vs_Control"].to_numpy()
p_clean = np.where(np.isnan(p), 1.0, p)
N = len(p_clean)

order = np.argsort(p_clean)
rank = np.empty_like(order)
rank[order] = np.arange(1, N + 1)

q = p_clean * N / rank
# enforce monotone decreasing q on sorted p
q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
q_adj = np.empty_like(q)
q_adj[order] = q_sorted
q_adj[np.isnan(p)] = np.nan

out["p_FDR"] = q_adj

out.to_csv(OUT_CSV, index=False)
print(f"Saved stats -> {OUT_CSV}")

print("\nTop few network pairs by |beta|:")
print(
    out.reindex(
        out["beta_ASD_vs_Control"].abs().sort_values(ascending=False).index
    ).head(10)
)
