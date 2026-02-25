import pandas as pd

csv_path = r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\results\louvain_bestK7_vs_yeo\OVERALL_sexbalanced_fd-0.3\BESTK7_module_x_yeo_counts.csv"

df = pd.read_csv(csv_path)

# Make sure first column is the module id
if "Louvain_module" not in df.columns:
    # sometimes pandas reads an unnamed first column
    df = df.rename(columns={df.columns[0]: "Louvain_module"})

df = df.set_index("Louvain_module")
df = df.apply(pd.to_numeric)

# totals
df["Total"] = df.sum(axis=1)
df["CortexTotal"] = df["Total"] - df.get("Background", 0)

# dominant cortex label + match %
net_cols = [c for c in df.columns if c not in ["Total", "CortexTotal", "Background"]]
dom = df[net_cols].idxmax(axis=1)
dom_ct = df[net_cols].max(axis=1)
match_pct = (dom_ct / df["CortexTotal"] * 100).round(1)

summary = pd.DataFrame({
    "n_all": df["Total"].astype(int),
    "BG": df.get("Background", 0).astype(int),
    "n_cortex": df["CortexTotal"].astype(int),
    "Dominant_Yeo7": dom,
    "Match_%_cortex": match_pct.map(lambda x: f"{x:.1f}%"),
})
print("\nPer-module summary:\n", summary)

# cortex-only composition table
pct = (df[net_cols].div(df["CortexTotal"], axis=0) * 100).round(1)
print("\nCortex-only % composition:\n", pct)

overall_purity = (dom_ct.sum() / df["CortexTotal"].sum() * 100)
print(f"\nOverall cortex-only purity: {overall_purity:.1f}%")