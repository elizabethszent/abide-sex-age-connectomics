# scripts/figures/plot_network_pairs_by_sex_dx.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==== paths ====
BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

F_MATS   = BASE / r"results\network_level\female_network_mats.csv"
M_MATS   = BASE / r"results\network_level\male_network_mats.csv"
F_STATS  = BASE / r"results\network_level\female_network_stats.csv"

FIG_DIR  = BASE / r"results\figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ==== load data ====
fem = pd.read_csv(F_MATS)
male = pd.read_csv(M_MATS)

fem["sex"] = "F"
male["sex"] = "M"

all_mats = pd.concat([fem, male], ignore_index=True)

# map DX_GROUP to ASD / Control (ABIDE: 1=ASD, 2=Control)
dx_map = {1: "ASD", 2: "Control"}
all_mats["dx_label"] = all_mats["DX_GROUP"].map(dx_map)

# drop any rows missing DX_GROUP just in case
all_mats = all_mats.dropna(subset=["DX_GROUP"])

# ==== choose top network pairs from female stats ====
f_stats = pd.read_csv(F_STATS)

# sort by absolute beta and take top 2 metrics
f_stats_sorted = f_stats.reindex(
    f_stats["beta_ASD_vs_Control"].abs().sort_values(ascending=False).index
)
top_pairs = list(f_stats_sorted["metric"].head(2))

print("Top network pairs (from female stats):", top_pairs)

# you can rename labels here if you know which module is which system
pair_labels = {col: col.replace("net_", "net ") for col in top_pairs}

# ==== helper to make one figure per pair ====
def plot_pair(col: str):
    # 4 groups: F-ASD, F-Control, M-ASD, M-Control
    group_order = [("F", "ASD"), ("F", "Control"),
                   ("M", "ASD"), ("M", "Control")]
    labels = ["F-ASD", "F-Control", "M-ASD", "M-Control"]

    data = []
    for sex, dx in group_order:
        vals = all_mats[(all_mats["sex"] == sex) &
                        (all_mats["dx_label"] == dx)][col].dropna().values
        data.append(vals)

    plt.figure(figsize=(6, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Fisher-z connectivity")
    plt.title(f"{pair_labels.get(col, col)} by sex and diagnosis")
    plt.tight_layout()

    out_path = FIG_DIR / f"{col}_by_sex_dx.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure for {col} -> {out_path}")

    # small summary table in console
    summary = (
        all_mats
        .groupby(["sex", "dx_label"])[col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    print(f"\nSummary for {col}:")
    print(summary.to_string(index=False))
    print("-" * 60)


# ==== make figures for the top pairs ====
for col in top_pairs:
    plot_pair(col)

print("Done.")
