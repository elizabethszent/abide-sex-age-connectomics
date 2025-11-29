
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#paths
BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

F_HUB = BASE / r"results\hubs\female_hub_metrics.csv"
M_HUB = BASE / r"results\hubs\male_hub_metrics.csv"

FIG_DIR = BASE / r"results\figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#load data
female = pd.read_csv(F_HUB)
male   = pd.read_csv(M_HUB)

#add sex labels
female["sex"] = "F"
male["sex"] = "M"

#concatenate
all_hubs = pd.concat([female, male], ignore_index=True)

#map DX_GROUP to labels (ABIDE: 1 = ASD, 2 = Control)
dx_map = {1: "ASD", 2: "Control"}
all_hubs["dx_label"] = all_hubs["DX_GROUP"].map(dx_map)

#drop rows with missing DX_GROUP just in case
all_hubs = all_hubs.dropna(subset=["DX_GROUP"])


#define groups in the order you want on the x-axis
group_order = [("F", "ASD"), ("F", "Control"),
               ("M", "ASD"), ("M", "Control")]
labels = ["F-ASD", "F-Control", "M-ASD", "M-Control"]

data = []
for sex, dx in group_order:
    vals = all_hubs[(all_hubs["sex"] == sex) &
                    (all_hubs["dx_label"] == dx)]["Q_fixed"].dropna().values
    data.append(vals)

plt.figure(figsize=(6, 5))
plt.boxplot(data, labels=labels, showmeans=True)

plt.ylabel("Modularity Q (fixed 7 modules)")
plt.title("Q_fixed by sex and diagnosis")
plt.tight_layout()

fig1_path = FIG_DIR / "Q_fixed_by_sex_dx.png"
plt.savefig(fig1_path, dpi=300)
plt.close()

print(f"Saved Figure 1: {fig1_path}")

#also print a small summary table in the console
summary = (
    all_hubs
    .groupby(["sex", "dx_label"])["Q_fixed"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
print("\nQ_fixed summary by sex × diagnosis:")
print(summary.to_string(index=False))


#work only with females for this plot
female_only = all_hubs[all_hubs["sex"] == "F"].copy()

#make sure dx_label is there
female_only["dx_label"] = female_only["DX_GROUP"].map(dx_map)

data_f = []
labels_f = ["ASD", "Control"]
for dx in labels_f:
    vals = female_only[female_only["dx_label"] == dx]["Z_med_m3"].dropna().values
    data_f.append(vals)

plt.figure(figsize=(4.5, 5))
plt.boxplot(data_f, labels=labels_f, showmeans=True)

plt.ylabel("Within-module degree z (module 3)")
plt.title("Female Z_med_m3 by diagnosis")
plt.tight_layout()

fig2_path = FIG_DIR / "Female_Z_med_m3_by_dx.png"
plt.savefig(fig2_path, dpi=300)
plt.close()

print(f"Saved Figure 2: {fig2_path}")

#quick summary for this module
summary_m3 = (
    female_only
    .groupby("dx_label")["Z_med_m3"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
print("\nFemale Z_med_m3 summary by diagnosis:")
print(summary_m3.to_string(index=False))
