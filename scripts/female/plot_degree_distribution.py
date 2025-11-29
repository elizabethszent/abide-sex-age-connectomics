import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#load all per-node degree CSVs
deg_files = sorted(glob.glob("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/results/female/degrees/*.csv"))
if not deg_files:
    raise FileNotFoundError("No files found in results/degrees/*.csv")
D = pd.concat((pd.read_csv(f) for f in deg_files), ignore_index=True)

#expect at least FILE_ID, node, degree
if not {"FILE_ID", "degree"}.issubset(D.columns):
    raise ValueError(f"Degree files missing required columns. Found: {D.columns.tolist()}")

#attach group labels (1=ASD, 2=Control) from your metadata
meta_path = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/female/metrics_merged.csv")
if not meta_path.exists():
    raise FileNotFoundError("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/female/metrics_merged.csv not found. Re-run your metrics merge step.")

META = pd.read_csv(meta_path, usecols=["FILE_ID", "DX_GROUP"])
D = D.merge(META, on="FILE_ID", how="left")

#clean and map to labels
missing = D["DX_GROUP"].isna().sum()
if missing:
    print(f"Warning: {missing} rows missing DX_GROUP after merge. They will be dropped.")
D = D.dropna(subset=["DX_GROUP"]).copy()
D["DX_GROUP"] = D["DX_GROUP"].astype(int)
D["group"] = D["DX_GROUP"].map({1: "ASD", 2: "Control"}).astype("category")

#pull node-level degrees for each group
asd_deg = D.loc[D["group"].eq("ASD"), "degree"].to_numpy()
ctl_deg = D.loc[D["group"].eq("Control"), "degree"].to_numpy()

print("ASD mean degree:", asd_deg.mean().round(3), "| Control mean degree:", ctl_deg.mean().round(3))

#histogram on linear bins (degrees are small integers at 10% density)
bins = np.arange(0, 61, 1)  # 0..60 is plenty for CC200 @ 10% density
asd_pdf, edges = np.histogram(asd_deg, bins=bins, density=True)
ctl_pdf, _     = np.histogram(ctl_deg, bins=bins, density=True)
x = 0.5 * (edges[1:] + edges[:-1])  #correct midpoints for linear bins

#plot
plt.figure(figsize=(6,4))
plt.plot(x, asd_pdf, 'o-', label="ASD", lw=1.5, alpha=0.9)
plt.plot(x, ctl_pdf, 'o-', label="Control", lw=1.5, alpha=0.9)
plt.xlabel("degree $k$")
plt.ylabel("$P(k)$")
plt.legend(frameon=False)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/results/female/figs/degree_distribution_groups.png", dpi=200)
plt.show()
