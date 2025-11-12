# scripts/degree_distribution.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- settings ----
DENSITY = 0.10  # 10% strongest |r| edges
TS_DIR = os.path.join("data", "roi_timeseries", "cpac", "nofilt_noglobal", "rois_cc200")
METRICS_CSV = os.path.join("results", "metrics_merged.csv")  # has FILE_ID and DX_GROUP
OUT_DIR = os.path.join("results", "degrees")
os.makedirs(OUT_DIR, exist_ok=True)

def corr_to_adj_topk(C, density=0.10):
    """Binarize to an undirected graph keeping top-|r| edges at the given density."""
    n = C.shape[0]
    k = max(1, int(density * n * (n - 1) / 2))  #number of edges to keep (upper triangle)
    iu = np.triu_indices(n, 1)
    vals = np.abs(C[iu])
    idx_top = np.argpartition(vals, -k)[-k:]  #indices of top-k by |r|
    A = np.zeros((n, n), dtype=np.uint8)
    top_pairs = (iu[0][idx_top], iu[1][idx_top])
    A[top_pairs] = 1
    A[(top_pairs[1], top_pairs[0])] = 1  #mirror to lower triangle
    np.fill_diagonal(A, 0)
    return A

def degrees_for_subject(file_id):
    ts_path = os.path.join(TS_DIR, f"{file_id}_rois_cc200.1D")

    #robust numeric load: ignore comment lines and coerce to float
    try:
        X = np.loadtxt(ts_path, dtype=float, comments="#")
    except Exception:
        # Fallback via pandas if needed
        df = pd.read_csv(ts_path, sep=r"\s+", header=None, comment="#")
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")
        X = df.values.astype(float)

    #drop any zero-variance columns (rare but defensive)
    std = X.std(axis=0)
    good = std > 0
    if not np.all(good):
        X = X[:, good]

    #correlation, zero the diagonal
    C = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(C, 0.0)

    #binarize to 10% density
    A = corr_to_adj_topk(C, density=DENSITY)

    #degree per node
    deg = A.sum(axis=0).astype(int)
    return deg


df = pd.read_csv(METRICS_CSV)  # columns
# ABIDE I code: 1=ASD, 2=Control
df["group"] = df["DX_GROUP"].map({1: "ASD", 2: "Control"})

all_rows = []
for _, row in df.iterrows():
    fid = row["FILE_ID"]
    try:
        deg = degrees_for_subject(fid)
    except FileNotFoundError:
        print(f"[WARN] missing time series for {fid} – skipping")
        continue

    #save per-subject degrees (200 values)
    pd.Series(deg, name="degree").to_csv(os.path.join(OUT_DIR, f"{fid}_degrees.csv"),
                                         index=False)
    #store for group histograms
    for d in deg:
        all_rows.append({"FILE_ID": fid, "group": row["group"], "degree": int(d)})

#aggregate
agg = pd.DataFrame(all_rows)
agg.to_csv(os.path.join(OUT_DIR, "all_degrees_long.csv"), index=False)

#plot: degree distribution by group (overlayed hist) 
bins = np.arange(-0.5, 200.5, 1)  #integer bins
plt.figure(figsize=(7,5))
for g, color in [("ASD", "tab:blue"), ("Control", "tab:orange")]:
    vals = agg.loc[agg["group"].eq(g), "degree"].values
    plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=2, label=g, color=color)

plt.xlabel("Node degree (number of edges)")
plt.ylabel("Density")
plt.title(f"Degree distribution (10% density; N subjects = {df.shape[0]})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "degree_distribution_by_group.png"), dpi=200)
plt.close()

#quick sanity print
print("Saved per-subject degrees to:", OUT_DIR)
print("Group means (subject-level mean degree):")
print(agg.groupby("group")["degree"].mean())
