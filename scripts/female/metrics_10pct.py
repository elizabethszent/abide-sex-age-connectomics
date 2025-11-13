# scripts/metrics_10pct.py
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

CONN_DIR = Path("data/connectomes/cpac/nofilt_noglobal/cc200_z")
IN_IDS = Path("data/female/female_subjects_included.txt")
OUT = Path("results"); OUT.mkdir(exist_ok=True)

ids = [line.strip() for line in IN_IDS.read_text().splitlines() if line.strip()]
rows = []

for sid in ids:
    Z = np.load(CONN_DIR / f"{sid}.npy")
    R = np.tanh(Z)#fisher z -> r
    np.fill_diagonal(R, 0.0)
    n = R.shape[0]

    #keep strongest 10% absolute edges
    m = n * (n - 1) // 2
    k = int(0.10 * m)
    iu = np.triu_indices(n, 1)
    vals = np.abs(R[iu])
    thr = np.partition(vals, -k)[-k]
    A = (np.abs(R) >= thr).astype(float)

    #degree
    deg = A.sum(0)

    #global clustering (transitivity)
    T = np.trace(np.linalg.matrix_power(A, 3)) / 6.0
    K = (deg * (deg - 1)).sum() / 2.0
    C = (3 * T) / K if K > 0 else np.nan

    #global efficiency on weighted graph (distance = 1/|r|)
    W = np.zeros_like(R)
    W[A > 0] = np.abs(R[A > 0])
    eps = 1e-9
    D = np.zeros_like(W)
    nz = W > 0
    D[nz] = 1.0 / (W[nz] + eps)
    dist = shortest_path(D, directed=False, unweighted=False)
    with np.errstate(divide="ignore"):
        invd = 1.0 / dist
    np.fill_diagonal(invd, 0.0)
    GE = invd.sum() / (n * (n - 1))

    rows.append({"FILE_ID": sid,
                 "mean_degree": float(deg.mean()),
                 "global_clustering": float(C),
                 "global_efficiency": float(GE)})

pd.DataFrame(rows).to_csv(OUT / "subject_metrics_10pct.csv", index=False)
print("Saved:", OUT / "subject_metrics_10pct.csv")
