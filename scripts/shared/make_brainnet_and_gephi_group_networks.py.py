# scripts/make_brainnet_and_gephi_group_networks.py
import numpy as np
import pandas as pd
from pathlib import Path
import os
import networkx as nx

ROOT = Path(r"C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")
IN = ROOT / "results" / "group_connectomes"

OUT_BRAINNET = ROOT / "results" / "vis" / "brainnet"
OUT_GEPHI    = ROOT / "results" / "vis" / "gephi"
OUT_BRAINNET.mkdir(parents=True, exist_ok=True)
OUT_GEPHI.mkdir(parents=True, exist_ok=True)

# fraction of undirected edges to keep
PKEEP = 0.10

GROUPS = ["F_ASD", "F_CTL", "M_ASD", "M_CTL"]

for name in GROUPS:
    zpath = IN / f"{name}_Zmean.npy"
    if not zpath.exists():
        print(f"[WARN] Missing {zpath}, skipping {name}.")
        continue

    Z = np.load(zpath)
    if Z.shape[0] != Z.shape[1]:
        print(f"[WARN] {name}: non-square matrix {Z.shape}, skipping.")
        continue

    n = Z.shape[0]
    W = np.abs(Z).astype(float)
    np.fill_diagonal(W, 0.0)

    # --- pick top p% undirected edges ---
    iu = np.triu_indices(n, 1)
    vals = W[iu]
    k = int(np.floor(PKEEP * vals.size))
    if k < 1:
        print(f"[WARN] {name}: too few edges to keep, skipping.")
        continue

    thresh = np.partition(vals, -k)[-k]
    keep_mask = W >= thresh

    # --- BrainNet .edge: full matrix with zeros elsewhere ---
    A = np.where(keep_mask, W, 0.0)
    edge_path = OUT_BRAINNET / f"{name}_top{int(PKEEP*100)}.edge"
    np.savetxt(edge_path, A, fmt="%.6f")
    print(f"[OK] BrainNet edge matrix for {name} -> {edge_path}")

    # --- Gephi edges ---
    rows, cols = np.where(np.triu(keep_mask, 1))
    weights = W[rows, cols]

    gephi_edges = pd.DataFrame({
        "Source": rows,
        "Target": cols,
        "Weight": weights,
        "Type": "Undirected",
    })
    gephi_edges_path = OUT_GEPHI / f"{name}_edges.csv"
    gephi_edges.to_csv(gephi_edges_path, index=False)
    print(f"[OK] Gephi edge list for {name} -> {gephi_edges_path}")

    # --- Gephi nodes (one file per group with degree) ---
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v, w in zip(rows, cols, weights):
        G.add_edge(int(u), int(v), weight=float(w))

    degrees = dict(G.degree())

    gephi_nodes = pd.DataFrame({
        "Id": np.arange(n),
        "Label": [f"ROI_{i+1}" for i in range(n)],
        "Degree": [degrees[i] for i in range(n)],
        # Optional group tag column if you want
        "Group": name,
    })
    gephi_nodes_path = OUT_GEPHI / f"{name}_nodes.csv"
    gephi_nodes.to_csv(gephi_nodes_path, index=False)
    print(f"[OK] Gephi node table for {name} -> {gephi_nodes_path}")

print("\nDone generating BrainNet and Gephi files for all available groups.")
