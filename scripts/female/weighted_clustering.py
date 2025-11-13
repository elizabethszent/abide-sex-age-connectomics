# scripts/weighted_clustering.py
# Computes weighted clustering and weighted ASPL per subject,
# plus two nulls (degree-preserving rewiring + weight-shuffle),
# and writes results/weighted_clustering_subjects.csv

import math
import random
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


#keep top p% strongest |r| edges
PKEEP = 0.10

#where your CPAC ROI time-series live
DATA_ROOT = Path("data/roi_timeseries/cpac/nofilt_noglobal/rois_cc200")

#merged metadata you already produced (has FILE_ID, DX_GROUP, AGE_AT_SCAN, func_mean_fd, …)
META = Path("data/female/metrics_merged.csv")

#null model settings
NSWAP_FACTOR = 5#Maslov–Sneppen swaps ~ NSWAP_FACTOR * M (edges)
R = 20 #number of null draws per subject (bump to 50–100 later if time allows)

SEED = 1234
rng = random.Random(SEED)
np.random.seed(SEED)


# Helpers
def ts_path_for(file_id: str) -> Path:
    """Build the absolute .1D path for a given FILE_ID."""
    return DATA_ROOT / f"{file_id}_rois_cc200.1D"


def load_ts(ts_path: Path) -> np.ndarray:
    """
    Read a CPAC ROI time-series .1D as a clean float array [T x R].
    - Splits on whitespace
    - Coerces to numeric (bad tokens -> NaN), interpolates along time, drops remaining-NaN cols
    - Drops zero-variance columns
    """
    df = pd.read_csv(
        ts_path,
        sep=r"\s+",
        header=None,
        engine="python",
        comment="#",
        dtype=str,#read strings first to cleanly coerce
    )
    #numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    #drop all-NaN columns
    df = df.dropna(axis=1, how="all")
    #interpolate occasional NaNs along time, then drop any columns still NaN
    df = df.interpolate(axis=0, limit_direction="both")
    df = df.dropna(axis=1)
    #drop zero-variance ROIs
    good = df.std(axis=0) > 0
    df = df.loc[:, good]
    return df.to_numpy(dtype=float)


 
#Build an undirected weighted graph from time-series: Pearson correlation across ROIs, absolute value if use_abs,keep the top p% edges by weight, edge weight = |r| (positive)
def corr_top_p_graph(ts: np.ndarray, pkeep: float = PKEEP, use_abs: bool = True) -> nx.Graph:
    C = np.corrcoef(ts, rowvar=False)
    W = np.abs(C) if use_abs else C.copy()
    np.fill_diagonal(W, 0.0)

    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    vals = W[iu]
    m = vals.size
    k = int(np.floor(pkeep * m))
    if k < 1:
        # fallback: connect nothing
        return nx.Graph()

    thresh = np.partition(vals, -k)[-k]
    keep_mask = W >= thresh

    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(np.triu(keep_mask, 1))
    for i, j in zip(rows, cols):
        w = float(W[i, j])
        if w > 0.0:
            G.add_edge(i, j, weight=w)
    return G

#Return a simple graph with consecutive int labels and float weights; self-loops removed.
def relabel_simple_float(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        w = float(d.get("weight", 1.0))
        H.add_edge(int(u), int(v), weight=w)
    H = nx.convert_node_labels_to_integers(H, ordering="sorted")
    return H

#Average clustering with defensive relabeling; falls back to unweighted if weighted fails.
def safe_avg_clustering(G: nx.Graph, use_weight: bool = True) -> float:
    H = relabel_simple_float(G)
    if H.number_of_edges() == 0:
        return 0.0
    try:
        return nx.average_clustering(H, weight=("weight" if use_weight else None))
    except Exception as e:
        print(f"[warn] weighted clustering failed ({e}); falling back to unweighted.")
        return nx.average_clustering(H, weight=None)

#Average shortest path on LCC given edge length attribute (e.g., length_inv = 1/weight).
def safe_weighted_aspl(G: nx.Graph, length_attr: str = "length_inv") -> float:
    if G.number_of_nodes() == 0:
        return float("nan")
    if nx.is_connected(G):
        H = G
    else:
        comps = list(nx.connected_components(G))
        if not comps:
            return float("nan")
        H = G.subgraph(max(comps, key=len)).copy()
    if H.number_of_nodes() < 2 or H.number_of_edges() == 0:
        return float("nan")
    return nx.average_shortest_path_length(H, weight=length_attr)

#Maslov–Sneppen rewiring (preserve degree sequence) + reassign weights from the empirical set. Returns a clean graph with 'weight' and 'length_inv' = 1/weight.
def dp_sample_weighted(G_base: nx.Graph, nswap_factor: int = NSWAP_FACTOR, keep_connected: bool = False) -> nx.Graph:
    G0 = relabel_simple_float(G_base)
    Gb = nx.Graph()
    Gb.add_nodes_from(G0.nodes())
    Gb.add_edges_from(G0.edges())

    M = Gb.number_of_edges()
    nswap = max(1, int(nswap_factor * M))
    max_tries = 10 * nswap
    if keep_connected:
        nx.connected_double_edge_swap(Gb, nswap=nswap, max_tries=max_tries)
    else:
        nx.double_edge_swap(Gb, nswap=nswap, max_tries=max_tries)

    weights = [float(d.get("weight", 1.0)) for _, _, d in G0.edges(data=True)]
    rng.shuffle(weights)

    H = nx.Graph()
    H.add_nodes_from(Gb.nodes())
    for (e, w) in zip(Gb.edges(), weights):
        u, v = e
        w = float(w)
        if w <= 0:
            w = 1e-12
        H.add_edge(u, v, weight=w)

    for u, v, d in H.edges(data=True):
        d["length_inv"] = 1.0 / float(d["weight"])
    return H

#Keep topology but randomly permute the edge weights
def weight_shuffle_topology(G_base: nx.Graph) -> nx.Graph:
    H = relabel_simple_float(G_base)
    wlist = [float(d.get("weight", 1.0)) for _, _, d in H.edges(data=True)]
    rng.shuffle(wlist)
    for (e, w) in zip(H.edges(), wlist):
        u, v = e
        w = float(w)
        if w <= 0:
            w = 1e-12
        H[u][v]["weight"] = w
    for _, _, d in H.edges(data=True):
        d["length_inv"] = 1.0 / float(d["weight"])
    return H


def mean_sd(a):
    a = np.asarray(a, float)
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else (float(a.mean()), float("nan"))



# Main
def main():
    if not META.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {META}")

    meta = pd.read_csv(META)
    required_cols = {"FILE_ID", "DX_GROUP"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"Missing columns in {META}: {sorted(missing)}")

    meta["group"] = meta["DX_GROUP"].map({1: "ASD", 2: "Control"}).astype("category")

    Path("results").mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (fid, grp) in enumerate(zip(meta["FILE_ID"], meta["group"]), 1):
        ts_file = ts_path_for(fid)
        if not ts_file.exists():
            print(f"[skip] {fid}: time-series not found -> {ts_file}")
            continue

        try:
            ts = load_ts(ts_file)
            if ts.shape[1] < 5:
                print(f"[skip] {fid}: too few usable ROIs after cleaning (shape {ts.shape})")
                continue

            #subject graph
            G = corr_top_p_graph(ts, pkeep=PKEEP, use_abs=True)

            if G.number_of_edges() == 0:
                print(f"[skip] {fid}: thresholding produced empty graph")
                continue

            #empirical metrics
            Cw_emp = safe_avg_clustering(G, use_weight=True)
            G_emp = relabel_simple_float(G)
            for _, _, d in G_emp.edges(data=True):
                d["length_inv"] = 1.0 / float(d["weight"])
            Lw_emp = safe_weighted_aspl(G_emp, "length_inv")

            # nulls
            dp_Cw, ws_Cw, dp_Lw, ws_Lw = [], [], [], []
            for _ in range(R):
                Hdp = dp_sample_weighted(G, nswap_factor=NSWAP_FACTOR, keep_connected=False)
                Hws = weight_shuffle_topology(G)

                dp_Cw.append(safe_avg_clustering(Hdp, use_weight=True))
                ws_Cw.append(safe_avg_clustering(Hws, use_weight=True))
                dp_Lw.append(safe_weighted_aspl(Hdp, "length_inv"))
                ws_Lw.append(safe_weighted_aspl(Hws, "length_inv"))

            dpC_mu, dpC_sd = mean_sd(dp_Cw)
            wsC_mu, wsC_sd = mean_sd(ws_Cw)
            dpL_mu, dpL_sd = mean_sd(dp_Lw)
            wsL_mu, wsL_sd = mean_sd(ws_Lw)

            rows.append(
                dict(
                    FILE_ID=fid,
                    group=grp,
                    Cw_emp=Cw_emp,
                    Cw_dp_mu=dpC_mu,
                    Cw_dp_sd=dpC_sd,
                    Cw_ws_mu=wsC_mu,
                    Cw_ws_sd=wsC_sd,
                    Lw_emp=Lw_emp,
                    Lw_dp_mu=dpL_mu,
                    Lw_dp_sd=dpL_sd,
                    Lw_ws_mu=wsL_mu,
                    Lw_ws_sd=wsL_sd,
                )
            )

            if idx % 10 == 0:
                print(f"[progress] finished {idx} subjects")

        except Exception as e:
            print(f"[warn] {fid}: error {e}")

    if not rows:
        print("No subjects processed—nothing to save.")
        return

    out = pd.DataFrame(rows)
    #z-scores vs nulls (protect against zero SD)
    for col_emp, mu, sd, zname in [
        ("Cw_emp", "Cw_dp_mu", "Cw_dp_sd", "z_Cw_vs_DP"),
        ("Cw_emp", "Cw_ws_mu", "Cw_ws_sd", "z_Cw_vs_WS"),
        ("Lw_emp", "Lw_dp_mu", "Lw_dp_sd", "z_Lw_vs_DP"),
        ("Lw_emp", "Lw_ws_mu", "Lw_ws_sd", "z_Lw_vs_WS"),
    ]:
        denom = out[sd].replace(0, np.nan)
        out[zname] = (out[col_emp] - out[mu]) / denom

    out_file = Path("data/female/weighted_clustering_subjects.csv")
    out.to_csv(out_file, index=False)

    print("\nGroup means (empirical weighted clustering):")
    print(out.groupby("group")["Cw_emp"].mean())
    print("\nMean z (Cw vs DP null) by group:")
    print(out.groupby("group")["z_Cw_vs_DP"].mean())
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
