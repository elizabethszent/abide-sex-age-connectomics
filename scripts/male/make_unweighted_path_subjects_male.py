import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

#time series directory
TS_DIR = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\roi_timeseries\cpac\nofilt_noglobal\rois_cc200")

#Metadata for the cohort you want to analyze
META_CSV = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\male\male_metadata_included.csv")

OUT_CSV = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\male\unweighted_path_subjects_male.csv")

PKEEP = 0.10 #keep top 10% |r| edges 
R = 100  #number of null graphs per subject
RNG_SEED = 1234


#helpers
def load_ts_for_subject(file_id: str) -> np.ndarray:

    ts_path = TS_DIR / f"{file_id}_rois_cc200.1D"
    if not ts_path.exists():
        raise FileNotFoundError(f"Time-series not found for {file_id}: {ts_path}")

    df = pd.read_csv(
        ts_path,
        sep=r"\s+",
        engine="python",
        header=None,
        comment="#",
        dtype=str,
    ).apply(pd.to_numeric, errors="coerce")

    #drop columns that are all NaN
    df = df.dropna(axis=1, how="all")

    #drop zero-variance columns
    std = df.std(axis=0, numeric_only=True)
    good = std > 0
    df = df.loc[:, good]

    #final TS as float
    ts = df.to_numpy(dtype=float)
    return ts

 
#Given time-series (T x n), compute correlation matrix and build binary graph with top pkeep proportion of edges 
def build_binary_graph_from_ts(ts: np.ndarray, pkeep: float = PKEEP, use_abs: bool = True) -> nx.Graph:

    #n_rois = ts.shape[1]
    C = np.corrcoef(ts, rowvar=False)

    if use_abs:
        W = np.abs(C)
    else:
        W = C.copy()

    np.fill_diagonal(W, 0.0)

    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    vals = W[iu]

    k = int(np.floor(pkeep * vals.size))
    if k <= 0:
        raise ValueError("pkeep too small; no edges selected.")

    #threshold so that k edges are kept
    thresh = np.partition(vals, -k)[-k]
    keep_mask = W >= thresh

    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(np.triu(keep_mask, 1))
    for i, j in zip(rows, cols):
        G.add_edge(i, j)
    return G

#return largest connected component as a copy
def largest_cc_subgraph(G: nx.Graph) -> nx.Graph:
    if nx.is_connected(G):
        return G
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()



#degree-preserving Maslov–Sneppen randomization:performs double edge swaps on an unweighted graph.
def dp_sample(G_base: nx.Graph, nswap_factor: float = 10.0, keep_connected: bool = False) -> nx.Graph:
    H = G_base.copy()
    M = H.number_of_edges()
    nswap = max(1, int(nswap_factor * M))
    max_tries = 10 * nswap

    if keep_connected:
        nx.connected_double_edge_swap(H, nswap=nswap, max_tries=max_tries)
    else:
        nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries)
    return H


def mean_sd(a):
    a = np.asarray(a, float)
    return float(a.mean()), float(a.std(ddof=1))



#main
def main():
    rng = np.random.default_rng(RNG_SEED)

    meta = pd.read_csv(META_CSV)
    #ensure we have FILE_ID and group label
    if "FILE_ID" not in meta.columns:
        raise ValueError("META_CSV must contain FILE_ID column.")

    if "DX_GROUP" in meta.columns:
        meta["group"] = meta["DX_GROUP"].map({1: "ASD", 2: "Control"}).astype("category")
    elif "group" in meta.columns:
        #already labeled
        pass
    else:
        raise ValueError("Need either DX_GROUP or group column in metadata.")

    rows = []

    for i, row in meta.iterrows():
        fid = row["FILE_ID"]
        grp = row.get("group", row.get("DX_GROUP"))
        print(f"[{i+1}/{len(meta)}] subject {fid} ({grp})")

        #build subject graph
        ts = load_ts_for_subject(fid)
        Gb = build_binary_graph_from_ts(ts, pkeep=PKEEP, use_abs=True)

        #empirical L (unweighted)
        H_emp = largest_cc_subgraph(Gb)
        L_emp = nx.average_shortest_path_length(H_emp)

        #ER null ensemble
        N = Gb.number_of_nodes()
        M = Gb.number_of_edges()
        p = 2.0 * M / (N * (N - 1))

        er_L = []
        for _ in range(R):
            Ger = nx.gnp_random_graph(N, p)
            Her = largest_cc_subgraph(Ger)
            er_L.append(nx.average_shortest_path_length(Her))

        L_ER_mu, L_ER_sd = mean_sd(er_L)

        #DP null ensemble
        dp_L = []
        for _ in range(R):
            Gdp = dp_sample(Gb, nswap_factor=10.0, keep_connected=False)
            Hdp = largest_cc_subgraph(Gdp)
            dp_L.append(nx.average_shortest_path_length(Hdp))

        L_DP_mu, L_DP_sd = mean_sd(dp_L)

        #z-scores vs nulls
        zL_ER = (L_emp - L_ER_mu) / L_ER_sd if L_ER_sd > 0 else np.nan
        zL_DP = (L_emp - L_DP_mu) / L_DP_sd if L_DP_sd > 0 else np.nan

        rows.append(
            dict(
                FILE_ID=fid,
                group=str(grp),
                L_emp=L_emp,
                L_ER_mu=L_ER_mu,
                L_ER_sd=L_ER_sd,
                L_DP_mu=L_DP_mu,
                L_DP_sd=L_DP_sd,
                zL_vs_ER=zL_ER,
                zL_vs_DP=zL_DP,
            )
        )

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved unweighted path metrics per subject -> {OUT_CSV}")
    print("\nQuick group summary (L_emp):")
    print(out.groupby("group")["L_emp"].describe())


if __name__ == "__main__":
    main()
