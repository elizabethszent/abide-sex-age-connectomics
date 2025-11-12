# scripts/unweighted_metrics_and_nulls.py
import argparse, random, numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

PKEEP = 0.10  #top |r| density already used in your pipeline
DATA_ROOT = Path("data/roi_timeseries/cpac/nofilt_noglobal/rois_cc200")
META = Path("results/metrics_merged.csv")
R = 100
rng = random.Random(123)

#helpers 
def load_ts(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python",
                     comment="#", dtype=str)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    df = df.interpolate(axis=0, limit_direction="both").dropna(axis=1)
    df = df.loc[:, df.std(0) > 0]
    return df.to_numpy(dtype=float)

def top_p_graph_from_ts(ts, pkeep=0.10, use_abs=True):
    if isinstance(ts, (str, Path)):
        ts = load_ts(ts)
    C = np.corrcoef(ts, rowvar=False)
    W = np.abs(C) if use_abs else C.copy()
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    vals = W[iu]
    k = int(np.floor(pkeep * vals.size))
    thr = np.partition(vals, -k)[-k] if k > 0 else np.inf
    keep = W >= thr
    rows, cols = np.where(np.triu(keep, 1))
    G_w = nx.Graph()
    G_w.add_nodes_from(range(n))
    for i, j in zip(rows, cols):
        G_w.add_edge(i, j, weight=float(W[i, j]))
    return G_w

def to_binary(G_w):
    Gb = nx.Graph()
    Gb.add_nodes_from(G_w.nodes())
    Gb.add_edges_from(G_w.edges())  #drops attributes unweighted
    return Gb

def lcc_avg_shortest_path(G):
    if nx.is_connected(G):
        H = G
    else:
        H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.average_shortest_path_length(H), H.number_of_nodes()

def dp_sample(Gb, nswap_factor=10, keep_connected=False):
    H = Gb.copy()
    m = H.number_of_edges()
    nswap = max(1, int(nswap_factor * m))
    max_tries = 10 * nswap
    if keep_connected:
        nx.connected_double_edge_swap(H, nswap=nswap, max_tries=max_tries)
    else:
        nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries)
    return H

#main 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="FILE_ID (e.g., Pitt_0050005)")
    args = ap.parse_args()

    meta = pd.read_csv(META)
    fid = args.file or meta["FILE_ID"].iloc[0]
    ts_path = DATA_ROOT / f"{fid}_rois_cc200.1D"

    #build subject graph (weighted) then drop weights -> unweighted Gb
    G_w = top_p_graph_from_ts(ts_path, pkeep=PKEEP, use_abs=True)
    Gb = to_binary(G_w)

    #empirical unweighted metrics
    N, M = Gb.number_of_nodes(), Gb.number_of_edges()
    k_avg = 2 * M / N
    density = 2 * M / (N * (N - 1))
    C_emp = nx.average_clustering(Gb)
    L_emp, n_lcc = lcc_avg_shortest_path(Gb)

    print(f"[{fid}] N={N}  M={M}  density={density:.5f}  k={k_avg:.2f}")
    print(f"  (a) Avg clustering (binary): {C_emp:.4f}")
    print(f"  (b) Avg shortest path (LCC n={n_lcc}): {L_emp:.4f}")

    #ER null (same N and density)
    p = density
    er_C, er_L = [], []
    for r_ in range(R):
        Ger = nx.fast_gnp_random_graph(N, p, seed=rng.randint(0, 10**9))
        er_C.append(nx.average_clustering(Ger))
        Lr, _ = lcc_avg_shortest_path(Ger)
        er_L.append(Lr)
    er_C_mu, er_C_sd = float(np.mean(er_C)), float(np.std(er_C, ddof=1))
    er_L_mu, er_L_sd = float(np.mean(er_L)), float(np.std(er_L, ddof=1))

    #DP null (degree-preserving rewires)
    dp_C, dp_L = [], []
    for _ in range(R):
        Gdp = dp_sample(Gb, nswap_factor=10, keep_connected=False)
        dp_C.append(nx.average_clustering(Gdp))
        Lr, _ = lcc_avg_shortest_path(Gdp)
        dp_L.append(Lr)
    dp_C_mu, dp_C_sd = float(np.mean(dp_C)), float(np.std(dp_C, ddof=1))
    dp_L_mu, dp_L_sd = float(np.mean(dp_L)), float(np.std(dp_L, ddof=1))

    print("\nER null:")
    print(f"  C mean±sd = {er_C_mu:.4f} ± {er_C_sd:.4f}")
    print(f"  L mean±sd = {er_L_mu:.4f} ± {er_L_sd:.4f}")
    print("DP (degree-preserving) null:")
    print(f"  C mean±sd = {dp_C_mu:.4f} ± {dp_C_sd:.4f}")
    print(f"  L mean±sd = {dp_L_mu:.4f} ± {dp_L_sd:.4f}")

    #plot boxplots with empirical line
    Path("results").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.boxplot([er_C, dp_C], tick_labels=["ER", "DP"], showfliers=False)
    plt.axhline(C_emp, linestyle="--", linewidth=1, label="Empirical")
    plt.ylabel("Average clustering (binary)")
    plt.title(f"{fid} — clustering nulls (empirical dashed)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"results/unweighted_clustering_nulls_{fid}.png", dpi=200)

    plt.figure(figsize=(6,4))
    plt.boxplot([er_L, dp_L], tick_labels=["ER", "DP"], showfliers=False)
    plt.axhline(L_emp, linestyle="--", linewidth=1, label="Empirical")
    plt.ylabel("Average shortest path (LCC)")
    plt.title(f"{fid} — path length nulls (empirical dashed)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"results/unweighted_path_nulls_{fid}.png", dpi=200)

    #save a small summary table
    out = pd.DataFrame([{
        "FILE_ID": fid, "N": N, "M": M, "density": density, "k_avg": k_avg,
        "C_emp": C_emp, "L_emp": L_emp,
        "ER_C_mu": er_C_mu, "ER_C_sd": er_C_sd, "ER_L_mu": er_L_mu, "ER_L_sd": er_L_sd,
        "DP_C_mu": dp_C_mu, "DP_C_sd": dp_C_sd, "DP_L_mu": dp_L_mu, "DP_L_sd": dp_L_sd
    }])
    out.to_csv(f"results/unweighted_metrics_{fid}.csv", index=False)
    print(f"\nSaved figures + CSV in results/ for {fid}")
