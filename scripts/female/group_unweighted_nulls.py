import os, math, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



PKEEP = 0.10#keep top 10% |r| edges
R = 100#null draws per subject
SEED = 1234
rng = random.Random(SEED)

DATA_ROOT = Path("data/roi_timeseries/cpac/nofilt_noglobal/rois_cc200")
META_CSV  = Path("data/female/metrics_merged.csv")#fILE_ID, DX_GROUP, AGE_AT_SCAN, func_mean_fd
OUT_CSV   = Path("data/female/unweighted_subject_nulls.csv")
FIG_DIR   = Path("results/female/figs"); FIG_DIR.mkdir(parents=True, exist_ok=True)

#helpers
#Robust read of a subject's CC200 ROI time-series -> float array [T x 200-ish]
def load_ts_by_id(file_id: str) -> np.ndarray:
    ts_path = DATA_ROOT / f"{file_id}_rois_cc200.1D"
    df = pd.read_csv(
        ts_path, sep=r"\s+", header=None, engine="python",
        comment="#", dtype=str
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.interpolate(axis=0, limit_direction="both").dropna(axis=1)
    good = df.std(axis=0) > 0
    df = df.loc[:, good]
    return df.to_numpy(dtype=float)

#keep top p% |r| (upper-tri) unweighted undirected graph
def binary_graph_from_ts(ts: np.ndarray, pkeep: float = PKEEP, use_abs: bool = True) -> nx.Graph:
    C = np.corrcoef(ts, rowvar=False)
    W = np.abs(C) if use_abs else C.copy()
    np.fill_diagonal(W, 0.0)

    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    vals = W[iu]
    k = int(np.floor(pkeep * vals.size))
    thresh = np.partition(vals, -k)[-k] if k > 0 else np.inf
    keep = (W >= thresh)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(np.triu(keep, 1))
    G.add_edges_from(zip(rows, cols))
    return G

#Largest connected component
def lcc(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()

#ER nulls matched on N and density p
def subject_ER_nulls(Gb: nx.Graph, R: int, rng: random.Random):
    N = Gb.number_of_nodes()
    M = Gb.number_of_edges()
    if N < 2:
        return [np.nan], [np.nan]

    p = 2.0 * M / (N * (N - 1))
    clust, path = [], []
    for _ in range(R):
        Ger = nx.gnp_random_graph(N, p, seed=rng.randint(1, 10_000_000))
        clust.append(nx.average_clustering(Ger))
        H = lcc(Ger)
        #graph with >=2 nodes required for path length
        if H.number_of_nodes() >= 2 and nx.is_connected(H):
            path.append(nx.average_shortest_path_length(H))
        else:
            path.append(np.nan)
    return clust, path

#Maslov–Sneppen degree-preserving rewiring on a binary graph
def degree_preserving_sample(Gb: nx.Graph, nswap_factor=10, rng=None, keep_connected=False):
    H = Gb.copy()
    M = H.number_of_edges()
    nswap = max(1, int(nswap_factor * M))
    max_tries = 10 * nswap
    try:
        if keep_connected:
            nx.connected_double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=rng.randint(1, 10_000_000))
        else:
            nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=rng.randint(1, 10_000_000))
    except Exception:
        #if swap fails (rare), return original copy
        pass
    return H

def subject_DP_nulls(Gb: nx.Graph, R: int, rng: random.Random):
    clust, path = [], []
    for _ in range(R):
        Gdp = degree_preserving_sample(Gb, nswap_factor=10, rng=rng, keep_connected=False)
        clust.append(nx.average_clustering(Gdp))
        H = lcc(Gdp)
        if H.number_of_nodes() >= 2 and nx.is_connected(H):
            path.append(nx.average_shortest_path_length(H))
        else:
            path.append(np.nan)
    return clust, path

def mean_sd(a):
    a = np.asarray(a, float)
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1))

#main
def main():
    meta = pd.read_csv(META_CSV)
    #map DX_GROUP to labels (1=ASD, 2=Control)
    meta["group"] = meta["DX_GROUP"].map({1: "ASD", 2: "Control"}).astype("category")

    rows = []
    for i, (fid, grp, age, fd) in enumerate(
        zip(meta["FILE_ID"], meta["group"], meta["AGE_AT_SCAN"], meta["func_mean_fd"])
    ):
        try:
            ts = load_ts_by_id(fid)
            Gb = binary_graph_from_ts(ts, pkeep=PKEEP, use_abs=True)

            N = Gb.number_of_nodes()
            M = Gb.number_of_edges()
            density = 2.0 * M / (N * (N - 1)) if N > 1 else np.nan

            #empirical metrics (binary)
            C_emp = nx.average_clustering(Gb)
            H = lcc(Gb)
            L_emp = nx.average_shortest_path_length(H) if H.number_of_nodes() >= 2 else np.nan

            #nulls
            erC, erL = subject_ER_nulls(Gb, R, rng)
            dpC, dpL = subject_DP_nulls(Gb, R, rng)

            erC_mu, erC_sd = mean_sd(erC); erL_mu, erL_sd = mean_sd(erL)
            dpC_mu, dpC_sd = mean_sd(dpC); dpL_mu, dpL_sd = mean_sd(dpL)

            #z-scores 
            zC_ER = (C_emp - erC_mu) / erC_sd if erC_sd > 0 else np.nan
            zL_ER = (L_emp - erL_mu) / erL_sd if erL_sd > 0 else np.nan
            zC_DP = (C_emp - dpC_mu) / dpC_sd if dpC_sd > 0 else np.nan
            zL_DP = (L_emp - dpL_mu) / dpL_sd if dpL_sd > 0 else np.nan

            #small-world index vs ER
            sigma_ER = (C_emp / erC_mu) / (L_emp / erL_mu) if erC_mu > 0 and erL_mu > 0 else np.nan

            rows.append(dict(
                FILE_ID=fid, group=grp, AGE_AT_SCAN=age, func_mean_fd=fd,
                N=N, M=M, density=density,
                C_emp=C_emp, L_emp=L_emp,
                erC_mu=erC_mu, erC_sd=erC_sd, erL_mu=erL_mu, erL_sd=erL_sd,
                dpC_mu=dpC_mu, dpC_sd=dpC_sd, dpL_mu=dpL_mu, dpL_sd=dpL_sd,
                zC_ER=zC_ER, zL_ER=zL_ER, zC_DP=zC_DP, zL_DP=zL_DP,
                sigma_ER=sigma_ER
            ))

            if (i + 1) % 10 == 0:
                print(f"[progress] finished {i+1} subjects")

        except Exception as e:
            print(f"[WARN] {fid}: {e}")
            continue

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved per-subject results -> {OUT_CSV}")
    print("\nGroup means (empirical binary clustering):")
    print(out.groupby("group")["C_emp"].mean())

    #group plots 
    def boxplot_by_group(series_name, ylabel, fname):
        plt.figure(figsize=(6,4))
        data = [out.loc[out["group"]=="ASD", series_name].dropna(),
                out.loc[out["group"]=="Control", series_name].dropna()]
        plt.boxplot(data, labels=["ASD","Control"], showfliers=False)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by group")
        plt.tight_layout()
        fp = FIG_DIR / fname
        plt.savefig(fp, dpi=200); plt.close()
        print(f"Saved {fp}")

    boxplot_by_group("C_emp", "Average clustering (binary)", "binary_clustering_groups.png")
    boxplot_by_group("L_emp", "Average shortest path (LCC)", "binary_path_groups.png")
    boxplot_by_group("zC_ER", "z(Clustering vs ER)", "zC_ER_groups.png")
    boxplot_by_group("zL_ER", "z(Path length vs ER)", "zL_ER_groups.png")
    boxplot_by_group("sigma_ER", "Small-world index σ (vs ER)", "sigma_ER_groups.png")

    #quick OLS
    try:
        import statsmodels.formula.api as smf
        df = out.dropna(subset=["C_emp","AGE_AT_SCAN","func_mean_fd"]).copy()
        df["group"] = df["group"].astype("category")
        m = smf.ols("C_emp ~ C(group) + AGE_AT_SCAN + func_mean_fd", data=df).fit()
        print("\n=== OLS: C_emp ~ group + age + motion ===")
        print(m.summary())
        with open(FIG_DIR / "binary_clustering_ols.txt", "w", encoding="utf-8") as f:
            f.write(m.summary().as_text())
        print(f"Saved OLS table -> {FIG_DIR/'binary_clustering_ols.txt'}")
    except Exception as e:
        print(f"[OLS skipped] {e}")

if __name__ == "__main__":
    main()
