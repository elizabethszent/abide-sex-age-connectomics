# scripts/compute_unweighted_path_subjects.py
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

#config 
PKEEP = 0.10 #keep top 10% strongest |r| edges
DATA_ROOT = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/roi_timeseries/cpac/nofilt_noglobal/rois_cc200")
META = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/female/metrics_merged.csv") 
OUT = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/female/unweighted_path_subjects.csv")

#helpers 
def load_ts(ts_path: Path) -> np.ndarray:
    df = pd.read_csv(
        ts_path, sep=r"\s+", header=None, engine="python",
        comment="#", dtype=str
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").interpolate(axis=0, limit_direction="both")
    df = df.dropna(axis=1)
    good = df.std(axis=0) > 0
    df = df.loc[:, good]
    return df.to_numpy(dtype=float)

def build_unweighted_graph(ts: np.ndarray, pkeep: float = 0.10, use_abs: bool = True) -> nx.Graph:
    C = np.corrcoef(ts, rowvar=False)
    W = np.abs(C) if use_abs else C.copy()
    np.fill_diagonal(W, 0.0)

    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    vals = W[iu]
    m = vals.size
    k = max(1, int(np.floor(pkeep * m))) #number of edges to keep

    #threshold at the k-th largest value
    thr = np.partition(vals, -k)[-k]
    keep = (W >= thr)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(np.triu(keep, 1))
    G.add_edges_from(zip(rows, cols))
    return G

#Average shortest path on the largest connected component
def aspl_lcc(G: nx.Graph) -> float:
    if nx.is_connected(G):
        H = G
    else:
        H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.average_shortest_path_length(H)

#main
meta = pd.read_csv(META)
fids = meta["FILE_ID"].tolist()

rows = []
for i, fid in enumerate(fids, 1):
    ts_path = DATA_ROOT / f"{fid}_rois_cc200.1D"
    if not ts_path.exists():
        print(f"[skip] no ts for {fid}")
        continue

    try:
        ts = load_ts(ts_path)
        G  = build_unweighted_graph(ts, pkeep=PKEEP, use_abs=True)
        L  = aspl_lcc(G)
        rows.append({"FILE_ID": fid, "L_emp": L})
    except Exception as e:
        print(f"[warn] {fid}: {e}")

    if i % 10 == 0:
        print(f"[progress] {i} subjects processed")

out = pd.DataFrame(rows)
OUT.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(OUT, index=False)
print(f"Saved -> {OUT}  (n={len(out)})")
