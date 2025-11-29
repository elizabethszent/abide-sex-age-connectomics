import numpy as np
from pathlib import Path
import networkx as nx
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
GROUP_MATS = [
    BASE / r"results\group_connectomes\F_ASD_Zmean.npy",
    BASE / r"results\group_connectomes\F_CTL_Zmean.npy",
    BASE / r"results\group_connectomes\M_ASD_Zmean.npy",
    BASE / r"results\group_connectomes\M_CTL_Zmean.npy",
]

OUT_NPY = BASE / r"results\group_connectomes\CC200_modules.npy"
OUT_TXT = BASE / r"results\group_connectomes\CC200_modules.txt"

TARGET_MIN, TARGET_MAX = 7, 20
SEED = 42

RES_GRID_COARSE = [0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
RES_GRID_EXPAND = np.linspace(0.6, 2.0, 29)

def build_graph_from_matrix(A: np.ndarray) -> nx.Graph:

    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    r, c = np.triu_indices(n, k=1)
    w = A[r, c]
    mask = w > 0
    for u, v, wt in zip(r[mask], c[mask], w[mask]):
        G.add_edge(int(u), int(v), weight=float(wt))
    return G

def louvain_best_in_range(G: nx.Graph):

    cands = []

    # coarse grid
    for gamma in RES_GRID_COARSE:
        comms = louvain_communities(G, weight="weight", resolution=float(gamma), seed=SEED)
        Q = modularity(G, comms, weight="weight")
        cands.append((float(gamma), len(comms), Q, comms))

    def in_range(k: int) -> bool:
        return TARGET_MIN <= k <= TARGET_MAX

    elig = [c for c in cands if in_range(c[1])]

    #if nothing in range, expand grid
    if not elig:
        for gamma in RES_GRID_EXPAND:
            comms = louvain_communities(
                G, weight="weight", resolution=float(gamma), seed=SEED
            )
            Q = modularity(G, comms, weight="weight")
            cands.append((float(gamma), len(comms), Q, comms))
        elig = [c for c in cands if in_range(c[1])]

    if elig:
        #pick: highest Q, then fewer modules
        gamma, k, Q, comms = sorted(
            elig, key=lambda t: (-t[2], t[1])
        )[0]
        return gamma, k, Q, comms

    #fallback: closest K to [TARGET_MIN,TARGET_MAX], then highest Q
    def dist_to_range(k: int) -> int:
        if k < TARGET_MIN:
            return TARGET_MIN - k
        if k > TARGET_MAX:
            return k - TARGET_MAX
        return 0

    gamma, k, Q, comms = sorted(
        cands, key=lambda t: (dist_to_range(t[1]), -t[2])
    )[0]
    return gamma, k, Q, comms

def assign_modules(comms):

    comms_sorted = sorted(comms, key=lambda s: (-len(s), min(s)))
    n2m = {}
    for mid, comm in enumerate(comms_sorted, start=1):
        for n in comm:
            n2m[n] = mid
    return n2m

def main():
    #load and average Z matrices
    mats = []
    for p in GROUP_MATS:
        if not p.exists():
            raise FileNotFoundError(f"Missing matrix: {p}")
        mats.append(np.load(p))

    A = np.mean(mats, axis=0)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Grand mean matrix not square: {A.shape}")

    n = A.shape[0]
    print(f"Grand mean matrix shape: {A.shape}")

    #keep positive weights, zero diagonal
    A = np.maximum(A, 0.0)
    np.fill_diagonal(A, 0.0)

    #build graph and run Louvain + resolution sweep
    G = build_graph_from_matrix(A)
    gamma, k, Q, comms = louvain_best_in_range(G)
    print(f"Louvain: gamma={gamma:.4f}, modules={k}, Q={Q:.4f}")

    #assign module ids 1..K for each node index 0..n-1
    n2m = assign_modules(comms)
    modules = np.array([n2m[i] for i in range(n)], dtype=int)

    #save modules
    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, modules)
    print(f"Saved modules to {OUT_NPY}")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("ROI_index\tModule\n")
        for i, m in enumerate(modules, start=1):
            f.write(f"{i}\t{m}\n")
    print(f"Saved human-readable modules to {OUT_TXT}")

if __name__ == "__main__":
    main()
