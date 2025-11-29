import numpy as np
from pathlib import Path
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

GROUP_MATS = [
    BASE / r"results\group_connectomes\F_ASD_Zmean.npy",
    BASE / r"results\group_connectomes\F_CTL_Zmean.npy",
    BASE / r"results\group_connectomes\M_ASD_Zmean.npy",
    BASE / r"results\group_connectomes\M_CTL_Zmean.npy",
]

OUT_NPY = BASE / r"results\group_connectomes\CC200_modules.npy"
OUT_TXT = BASE / r"results\group_connectomes\CC200_modules.txt"

def main():
    mats = []
    for p in GROUP_MATS:
        if not p.exists():
            raise FileNotFoundError(f"Missing matrix: {p}")
        mat = np.load(p)
        mats.append(mat)

    #grand mean connectivity
    A = np.mean(mats, axis=0)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Grand mean matrix not square: {A.shape}")

    n = A.shape[0]
    print(f"Grand mean matrix shape: {A.shape}")

    #keep only positive weights, zero diagonal
    A = A.copy()
    A[A < 0] = 0.0
    np.fill_diagonal(A, 0.0)

    #build weighted graph
    G = nx.from_numpy_array(A)  # nodes 0..n-1, 'weight' attribute

    print("Running greedy modularity community detection...")
    comms = list(greedy_modularity_communities(G, weight="weight"))
    print(f"Found {len(comms)} modules")

    # Map node to module ID (1..K)
    modules = np.zeros(n, dtype=int)
    for mid, nodes in enumerate(comms, start=1):
        for node in nodes:
            modules[node] = mid

    if (modules == 0).any():
        raise RuntimeError("Some nodes got module 0 (unassigned).")

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
