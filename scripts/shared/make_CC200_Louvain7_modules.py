import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import community as community_louvain  #python-louvain package


GRAND_MEAN_PATH = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/results/shared/child_grand_mean_connectome.npy")

OUT_CSV = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/parcellation/CC200_Louvain7_modules.csv")


BRAINNET_NODE = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/parcellation/CC200_base.node")

N_MODULES_TARGET = 7#we want 7 modules

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def load_cc200_node_count(node_path: Path) -> int:
    with node_path.open("r") as f:
        n_lines = sum(1 for _ in f)
    return n_lines


def build_graph_from_matrix(C: np.ndarray, threshold: float = None) -> nx.Graph:
    if C.shape[0] != C.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {C.shape}")

    n = C.shape[0]
    W = C.copy()

    #zero out diagonal
    np.fill_diagonal(W, 0.0)

    if threshold is not None:
        mask = np.abs(W) >= threshold
    else:
        #keep all edges
        mask = np.abs(W) > 0

    G = nx.Graph()
    G.add_nodes_from(range(n))

    rows, cols = np.where(np.triu(mask, 1))
    for i, j in zip(rows, cols):
        w = float(W[i, j])
        if w != 0.0:
            G.add_edge(i, j, weight=w)

    return G


def louvain_with_k_approx(G: nx.Graph, k_target: int, n_trials: int = 50, random_state_base: int = 42):
    best_part = None
    best_k = None
    best_diff = None

    for t in range(n_trials):
        seed = random_state_base + t
        part = community_louvain.best_partition(G, weight="weight", random_state=seed)
        #number of communities
        comm_ids = set(part.values())
        k = len(comm_ids)
        diff = abs(k - k_target)

        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_k = k
            best_part = part

    return best_part, best_k


def relabel_communities_to_1based(partition: dict) -> np.ndarray:
    nodes = sorted(partition.keys())
    raw_labels = [partition[i] for i in nodes]
    unique_labels = sorted(set(raw_labels))
    label_map = {lab: i + 1 for i, lab in enumerate(unique_labels)}

    mapped = np.array([label_map[lab] for lab in raw_labels], dtype=int)
    return mapped  #index 0 = ROI 1, index 199 = ROI 200


def main():
    #check BrainNet node count
    if BRAINNET_NODE.exists():
        n_node = load_cc200_node_count(BRAINNET_NODE)
        print(f"BrainNet node file has {n_node} lines")
        if n_node != 200:
            raise RuntimeError(f"Expected 200 ROIs, got {n_node} in CC200_base.node")
    else:
        print("Warning: BrainNet node file not found, skipping N=200 sanity check")

    #Load grand-mean 200x200 matrix
    if not GRAND_MEAN_PATH.exists():
        raise FileNotFoundError(f"Grand-mean matrix not found at {GRAND_MEAN_PATH}")
    C = np.load(GRAND_MEAN_PATH)
    print(f"Loaded grand-mean connectome: shape={C.shape}")

    if C.shape != (200, 200):
        raise ValueError(f"Expected 200x200 matrix, got {C.shape}")

    #Build weighted graph
    # You can tweak threshold if you want a sparser graph, e.g. threshold=0.05
    G = build_graph_from_matrix(C, threshold=None)
    print(f"Graph: N={G.number_of_nodes()}, M={G.number_of_edges()}")

    #Louvain, searching for ~7 modules
    part, k_found = louvain_with_k_approx(G, k_target=N_MODULES_TARGET, n_trials=50)
    print(f"Louvain found {k_found} communities (target={N_MODULES_TARGET})")

    #Relabel modules to 1..K and build ROI->module table
    modules = relabel_communities_to_1based(part)  # length 200
    roi_ids = np.arange(1, 201, dtype=int)

    out_df = pd.DataFrame({"roi": roi_ids, "module": modules})
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved ROI→module mapping -> {OUT_CSV}")

    #quick sanity check summary
    print("\nModule sizes:")
    print(out_df["module"].value_counts().sort_index())


if __name__ == "__main__":
    main()
