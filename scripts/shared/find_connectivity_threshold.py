import numpy as np
from pathlib import Path
from typing import Dict, Any


# --- paths -------------------------------------------------------------

ROOT = Path(r"C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")
GROUP_DIR = ROOT / "results" / "group_connectomes"

MATS = {
    "F_ASD": GROUP_DIR / "F_ASD_Zmean.npy",
    "F_CTL": GROUP_DIR / "F_CTL_Zmean.npy",
    "M_ASD": GROUP_DIR / "M_ASD_Zmean.npy",
    "M_CTL": GROUP_DIR / "M_CTL_Zmean.npy",
}


# --- helper functions --------------------------------------------------


def check_matrix(W: np.ndarray) -> None:
    """Basic sanity checks on the connectivity matrix."""
    if W.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape {W.shape}")
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {W.shape}")
    if not np.allclose(W, W.T, atol=1e-6):
        print("[WARN] Matrix is not perfectly symmetric; "
              "results assume undirected graph.")


def is_connected_at_threshold(W: np.ndarray, thr: float) -> bool:
    """
    Check if the graph is connected when keeping edges with |w| >= thr.
    Uses a simple BFS/DFS over the thresholded adjacency.
    """
    n = W.shape[0]

    # adjacency after thresholding (ignore self-loops)
    A = (np.abs(W) >= thr).astype(bool)
    np.fill_diagonal(A, False)

    # if there are no edges at all, not connected
    if not A.any():
        return False

    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True

    while stack:
        i = stack.pop()
        neighbors = np.where(A[i])[0]
        for j in neighbors:
            if not visited[j]:
                visited[j] = True
                stack.append(j)

    return visited.all()


def connectivity_threshold(W: np.ndarray) -> Dict[str, Any]:
    """
    Find the *highest* threshold thr such that the graph with edges
    |w| >= thr is still fully connected.

    This corresponds to the **lowest density** at which the network
    remains connected.
    """
    check_matrix(W)
    n = W.shape[0]

    iu = np.triu_indices(n, k=1)
    weights = np.abs(W[iu])

    # keep only non-zero edges
    weights = weights[weights > 0]

    if weights.size == 0:
        return {
            "threshold": None,
            "num_edges": 0,
            "density": 0.0,
            "note": "All edges are zero; graph cannot be connected.",
        }

    # unique candidate thresholds, sorted descending
    uniq = np.unique(weights)
    uniq.sort()
    candidates = uniq[::-1]  # from largest to smallest

    # scan from sparse to dense; first connected = sparsest connected graph
    for thr in candidates:
        if is_connected_at_threshold(W, thr):
            A = (np.abs(W) >= thr)
            np.fill_diagonal(A, False)
            num_edges = int(A.sum() // 2)
            max_edges = n * (n - 1) // 2
            density = num_edges / max_edges

            return {
                "threshold": float(thr),
                "num_edges": num_edges,
                "density": float(density),
                "note": "Sparsest fully connected graph at this |w| threshold.",
            }

    # if we never hit connectivity even at the smallest non-zero weight,
    # check the graph with all edges (thr = 0.0)
    if is_connected_at_threshold(W, 0.0):
        A = (np.abs(W) > 0.0)
        np.fill_diagonal(A, False)
        num_edges = int(A.sum() // 2)
        max_edges = n * (n - 1) // 2
        density = num_edges / max_edges
        return {
            "threshold": 0.0,
            "num_edges": num_edges,
            "density": float(density),
            "note": "Graph only connected when including all non-zero edges.",
        }

    return {
        "threshold": None,
        "num_edges": 0,
        "density": 0.0,
        "note": "Graph is disconnected even when all edges are present.",
    }


# --- main --------------------------------------------------------------


def main() -> None:
    for label, path in MATS.items():
        print(f"\n=== {label} ===")
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue

        W = np.load(path)
        print(f"  Loaded {path.name}, shape {W.shape}")

        try:
            result = connectivity_threshold(W)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            continue

        print(f"  threshold: {result['threshold']}")
        print(f"  num_edges: {result['num_edges']}")
        print(f"  density : {result['density']:.4f}")
        print(f"  note    : {result['note']}")


if __name__ == "__main__":
    main()
