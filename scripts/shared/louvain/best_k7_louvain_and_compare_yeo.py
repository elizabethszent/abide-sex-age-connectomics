# scripts/shared/louvain/make_bestk7_from_mean_z.py

import numpy as np
import pandas as pd
from pathlib import Path

# ===== USER CONFIG =====
ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

MEAN_Z = ROOT / "results" / "group_connectomes" / "ABIDE1_CC200" / "OVERALL_ageSexMatched_fd-0.2_mean_z.npy"
SCENARIO_DIR = ROOT / "results" / "louvain_bestK7_vs_yeo" / "OVERALL_ageSexMatched_fd-0.2"

N_NODES = 200
TARGET_K = 7

DENSITY = 0.10          # keep top 10% |r| edges
GAMMA_MIN = 0.50
GAMMA_MAX = 3.00
GAMMA_STEP = 0.02
REPEATS_PER_GAMMA = 10  # number of random seeds per gamma
SEED0 = 123             # base seed
# =======================


def load_mean_z(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mean-z file not found: {path}")
    z = np.load(path)
    z = np.asarray(z, dtype=float)
    if z.shape != (N_NODES, N_NODES):
        raise ValueError(f"Expected {(N_NODES, N_NODES)}, got {z.shape}")
    # zero diagonal
    np.fill_diagonal(z, 0.0)
    return z


def threshold_top_density_abs_r(z: np.ndarray, density: float) -> np.ndarray:
    """
    Convert Fisher-z -> r, take abs, keep top `density` of undirected edges.
    Returns weighted adjacency W (NxN) with zeros elsewhere.
    """
    r = np.tanh(z)
    w = np.abs(r)
    np.fill_diagonal(w, 0.0)

    # upper triangle edges
    iu = np.triu_indices_from(w, k=1)
    edges = w[iu]
    n_edges = edges.size
    k_keep = int(np.floor(density * n_edges))

    if k_keep <= 0:
        raise ValueError("density too small -> keeps 0 edges")
    if k_keep >= n_edges:
        # keep all
        W = w.copy()
        return W

    # Find threshold value for top-k (kth largest)
    # Use partition for efficiency
    kth = np.partition(edges, n_edges - k_keep)[n_edges - k_keep]
    keep_mask = w >= kth

    W = np.zeros_like(w)
    W[keep_mask] = w[keep_mask]
    np.fill_diagonal(W, 0.0)
    return W


def run_louvain_bestk7(W: np.ndarray):
    """
    Use NetworkX Louvain (resolution=gamma) and choose best partition with K=TARGET_K.
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        from networkx.algorithms.community.quality import modularity
    except Exception as e:
        raise RuntimeError(
            "This script needs networkx with louvain_communities.\n"
            "Install/upgrade with:\n"
            "  pip install -U networkx\n"
        ) from e

    # Build weighted graph
    G = nx.Graph()
    G.add_nodes_from(range(N_NODES))
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            w = float(W[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)

    gammas = np.arange(GAMMA_MIN, GAMMA_MAX + 1e-9, GAMMA_STEP)

    rows = []
    best = None  # (Q, gamma, seed, communities)

    for gi, gamma in enumerate(gammas):
        for rep in range(REPEATS_PER_GAMMA):
            seed = SEED0 + gi * 1000 + rep
            comms = louvain_communities(G, weight="weight", resolution=float(gamma), seed=int(seed))
            k = len(comms)
            Q = modularity(G, comms, weight="weight")

            rows.append({"gamma": float(gamma), "seed": int(seed), "k": int(k), "Q": float(Q)})

            if k == TARGET_K:
                if best is None or Q > best[0]:
                    best = (Q, float(gamma), int(seed), comms)

    df = pd.DataFrame(rows).sort_values(["gamma", "seed"]).reset_index(drop=True)
    return best, df


def communities_to_modules(comms) -> np.ndarray:
    """
    Convert list-of-sets communities -> module labels 1..K (length N_NODES).
    Label modules by size descending for stability.
    """
    # sort communities by size desc
    comms_sorted = sorted(comms, key=lambda s: (-len(s), min(s)))
    mods = np.zeros(N_NODES, dtype=int)
    for idx, nodeset in enumerate(comms_sorted, start=1):
        for n in nodeset:
            mods[int(n)] = idx
    return mods


def save_modules(mods: np.ndarray, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{prefix}_BESTK7_modules.npy"
    txt_path = out_dir / f"{prefix}_BESTK7_modules.txt"

    np.save(npy_path, mods)

    # one label per line (200 lines)
    txt_path.write_text("\n".join(str(int(x)) for x in mods.tolist()) + "\n", encoding="utf-8")

    return npy_path, txt_path


def main():
    print(f"[INFO] Loading mean_z: {MEAN_Z}")
    z = load_mean_z(MEAN_Z)

    print(f"[INFO] Thresholding to top {DENSITY*100:.1f}% |r| edges")
    W = threshold_top_density_abs_r(z, DENSITY)

    print("[INFO] Running Louvain gamma sweep, looking for K=7")
    best, sweep = run_louvain_bestk7(W)

    sweep_path = SCENARIO_DIR / "OVERALL_ageSexMatched_fd-0.2_gamma_sweep_summary_K7focus.csv"
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(sweep_path, index=False)
    print(f"[SAVED] {sweep_path}")

    if best is None:
        # No K=7 found — still saved the sweep so you can adjust gamma range/step.
        bestk = sweep.groupby("k")["Q"].max().sort_values(ascending=False).head(10)
        raise RuntimeError(
            "No Louvain partition with K=7 found in the gamma sweep.\n"
            "Try widening GAMMA_MIN/GAMMA_MAX or reducing GAMMA_STEP.\n"
            f"Top Q by k (max over sweep):\n{bestk}\n"
        )

    Q, gamma, seed, comms = best
    print(f"[BEST] Found K=7 with Q={Q:.6f} at gamma={gamma} seed={seed}")

    mods = communities_to_modules(comms)
    prefix = "OVERALL_ageSexMatched_fd-0.2"
    npy_path, txt_path = save_modules(mods, SCENARIO_DIR, prefix)
    print(f"[SAVED] {npy_path}")
    print(f"[SAVED] {txt_path}")

    # quick sanity
    unique = sorted(set(mods.tolist()))
    counts = {u: int(np.sum(mods == u)) for u in unique}
    print(f"[INFO] Module labels: {unique}")
    print(f"[INFO] Module sizes: {counts}")


if __name__ == "__main__":
    main()