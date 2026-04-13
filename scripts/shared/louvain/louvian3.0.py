import numpy as np
from pathlib import Path
from collections import Counter
import bct


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "results").exists() and (p / "phenotypes").exists():
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

INPUT_MATRICES = {
    "0.2": ROOT / "results" / "group_connectomes" / "ABIDE12_CC200" / "OVERALL_ageSexMatched_fd-0.2_mean_r.npy",
    "0.3": ROOT / "results" / "group_connectomes" / "ABIDE12_CC200" / "OVERALL_ageSexMatched_fd-0.3_mean_r.npy",
}

OUT_DIR = ROOT / "results" / "group_connectomes" / "ABIDE12_CC200"

# map the optimized gammas to their respective FD cutoffs
GAMMAS = {
    "0.2": 1.65,
    "0.3": 1.60
}

N_TRIALS = 1000
SEED_BASE = 42
K_TARGET = 7
MIN_SIZE = 10
EXPECTED_N = 200


def preprocess_weights(A: np.ndarray) -> np.ndarray:
    W = A.astype(float).copy()
    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)
    return W


def consolidate_small_modules(W: np.ndarray, ci: np.ndarray, min_size: int) -> np.ndarray:
    ci = np.asarray(ci).astype(int).copy()

    while True:
        counts = Counter(ci.tolist())
        small_mods = [m for m, count in counts.items() if count < min_size]

        if not small_mods:
            break

        m_small = min(small_mods, key=lambda x: counts[x])
        nodes_in_m = np.where(ci == m_small)[0]

        other_mods = [m for m in counts.keys() if m != m_small]
        best_neighbor = None
        max_conn = -np.inf

        for m_neighbor in other_mods:
            nodes_in_neighbor = np.where(ci == m_neighbor)[0]
            sub_matrix = W[np.ix_(nodes_in_m, nodes_in_neighbor)]
            avg_conn = float(np.mean(sub_matrix))

            if avg_conn > max_conn:
                max_conn = avg_conn
                best_neighbor = m_neighbor

        if best_neighbor is None:
            break

        ci[nodes_in_m] = best_neighbor

    uniq = sorted(np.unique(ci))
    remap = {old: i + 1 for i, old in enumerate(uniq)}
    return np.array([remap[x] for x in ci], dtype=int)


def run_louvain_ensemble(W: np.ndarray, gamma: float):
    cands = []
    failures = 0

    print(f"Running {N_TRIALS} iterations with Gamma={gamma}...")

    for t in range(N_TRIALS):
        seed = SEED_BASE + t
        try:
            # use seed=seed directly in the BCT function for consistency
            ci, Q = bct.community_louvain(W, gamma=float(gamma), B="negative_asym", seed=seed)
            ci_fixed = consolidate_small_modules(W, ci, MIN_SIZE)
            k = len(np.unique(ci_fixed))
            cands.append({"k": k, "Q": float(Q), "ci": ci_fixed, "seed": seed})
        except Exception:
            failures += 1

    if not cands:
        raise RuntimeError("All Louvain trials failed.")

    print(f"Completed {len(cands)} / {N_TRIALS} iterations (failures={failures})")

    # force exact match for K_TARGET
    matches = [c for c in cands if c["k"] == K_TARGET]
    
    if matches:
        print(f"Found {len(matches)} solutions with exactly {K_TARGET} modules.")
        best = sorted(matches, key=lambda x: -x["Q"])[0]
    else:
        print(f"[WARNING] No solutions with exactly {K_TARGET} modules found! Picking closest.")
        best = sorted(cands, key=lambda x: (abs(x["k"] - K_TARGET), -x["Q"]))[0]

    return best


def main_one_fd(fd_label: str, input_matrix: Path):
    print("\n====================")
    print(f"fd = {fd_label}")
    print("====================")
    print(f"Input: {input_matrix}")

    if not input_matrix.exists():
        print(f"[SKIP] Missing input matrix: {input_matrix}")
        return

    A = np.load(input_matrix)

    if A.shape != (EXPECTED_N, EXPECTED_N):
        raise ValueError(f"{input_matrix.name}: expected {EXPECTED_N}x{EXPECTED_N}, got {A.shape}")

    if not np.isfinite(A).all():
        bad = np.argwhere(~np.isfinite(A))
        raise ValueError(f"{input_matrix.name}: contains NaN/Inf at e.g. {bad[:5].tolist()}")

    W = preprocess_weights(A)
    gamma = GAMMAS[fd_label]

    best = run_louvain_ensemble(W, gamma)
    modules = best["ci"]

    print("\n--- FINAL PARTITION ---")
    print(f"fd cutoff: {fd_label}")
    print(f"Resulting Modules (K): {best['k']}")
    print(f"Modularity (Q) before merge: {best['Q']:.4f}")
    print(f"Best Seed: {best['seed']}")

    uniq, cnts = np.unique(modules, return_counts=True)
    for u, c in zip(uniq, cnts):
        print(f"Module {u}: {c} ROIs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_npy = OUT_DIR / f"ABIDE_modules_asym_min10_fd-{fd_label}.npy"
    out_txt = OUT_DIR / f"ABIDE_modules_asym_min10_fd-{fd_label}.txt"

    np.save(out_npy, modules)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("ROI_index\tModule\n")
        for i, m in enumerate(modules, start=1):
            f.write(f"{i}\t{int(m)}\n")

    print(f"\nSaved modules to: {out_npy}")
    print(f"Saved text file to: {out_txt}")


def main():
    print(f"[INFO] repo root: {ROOT}")
    for fd_label, input_matrix in INPUT_MATRICES.items():
        main_one_fd(fd_label, input_matrix)


if __name__ == "__main__":
    main()