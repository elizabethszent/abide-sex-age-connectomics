import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import nibabel as nib
from nilearn import datasets, image

import bct  # pip install bctpy


# =========================
# PATHS
# =========================
BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

IN_DIR = BASE / r"results\group_connectomes\ABIDE1_CC200"
INPUTS = [
    IN_DIR / "OVERALL_sexbalanced_fd-0.2_mean_z.npy",
    IN_DIR / "OVERALL_sexbalanced_fd-0.3_mean_z.npy",
    IN_DIR / "OVERALL_ageSexMatched_fd-0.2_mean_z.npy",
    IN_DIR / "OVERALL_ageSexMatched_fd-0.3_mean_z.npy",
]

OUT_DIR = BASE / r"results\louvain_bestK7_vs_yeo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CC200_ATLAS = BASE / r"atlases\cc200\cc200_roi_atlas.nii.gz"


# =========================
# LOUVAIN SETTINGS
# =========================
EXPECTED_N = 200
TARGET_K = 7
MIN_SIZE = 10

SIGNED_MODE = "negative_asym"
NEG_SCALE = 0.5

# Gamma sweep
GAMMA_START = 0.80
GAMMA_END   = 2.00
GAMMA_STEP  = 0.02

# Restarts
N_SWEEP_TRIALS = 500   # per gamma (to estimate how often K=7 happens + K7-stability)
N_FINAL_TRIALS = 1000  # final best partition at chosen gamma
SEED_BASE = 42

# Require at least this fraction of runs to be valid K=7 (helps avoid flaky gammas)
MIN_K7_FRAC_FOR_SELECTION = 0.10  # set 0.20 if you want stricter


# =========================
# YEO SETTINGS
# =========================
YEO_LABEL_MAP = {
    0: "Background",
    1: "Visual",
    2: "Somatomotor",
    3: "DorsalAttention",
    4: "VentralAttention",
    5: "Limbic",
    6: "Frontoparietal",
    7: "DefaultMode",
}


# =========================
# HELPERS
# =========================
def preprocess_signed_weights(A: np.ndarray, neg_scale: float) -> np.ndarray:
    W = A.astype(float).copy()
    np.fill_diagonal(W, 0.0)

    if neg_scale != 1.0:
        W[W < 0] *= float(neg_scale)

    W = 0.5 * (W + W.T)

    if not np.isfinite(W).all():
        bad = np.argwhere(~np.isfinite(W))
        raise ValueError(f"W contains NaN/Inf at e.g. {bad[:5].tolist()}")

    return W


def relabel_ci_to_1based(ci: np.ndarray) -> np.ndarray:
    ci = np.asarray(ci).astype(int)
    uniq = sorted(set(ci.tolist()))
    remap = {lab: i + 1 for i, lab in enumerate(uniq)}
    return np.array([remap[x] for x in ci], dtype=int)


def run_louvain_restarts(W: np.ndarray, gamma: float, n_trials: int, seed_base: int):
    results = []
    failures = 0
    for t in range(n_trials):
        seed = seed_base + t
        try:
            np.random.seed(seed)
            ci, Q = bct.community_louvain(W, gamma=float(gamma), B=SIGNED_MODE)
            ci = relabel_ci_to_1based(ci)
            sizes = Counter(ci.tolist())
            results.append({
                "k": len(sizes),
                "min_size": min(sizes.values()),
                "Q": float(Q),
                "seed": int(seed),
                "ci": ci,
            })
        except Exception:
            failures += 1
    return results, failures


def stability_from_coassignment(partitions: list[np.ndarray]) -> float:
    """
    Stability in [0,1] from co-assignment probabilities.
    1 - 4*E[p(1-p)] over i<j. High means consistent partitions.
    """
    if len(partitions) == 0:
        return 0.0
    n = len(partitions[0])
    C = np.zeros((n, n), dtype=np.int32)
    for ci in partitions:
        C += (ci[:, None] == ci[None, :]).astype(np.int32)
    P = C.astype(np.float32) / float(len(partitions))
    iu = np.triu_indices(n, k=1)
    p = P[iu]
    return float(1.0 - 4.0 * np.mean(p * (1.0 - p)))


def pick_best_k7(results: list[dict]) -> dict | None:
    """Pick best partition among valid K=7 partitions (min_size>=MIN_SIZE), by max Q."""
    k7 = [r for r in results if (r["k"] == TARGET_K and r["min_size"] >= MIN_SIZE)]
    if not k7:
        return None
    return max(k7, key=lambda r: r["Q"])


def save_partition_txt(modules: np.ndarray, out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("ROI_index\tModule\n")
        for i, m in enumerate(modules, start=1):
            f.write(f"{i}\t{int(m)}\n")


# ---------- Yeo mapping ----------
def pick_yeo7_path(yeo_bunch) -> str:
    candidates = ["thin_7", "thick_7", "yeo_7", "maps_7", "yeo7", "thin7", "thick7"]
    for k in candidates:
        if hasattr(yeo_bunch, k):
            p = getattr(yeo_bunch, k)
            if isinstance(p, str) and p.endswith((".nii", ".nii.gz")):
                return p
    for key in getattr(yeo_bunch, "keys", lambda: [])():
        v = yeo_bunch[key]
        if isinstance(v, str) and v.endswith((".nii", ".nii.gz")):
            name = Path(v).name.lower()
            if ("7" in name) and ("17" not in name):
                return v
    raise RuntimeError("Could not find a Yeo7 nifti path from fetch_atlas_yeo_2011().")


def atlas_roi_labels(atlas_data: np.ndarray) -> np.ndarray:
    vals = np.unique(atlas_data.astype(int))
    vals = vals[vals != 0]
    return np.sort(vals)


def roi_yeo_label_cortical_vote(cc_data: np.ndarray, yeo_data: np.ndarray, roi_label: int) -> int:
    mask = (cc_data == roi_label)
    if not np.any(mask):
        return 0
    vals = yeo_data[mask].astype(int)
    cortical = vals[vals != 0]
    if cortical.size == 0:
        return 0
    labs, cnts = np.unique(cortical, return_counts=True)
    return int(labs[np.argmax(cnts)])


def nmi_from_labels(x: np.ndarray, y: np.ndarray) -> float:
    """
    Normalized Mutual Information between two labelings.
    Returns 0..1-ish (can be slightly >1 due to float error; we clamp).
    """
    x = np.asarray(x).astype(int)
    y = np.asarray(y).astype(int)
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size != y.size:
        raise ValueError(f"NMI: x and y must be same length, got {x.size} vs {y.size}")

    # relabel to 0..K-1
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)

    n = xi.size
    Kx = int(xi.max()) + 1
    Ky = int(yi.max()) + 1

    # contingency
    M = np.zeros((Kx, Ky), dtype=np.int64)
    np.add.at(M, (xi, yi), 1)

    Pxy = M.astype(np.float64) / n
    Px = Pxy.sum(axis=1)              # (Kx,)
    Py = Pxy.sum(axis=0)              # (Ky,)
    denom = Px[:, None] * Py[None, :] # (Kx,Ky)

    nz = Pxy > 0
    MI = float(np.sum(Pxy[nz] * np.log(Pxy[nz] / denom[nz])))

    Hx = float(-np.sum(Px[Px > 0] * np.log(Px[Px > 0])))
    Hy = float(-np.sum(Py[Py > 0] * np.log(Py[Py > 0])))

    if Hx <= 0 or Hy <= 0:
        return 0.0

    nmi = MI / np.sqrt(Hx * Hy)
    # clamp tiny numeric overshoot
    return float(max(0.0, min(1.0, nmi)))


def yeo_purity(louvain: np.ndarray, yeo: np.ndarray) -> float:
    mask = (yeo != 0)
    l = louvain[mask]
    y = yeo[mask]
    if l.size == 0:
        return 0.0
    s = 0
    for m in np.unique(l):
        idx = (l == m)
        counts = Counter(y[idx].tolist())
        s += max(counts.values())
    return float(s / l.size)


def compare_to_yeo(modules: np.ndarray, out_subdir: Path):
    if not CC200_ATLAS.exists():
        raise FileNotFoundError(f"Missing CC200 atlas: {CC200_ATLAS}")

    cc_img = nib.load(str(CC200_ATLAS))
    cc_data = cc_img.get_fdata().astype(int)
    roi_labels = atlas_roi_labels(cc_data)
    N = min(EXPECTED_N, len(roi_labels))

    yeo = datasets.fetch_atlas_yeo_2011()
    yeo7_path = pick_yeo7_path(yeo)
    yeo_img = nib.load(yeo7_path)
    yeo_rs = image.resample_to_img(yeo_img, cc_img, interpolation="nearest")
    yeo_data = yeo_rs.get_fdata().astype(int)

    yeo_lab = np.zeros(N, dtype=int)
    for i in range(N):
        yeo_lab[i] = roi_yeo_label_cortical_vote(cc_data, yeo_data, int(roi_labels[i]))

    df = pd.DataFrame({
        "ROI_index": np.arange(1, N + 1),
        "Louvain_module": modules[:N].astype(int),
        "Yeo7_label": yeo_lab.astype(int),
        "Yeo7_name": [YEO_LABEL_MAP[int(x)] for x in yeo_lab],
    })

    # Tables
    ctab = pd.crosstab(df["Louvain_module"], df["Yeo7_name"])
    ctab.to_csv(out_subdir / "BESTK7_module_x_yeo_counts.csv")
    ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100.0
    ctab_pct.to_csv(out_subdir / "BESTK7_module_x_yeo_rowpct.csv")

    # Metrics (exclude Background)
    purity = yeo_purity(df["Louvain_module"].to_numpy(), df["Yeo7_label"].to_numpy())
    nmi = nmi_from_labels(
        df.loc[df["Yeo7_label"] != 0, "Louvain_module"].to_numpy(),
        df.loc[df["Yeo7_label"] != 0, "Yeo7_label"].to_numpy(),
    )
    bg = int((df["Yeo7_label"] == 0).sum())

    metrics = pd.DataFrame([{
        "purity_excl_background": purity,
        "nmi_excl_background": nmi,
        "background_rois": bg,
        "cortex_rois_used": int(N - bg),
    }])
    metrics.to_csv(out_subdir / "BESTK7_yeo_metrics.csv", index=False)

    df.to_csv(out_subdir / "BESTK7_per_roi_yeo_mapping.csv", index=False)


# =========================
# PICK GAMMA (K=7 REQUIRED)
# =========================
def choose_gamma_k7(sweep_df: pd.DataFrame) -> float:
    """
    Choose one gamma with K=7 requirement:
      - require k7_frac >= MIN_K7_FRAC_FOR_SELECTION
      - maximize stability_k7 (stability computed over K=7 partitions only)
      - tie-break by Q_best_k7
      - tie-break by k7_frac
    """
    d = sweep_df.copy()
    d = d[d["k7_frac"] >= MIN_K7_FRAC_FOR_SELECTION]
    if d.empty:
        raise RuntimeError(
            f"No gammas had k7_frac >= {MIN_K7_FRAC_FOR_SELECTION}. "
            "Increase gamma range or lower MIN_K7_FRAC_FOR_SELECTION."
        )
    d = d.sort_values(["stability_k7", "Q_best_k7", "k7_frac"], ascending=False)
    return float(d.iloc[0]["gamma"])


def main_one_matrix(fp: Path):
    label = fp.stem.replace("_mean_z", "")
    print(f"\n==============================")
    print(f"Matrix: {label}")
    print(f"File  : {fp}")
    print(f"==============================")

    A = np.load(fp)
    if A.shape != (EXPECTED_N, EXPECTED_N):
        raise ValueError(f"{fp.name}: expected {EXPECTED_N}x{EXPECTED_N}, got {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError(f"{fp.name}: contains NaN/Inf")

    W = preprocess_signed_weights(A, NEG_SCALE)

    gammas = np.round(np.arange(GAMMA_START, GAMMA_END + 1e-9, GAMMA_STEP), 2)

    label_dir = OUT_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"[SWEEP] gammas {GAMMA_START:.2f}..{GAMMA_END:.2f} step {GAMMA_STEP:.2f} | trials/gamma={N_SWEEP_TRIALS}")
    for gamma in gammas:
        results, failures = run_louvain_restarts(W, gamma, N_SWEEP_TRIALS, SEED_BASE)
        if not results:
            rows.append({
                "gamma": float(gamma),
                "runs_ok": 0,
                "failures": int(failures),
                "k7_frac": 0.0,
                "k_counts": "",
                "stability_k7": 0.0,
                "Q_best_k7": np.nan,
                "min_size_best_k7": np.nan,
                "seed_best_k7": np.nan,
            })
            continue

        ks = [r["k"] for r in results]
        k_counts = Counter(ks)
        k7_total = k_counts.get(TARGET_K, 0)
        k7_frac = float(k7_total / len(results))

        # restrict to valid K=7 with min_size constraint for "network analysis"
        k7_valid = [r for r in results if (r["k"] == TARGET_K and r["min_size"] >= MIN_SIZE)]
        stability_k7 = stability_from_coassignment([r["ci"] for r in k7_valid])

        best_k7 = pick_best_k7(results)
        if best_k7 is None:
            Q_best_k7 = np.nan
            min_best_k7 = np.nan
            seed_best_k7 = np.nan
        else:
            Q_best_k7 = float(best_k7["Q"])
            min_best_k7 = int(best_k7["min_size"])
            seed_best_k7 = int(best_k7["seed"])

        rows.append({
            "gamma": float(gamma),
            "runs_ok": int(len(results)),
            "failures": int(failures),
            "k7_frac": float(k7_frac),
            "k_counts": "; ".join([f"{k}:{v}" for k, v in sorted(k_counts.items())]),
            "stability_k7": float(stability_k7),
            "Q_best_k7": Q_best_k7,
            "min_size_best_k7": min_best_k7,
            "seed_best_k7": seed_best_k7,
        })

        print(f"  gamma={gamma:.2f}  k7_frac={k7_frac:.3f}  stab_k7={stability_k7:.3f}  Q_best_k7={Q_best_k7 if np.isfinite(Q_best_k7) else 'NA'}")

    sweep = pd.DataFrame(rows)
    out_sweep = label_dir / f"{label}_gamma_sweep_summary_K7focus.csv"
    sweep.to_csv(out_sweep, index=False)
    print(f"[SAVED] {out_sweep}")

    # Choose gamma under K=7 requirement
    best_gamma = choose_gamma_k7(sweep)
    print(f"[CHOSEN] gamma={best_gamma:.2f} (maximize stability_k7 then Q_best_k7; require k7_frac>={MIN_K7_FRAC_FOR_SELECTION})")

    # Final run at chosen gamma (1000 restarts)
    print(f"[FINAL] gamma={best_gamma:.2f} | trials={N_FINAL_TRIALS}")
    final_results, final_fail = run_louvain_restarts(W, best_gamma, N_FINAL_TRIALS, SEED_BASE)

    # pick best valid K=7
    best_final = pick_best_k7(final_results)
    if best_final is None:
        raise RuntimeError(
            f"At gamma={best_gamma:.2f}, no valid K=7 partitions met MIN_SIZE={MIN_SIZE} "
            f"within {N_FINAL_TRIALS} trials. Try raising gamma range or lowering MIN_SIZE."
        )

    modules = best_final["ci"]
    sizes = Counter(modules.tolist())

    out_npy = label_dir / f"{label}_BESTK7_modules.npy"
    out_txt = label_dir / f"{label}_BESTK7_modules.txt"
    np.save(out_npy, modules)
    save_partition_txt(modules, out_txt)

    print(f"[BESTK7] K={len(sizes)}  Q={best_final['Q']:.4f}  min_size={min(sizes.values())}  seed={best_final['seed']}  failures={final_fail}")
    print("         module sizes:", ", ".join([f"{k}:{v}" for k, v in sorted(sizes.items())]))
    print(f"[SAVED] {out_npy}")
    print(f"[SAVED] {out_txt}")

    # Compare to Yeo
    print("[YEO] Comparing BESTK7 partition to Yeo7 (metrics exclude Background)...")
    compare_to_yeo(modules, label_dir)
    print(f"[SAVED] {label_dir / 'BESTK7_yeo_metrics.csv'}")
    print(f"[SAVED] {label_dir / 'BESTK7_module_x_yeo_counts.csv'}")


def main():
    for fp in INPUTS:
        if not fp.exists():
            print(f"[SKIP] Missing input matrix: {fp}")
            continue
        main_one_matrix(fp)

    print("\nDone for all 4 matrices.")


if __name__ == "__main__":
    main()