import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
CONN_DIR = BASE / r"data\connectomes\cpac\nofilt_noglobal\cc200_z"
MODULE_PATH = BASE / r"results\group_connectomes\CC200_modules.npy"

FEMALE_META = BASE / r"data\female\female_metadata_included.csv"
MALE_META   = BASE / r"data\male\male_metadata_included.csv"

OUT_DIR = BASE / r"results\hubs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

modules = np.load(MODULE_PATH).astype(int)
if modules.ndim != 1 or modules.shape[0] != 200:
    raise ValueError("Expected modules of shape (200,)")

# relabel to 0..K-1
uniq = np.unique(modules)
mapping = {old: i for i, old in enumerate(uniq)}
modules = np.array([mapping[x] for x in modules], dtype=int)
K = len(uniq)
print(f"K = {K} modules")

def fisher_to_weight(Z):
    # convert Fisher-z back to r
    return np.tanh(Z)

def modularity_Q(W, comm):
    """Weighted undirected modularity with fixed community labels."""
    W = W.copy()
    W[W < 0] = 0.0
    k = W.sum(axis=1)
    m = k.sum() / 2.0
    if m <= 0:
        return 0.0
    Q = 0.0
    for i in range(len(W)):
        for j in range(len(W)):
            if comm[i] == comm[j]:
                Q += W[i, j] - k[i] * k[j] / (2 * m)
    return Q / (2 * m)

def participation_coefficient(W, comm):
    """Participation coefficient per node, positive weights only."""
    W = W.copy()
    W[W < 0] = 0.0
    k = W.sum(axis=1)
    PC = np.zeros_like(k)

    for s in range(K):
        idx_s = np.where(comm == s)[0]
        if idx_s.size == 0:
            continue
        k_is = W[:, idx_s].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            PC += (k_is / k) ** 2

    PC = 1.0 - PC
    PC[~np.isfinite(PC)] = 0.0
    return PC

def within_module_z(W, comm):
    """Within-module degree z-score per node."""
    W = W.copy()
    W[W < 0] = 0.0
    z = np.zeros(len(W))

    for s in range(K):
        idx = np.where(comm == s)[0]
        if idx.size <= 1:
            continue
        k_in = W[np.ix_(idx, idx)].sum(axis=1)
        mu = k_in.mean()
        sd = k_in.std(ddof=1)
        if sd == 0:
            z[idx] = 0.0
        else:
            z[idx] = (k_in - mu) / sd
    return z

def conn_path(fid):
    # adjust if needed
    return CONN_DIR / f"{fid}.npy"

def process_group(label, meta_csv):
    meta = pd.read_csv(meta_csv)
    rows = []
    skipped = 0

    for _, row in meta.iterrows():
        fid = str(row["FILE_ID"])
        cp = conn_path(fid)
        if not cp.exists():
            print(f"[{label}] missing {fid} at {cp}")
            skipped += 1
            continue

        Z = np.load(cp)
        if Z.shape != (200, 200):
            print(
                f"[{label}] skipping {fid}: connectome shape {Z.shape}, "
                "expected (200, 200)"
            )
            skipped += 1
            continue

        W = fisher_to_weight(Z)
        np.fill_diagonal(W, 0.0)

        Q = modularity_Q(W, modules)
        PC = participation_coefficient(W, modules)
        Zm = within_module_z(W, modules)

        data = {
            "FILE_ID": fid,
            "DX_GROUP": row.get("DX_GROUP", np.nan),
            "AGE_AT_SCAN": row.get("AGE_AT_SCAN", np.nan),
            "SITE_ID": row.get("SITE_ID", np.nan),
            "func_mean_fd": row.get("func_mean_fd", np.nan),
            "Q_fixed": Q,
        }
        for s in range(K):
            idx = np.where(modules == s)[0]
            data[f"PC_med_m{s}"] = np.median(PC[idx])
            data[f"Z_med_m{s}"]  = np.median(Zm[idx])

        rows.append(data)

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / f"{label}_hub_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(out)} rows (skipped {skipped})")

process_group("female", FEMALE_META)
process_group("male",   MALE_META)
