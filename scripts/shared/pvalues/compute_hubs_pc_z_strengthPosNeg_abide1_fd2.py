# scripts/shared/pvalues/compute_hubs_pc_z_strengthPosNeg_abide1_fd2.py
#
# Computes per-node hub features (PC / Z / strength) split into:
#   - abs (PC, z) computed on |W|
#   - pos (PC_pos, z_pos) computed on W+
#   - neg (PC_neg, z_neg) computed on |W-| magnitudes
# plus strength_pos / strength_neg
#
# Outputs one CSV per sex x age_group:
#   results/hubs/<sex>_<age>_pc_z_strengthPosNeg_abide1_fd2.csv

import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# ---- ABIDE1 connectomes (per-subject, per-run) ----
CONN_DIR = ROOT / "connectomes" / "CC200" / "ABIDE1" / "FDpersubject2"

# ---- Louvain modules file (ROI_index Module) OR a 200-line vector ----
MODULE_FILE = ROOT / "results" / "group_connectomes" / "CC200_modules_ALLSUBJ_signed_asym1000.txt"

# ---- Metadata files (must contain FILE_ID, DX_GROUP, AGE_AT_SCAN) ----
FEMALE_META = ROOT / "data" / "female" / "female_metadata_included.csv"
MALE_META   = ROOT / "data" / "male"   / "male_metadata_included.csv"

OUT_DIR = ROOT / "results" / "hubs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Keep your edge density
DENSITY = 0.0797

# Age bins
BINS   = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
RIGHT  = False  # [0,10), [10,13), [13,18), [18,200)

AGE_GROUPS = LABELS

# Optional: auto-convert Fisher z -> r if values look like z
AUTO_TANH_IF_Z = True
Z_LIKE_ABS_MAX = 1.5  # if max(|W|) > this, likely Fisher z


# ----------------------------
# Load modules
# ----------------------------
def load_modules(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")

    if path.suffix.lower() == ".npy":
        v = np.load(path).reshape(-1).astype(int)
        return v

    txt = path.read_text(encoding="utf-8").strip().splitlines()
    if not txt:
        raise ValueError(f"Empty module file: {path}")

    # "ROI_index Module" table
    if ("ROI_index" in txt[0]) and ("Module" in txt[0]):
        df = pd.read_csv(path, sep=r"\s+")
        df = df.sort_values("ROI_index")
        return df["Module"].to_numpy().astype(int)

    # else: one int per line
    vals = []
    for line in txt:
        line = line.strip()
        if line:
            vals.append(int(line))
    return np.array(vals, dtype=int)


roi2mod = load_modules(MODULE_FILE)
N_ROI   = int(len(roi2mod))
N_MOD   = int(np.max(roi2mod))

print(f"Loaded module file with {N_ROI} ROIs, {N_MOD} modules")


# ----------------------------
# Build connectome index from disk
# ----------------------------
def build_connectome_index(conn_dir: Path) -> dict[int, Path]:
    """
    Build mapping: subject_int -> best .npy path
    Prefers smallest run number (run-1 if present).
    Handles filenames like:
      sub-0050002_task-rest_run-1.npy
      sub-50002_task-rest_run-1.npy
      sub-28743_ses-1_task-rest_run-1.npy  (still extracts sub id)
    """
    if not conn_dir.exists():
        raise FileNotFoundError(f"Connectome directory not found: {conn_dir}")

    files = list(conn_dir.glob("*.npy"))
    rx_sub = re.compile(r"sub-(\d+)")
    rx_run = re.compile(r"run-(\d+)")

    best: dict[int, tuple[int, Path]] = {}

    for fp in files:
        ms = rx_sub.search(fp.name)
        if not ms:
            continue
        subj_int = int(ms.group(1))  # normalize (kills leading zeros)

        mr = rx_run.search(fp.name)
        run = int(mr.group(1)) if mr else 999

        # keep the lowest run number
        if subj_int not in best or run < best[subj_int][0]:
            best[subj_int] = (run, fp)

    return {k: v[1] for k, v in best.items()}


CONN_INDEX = build_connectome_index(CONN_DIR)
print(f"[INFO] CONN_DIR={CONN_DIR} (*.npy count={len(list(CONN_DIR.glob('*.npy')))})")
print(f"[INFO] Unique subjects indexed from connectomes = {len(CONN_INDEX)}")


# ----------------------------
# Helpers
# ----------------------------
def threshold_top_density(mat: np.ndarray, density: float) -> np.ndarray:
    """
    Keep top 'density' fraction of absolute edges (upper triangle),
    preserve original sign/weight, symmetrize, zero diagonal.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix not square: {mat.shape}")
    n = mat.shape[0]

    W = mat.astype(float, copy=True)
    np.fill_diagonal(W, 0.0)

    iu, ju = np.triu_indices(n, k=1)
    vals = np.abs(W[iu, ju])

    m = len(vals)
    k = int(np.floor(density * m))
    if k < 1:
        out = np.zeros_like(W)
        np.fill_diagonal(out, 0.0)
        return out

    thresh = np.partition(vals, -k)[-k]
    mask = np.abs(W) >= thresh

    out = np.where(mask, W, 0.0)
    out = np.triu(out, 1)
    out = out + out.T
    np.fill_diagonal(out, 0.0)
    return out


def compute_pc_z_strength_posneg(W: np.ndarray, roi2mod: np.ndarray):
    """
    Returns:
      PC_abs, Z_abs computed on ABS weights
      PC_pos, Z_pos computed on positive weights only
      PC_neg, Z_neg computed on negative magnitudes only
      strength_pos, strength_neg (neg is magnitude)
    """
    if W.shape != (N_ROI, N_ROI):
        raise ValueError(f"Expected {(N_ROI, N_ROI)}, got {W.shape}")

    N = W.shape[0]
    M = int(roi2mod.max())

    # Split weights
    Wp = np.where(W > 0, W, 0.0)
    Wn = np.where(W < 0, -W, 0.0)   # magnitude of negatives (positive-valued)
    Wabs = Wp + Wn                  # == np.abs(W)

    # Strengths
    strength_pos = Wp.sum(axis=1)
    strength_neg = Wn.sum(axis=1)
    strength_abs = Wabs.sum(axis=1)

    def k_by_mod_from(WX: np.ndarray) -> np.ndarray:
        kbm = np.zeros((N, M), dtype=float)
        for m in range(1, M + 1):
            idx = np.where(roi2mod == m)[0]
            if idx.size == 0:
                continue
            kbm[:, m - 1] = WX[:, idx].sum(axis=1)
        return kbm

    k_abs = k_by_mod_from(Wabs)
    k_pos = k_by_mod_from(Wp)
    k_neg = k_by_mod_from(Wn)

    def pc_from(kbm: np.ndarray, strength: np.ndarray) -> np.ndarray:
        pc = np.zeros(N, dtype=float)
        for i in range(N):
            ki = strength[i]
            if ki <= 0:
                pc[i] = 0.0
            else:
                frac = kbm[i, :] / ki
                pc[i] = 1.0 - np.sum(frac ** 2)
        return pc

    PC_abs = pc_from(k_abs, strength_abs)
    PC_pos = pc_from(k_pos, strength_pos)
    PC_neg = pc_from(k_neg, strength_neg)

    def z_from(kbm: np.ndarray) -> np.ndarray:
        z = np.zeros(N, dtype=float)
        for m in range(1, M + 1):
            idx = np.where(roi2mod == m)[0]
            if idx.size == 0:
                continue
            k_within = kbm[idx, m - 1]
            mu = k_within.mean()
            sd = k_within.std(ddof=1)
            z[idx] = (k_within - mu) / sd if sd > 0 else 0.0
        return z

    Z_abs = z_from(k_abs)
    Z_pos = z_from(k_pos)
    Z_neg = z_from(k_neg)

    return PC_abs, PC_pos, PC_neg, Z_abs, Z_pos, Z_neg, strength_pos, strength_neg


def add_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
    df["AGE_GROUP"] = pd.cut(
        df["AGE_AT_SCAN"],
        bins=BINS,
        labels=LABELS,
        right=RIGHT,
        include_lowest=True
    )
    return df


def normalize_matrix_if_needed(mat: np.ndarray) -> np.ndarray:
    W = mat.astype(float, copy=False)
    if not AUTO_TANH_IF_Z:
        return W
    mx = float(np.nanmax(np.abs(W))) if np.size(W) else 0.0
    if np.isfinite(mx) and mx > Z_LIKE_ABS_MAX:
        return np.tanh(W)
    return W


def file_id_to_subject_int(file_id: str) -> int | None:
    """
    Extract a subject id integer from FILE_ID.
    Works for:
      'NYU_0051062', '0051062', '50002', '50002.0', 'Leuven_2_0050736', etc.
    """
    s = str(file_id).strip()
    s = re.sub(r"\.0$", "", s)  # strip trailing .0

    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return int(m.group(1))


def read_meta(meta_path: Path) -> pd.DataFrame:
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta = pd.read_csv(meta_path)
    meta.columns = meta.columns.str.strip()
    meta = meta.rename(columns={c: c.upper() for c in meta.columns})

    required = {"FILE_ID", "DX_GROUP", "AGE_AT_SCAN"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"{meta_path} missing columns: {missing}")

    meta["FILE_ID"] = meta["FILE_ID"].astype(str).str.strip()
    meta["DX_GROUP"] = pd.to_numeric(meta["DX_GROUP"], errors="coerce")

    meta = add_age_group_column(meta)
    return meta


def process_sex(sex_label: str, meta_path: Path):
    print(f"\n{sex_label.upper()}")

    meta = read_meta(meta_path)

    for age_group in AGE_GROUPS:
        sub = meta[meta["AGE_GROUP"] == age_group].copy()
        if sub.empty:
            print(f"  [{sex_label} {age_group}] No subjects, skipping.")
            continue

        print(f"  [{sex_label} {age_group}] N={len(sub)}")
        print(sub["DX_GROUP"].value_counts().rename({1: "ASD", 2: "CTL"}))

        rows = []
        n_missing = 0
        n_shape = 0
        n_nonfinite = 0
        n_badid = 0

        for _, row in sub.iterrows():
            fid = row["FILE_ID"]
            subj_int = file_id_to_subject_int(fid)
            if subj_int is None:
                n_badid += 1
                continue

            conn_fp = CONN_INDEX.get(subj_int, None)
            if conn_fp is None or not conn_fp.exists():
                n_missing += 1
                continue

            mat = np.load(conn_fp)
            if mat.shape != (N_ROI, N_ROI):
                n_shape += 1
                continue
            if not np.isfinite(mat).all():
                n_nonfinite += 1
                continue

            W = normalize_matrix_if_needed(mat)
            W = threshold_top_density(W, density=DENSITY)
            W = 0.5 * (W + W.T)
            np.fill_diagonal(W, 0.0)

            PC_abs, PC_pos, PC_neg, Z_abs, Z_pos, Z_neg, strength_pos, strength_neg = \
                compute_pc_z_strength_posneg(W, roi2mod)

            dx_val = int(row["DX_GROUP"]) if pd.notna(row["DX_GROUP"]) else np.nan
            age_val = float(row["AGE_AT_SCAN"]) if pd.notna(row["AGE_AT_SCAN"]) else np.nan

            for node_idx in range(N_ROI):
                rows.append({
                    "FILE_ID": fid,
                    "subject_int": subj_int,
                    "connectome_file": conn_fp.name,
                    "sex": sex_label,
                    "AGE_GROUP": age_group,
                    "DX_GROUP": dx_val,
                    "AGE_AT_SCAN": age_val,
                    "node": node_idx + 1,
                    "module": int(roi2mod[node_idx]),

                    # ABS
                    "PC": float(PC_abs[node_idx]),
                    "z": float(Z_abs[node_idx]),

                    # POS / NEG
                    "PC_pos": float(PC_pos[node_idx]),
                    "PC_neg": float(PC_neg[node_idx]),
                    "z_pos": float(Z_pos[node_idx]),
                    "z_neg": float(Z_neg[node_idx]),

                    # Strengths
                    "strength_pos": float(strength_pos[node_idx]),
                    "strength_neg": float(strength_neg[node_idx]),
                })

        if not rows:
            print(
                f"  [{sex_label} {age_group}] No usable subjects after QC. "
                f"(badid={n_badid}, missing={n_missing}, bad_shape={n_shape}, nonfinite={n_nonfinite})"
            )
            continue

        out_df = pd.DataFrame(rows)
        out_path = OUT_DIR / f"{sex_label}_{age_group}_pc_z_strengthPosNeg_abide1_fd2.csv"
        out_df.to_csv(out_path, index=False)

        print(
            f"  [{sex_label} {age_group}] Saved {len(out_df)} node-rows -> {out_path}\n"
            f"    skipped: badid={n_badid}, missing={n_missing}, bad_shape={n_shape}, nonfinite={n_nonfinite}"
        )


def main():
    process_sex("female", FEMALE_META)
    process_sex("male", MALE_META)
    print("\nDone.")


if __name__ == "__main__":
    main()