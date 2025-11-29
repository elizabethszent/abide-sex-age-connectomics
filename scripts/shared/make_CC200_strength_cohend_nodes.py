import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

CONN_DIR = ROOT / "data/connectomes/cpac/nofilt_noglobal/cc200_z"
MODULE_FILE = ROOT / "results/group_connectomes/CC200_modules.txt"
BASE_NODE_FILE = ROOT / "results/vis/brainnet/CC200_base.node"

FEMALE_META = ROOT / "data/female/female_metadata_included.csv"
MALE_META   = ROOT / "data/male/male_metadata_included.csv"

OUT_DIR = ROOT / "results/vis/brainnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DENSITY = 0.10  #top 10% absolute edges
AGE_GROUPS = ["child", "teen", "young_adult"]


def threshold_top_density(mat: np.ndarray, density: float = 0.10) -> np.ndarray:

    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix not square: {mat.shape}")
    n = mat.shape[0]

    mat = mat.copy()
    np.fill_diagonal(mat, 0.0)

    iu, ju = np.triu_indices(n, k=1)
    vals = np.abs(mat[iu, ju])

    m = len(vals)
    k = int(np.floor(density * m))
    if k < 1:
        return np.zeros_like(mat)

    thresh = np.partition(vals, -k)[-k]
    mask = np.abs(mat) >= thresh

    out = np.where(mask, mat, 0.0)
    out = np.triu(out, 1)
    out = out + out.T
    np.fill_diagonal(out, 0.0)
    return out


def cohen_d(asd_vals, ctl_vals):

    asd = np.asarray(asd_vals, dtype=float)
    ctl = np.asarray(ctl_vals, dtype=float)
    asd = asd[~np.isnan(asd)]
    ctl = ctl[~np.isnan(ctl)]

    n1, n2 = len(asd), len(ctl)
    if n1 < 2 or n2 < 2:
        return np.nan

    m1, m2 = asd.mean(), ctl.mean()
    s1, s2 = asd.std(ddof=1), ctl.std(ddof=1)

    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0 or np.isnan(sp):
        return np.nan

    return (m2 - m1) / sp  # CTL - ASD


def add_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [0, 13, 18, 30, 45, 120]
    labels = ["child", "teen", "young_adult", "adult", "older"]
    df = df.copy()
    df["AGE_GROUP"] = pd.cut(df["AGE_AT_SCAN"], bins=bins,
                             labels=labels, right=False)
    return df


mods = pd.read_csv(MODULE_FILE, sep=r"\s+").sort_values("ROI_index")
roi2mod = mods["Module"].to_numpy()  # 1..7
N_ROI = len(roi2mod)

base_nodes = pd.read_csv(
    BASE_NODE_FILE,
    sep=r"\s+",
    header=None,
    names=["x", "y", "z", "size", "color", "label"],
)
assert len(base_nodes) == N_ROI, "Base node file must have 200 rows"


def process_sex(sex_label: str, meta_path: Path):
    print(f"\n=== {sex_label.upper()} ===")
    meta = pd.read_csv(meta_path)
    meta.columns = meta.columns.str.strip()
    meta = add_age_group_column(meta)

    required = {"FILE_ID", "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd", "AGE_GROUP"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"{meta_path} is missing columns: {missing}")

    for age_group in AGE_GROUPS:
        sub = meta[meta["AGE_GROUP"] == age_group].copy()
        if sub.empty:
            print(f"  [{sex_label} {age_group}] No subjects, skipping.")
            continue

        print(f"\n  [{sex_label} {age_group}] N = {len(sub)}")
        print(sub["DX_GROUP"].value_counts().rename({1: "ASD", 2: "CTL"}))

        #collect node strengths per subject
        all_rows = []
        n_bad = 0

        for _, row in sub.iterrows():
            fid = row["FILE_ID"]
            dx  = row["DX_GROUP"]
            conn_fp = CONN_DIR / f"{fid}.npy"

            if not conn_fp.exists():
                print(f"    [WARN] missing connectome: {conn_fp}")
                n_bad += 1
                continue

            mat = np.load(conn_fp)
            if mat.shape != (N_ROI, N_ROI):
                print(f"    [WARN] skipping {fid}: shape {mat.shape}")
                n_bad += 1
                continue

            W = threshold_top_density(mat, density=DENSITY)
            strength = W.sum(axis=1) #node strength

            for node_idx in range(N_ROI):
                all_rows.append({
                    "FILE_ID": fid,
                    "DX_GROUP": dx,
                    "node": node_idx + 1,
                    "strength": float(strength[node_idx]),
                })

        if not all_rows:
            print(f"  [{sex_label} {age_group}] No usable subjects after QC.")
            continue

        df_strength = pd.DataFrame(all_rows)

        #compute Cohen's d per node (CTL - ASD)
        cohend = []
        for node_id in range(1, N_ROI + 1):
            sub_node = df_strength[df_strength["node"] == node_id]
            asd_vals = sub_node[sub_node["DX_GROUP"] == 1]["strength"]
            ctl_vals = sub_node[sub_node["DX_GROUP"] == 2]["strength"]
            cohend.append(cohen_d(asd_vals, ctl_vals))

        cohend = np.array(cohend)

        #build node table: x y z ModuleID CohenD Label
        node_df = base_nodes.copy()
        node_df["ModuleID"] = roi2mod
        node_df["CohenD"] = cohend

        out = node_df[["x", "y", "z", "ModuleID", "CohenD", "label"]]

        out_path = OUT_DIR / f"{sex_label}_{age_group}_strength_cohend.node"
        out.to_csv(out_path, sep="\t", header=False, index=False)

        print(
            f"  [{sex_label} {age_group}] "
            f"Saved node file -> {out_path} (skipped {n_bad} subjects)"
        )


if __name__ == "__main__":
    process_sex("female", FEMALE_META)
    process_sex("male", MALE_META)
    print("\nDone.")

