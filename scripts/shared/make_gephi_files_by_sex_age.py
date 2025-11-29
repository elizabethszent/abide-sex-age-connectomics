import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

CONN_DIR      = ROOT / "data/connectomes/cpac/nofilt_noglobal/cc200_z"
MODULE_FILE   = ROOT / "results/group_connectomes/CC200_modules.txt"
BASE_NODEFILE = ROOT / "results/vis/brainnet/CC200_base.node"

FEMALE_META = ROOT / "data/female/female_metadata_included.csv"
MALE_META   = ROOT / "data/male/male_metadata_included.csv"

OUT_DIR = ROOT / "results/vis/gephi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DENSITY = 0.10  #top 10% absolute edges
AGE_GROUPS = ["child", "teen", "young_adult"]


def add_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [0, 13, 18, 30, 45, 120]
    labels = ["child", "teen", "young_adult", "adult", "older"]
    df = df.copy()
    df["AGE_GROUP"] = pd.cut(df["AGE_AT_SCAN"], bins=bins,
                             labels=labels, right=False)
    return df


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

#modules + base node labels
mods = pd.read_csv(MODULE_FILE, sep=r"\s+").sort_values("ROI_index")
roi2mod = mods["Module"].to_numpy()
N_ROI = len(roi2mod)

base_nodes = pd.read_csv(
    BASE_NODEFILE,
    sep=r"\s+",
    header=None,
    names=["x", "y", "z", "size", "color", "label"],
)
assert len(base_nodes) == N_ROI, "CC200_base.node must have 200 rows"

#Main: build Gephi files per sex × age_group
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

        print(f"\n  [{sex_label} {age_group}] N total = {len(sub)}")
        print(sub["DX_GROUP"].value_counts().rename({1: "ASD", 2: "CTL"}))

        mats = []
        n_bad = 0
        for _, row in sub.iterrows():
            fid = row["FILE_ID"]
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

            mats.append(mat)

        if not mats:
            print(f"  [{sex_label} {age_group}] No usable 200x200 matrices after QC.")
            continue

        mats = np.stack(mats, axis=0)  # (N_sub, 200, 200)
        print(f"  [{sex_label} {age_group}] Usable subjects: {mats.shape[0]}, skipped: {n_bad}")

        #group-mean connectome
        group_mat = mats.mean(axis=0)
        group_mat = (group_mat + group_mat.T) / 2.0
        np.fill_diagonal(group_mat, 0.0)

        #threshold to top 10% absolute edges
        W = threshold_top_density(group_mat, density=DENSITY)

   
        iu, ju = np.triu_indices(N_ROI, k=1)
        weights = W[iu, ju]
        mask = weights != 0

        edges_df = pd.DataFrame({
            "Source": iu[mask] + 1, #ROI indices start at 1
            "Target": ju[mask] + 1,
            "Weight": weights[mask],})

        edges_path = OUT_DIR / f"{sex_label}_{age_group}_edges_gephi.csv"
        edges_df.to_csv(edges_path, index=False)
        print(f"  [{sex_label} {age_group}] Saved edges -> {edges_path}")


        #degree = node strength in this group-level W
        degree = W.sum(axis=1)

        nodes_df = pd.DataFrame({
            "Id": np.arange(1, N_ROI + 1),
            "Module": roi2mod,
            "Degree": degree,
            "Label": base_nodes["label"],
        })

        nodes_path = OUT_DIR / f"{sex_label}_{age_group}_nodes_gephi.csv"
        nodes_df.to_csv(nodes_path, index=False)
        print(f"  [{sex_label} {age_group}] Saved nodes -> {nodes_path}")


if __name__ == "__main__":
    process_sex("female", FEMALE_META)
    process_sex("male",   MALE_META)
    print("\nDone.")
