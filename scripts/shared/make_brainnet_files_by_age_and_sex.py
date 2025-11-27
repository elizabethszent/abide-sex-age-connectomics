import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")

CONN_DIR = ROOT / "data/connectomes/cpac/nofilt_noglobal/cc200_z"
MODULE_FILE = ROOT / "results/group_connectomes/CC200_modules.txt"
NODE_TEMPLATE = ROOT / "results/group_connectomes/CC200_base.node"

FEMALE_META = ROOT / "data/female/female_metadata_included.csv"
MALE_META   = ROOT / "data/male/male_metadata_included.csv"

PCZ_STATS_DIR = ROOT / "results/hubs"
OUT_DIR = ROOT / "results/brainnet"
OUT_DIR.mkdir(exist_ok=True, parents=True)

SEXES = ["female", "male"]
AGE_GROUPS = ["child", "teen", "young_adult"]


# === AGE GROUP HELPER ===
def add_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add AGE_GROUP from AGE_AT_SCAN using same bins as before.
    Also strips whitespace from column names.
    """
    df = df.copy()
    # strip stray spaces/commas in column names
    df.columns = df.columns.str.strip()

    # if AGE_GROUP already exists, just return
    if "AGE_GROUP" in df.columns:
        return df

    if "AGE_AT_SCAN" not in df.columns:
        raise KeyError(
            f"AGE_AT_SCAN not found in columns. Got: {list(df.columns)}"
        )

    bins = [0, 13, 18, 30, 45, 120]
    labels = ["child", "teen", "young_adult", "adult", "older"]
    df["AGE_GROUP"] = pd.cut(df["AGE_AT_SCAN"], bins=bins,
                             labels=labels, right=False)
    return df


# === LOAD MODULE ASSIGNMENTS ===
mods = pd.read_csv(MODULE_FILE, sep=r"\s+")
mods = mods.sort_values("ROI_index")

roi2mod = mods["Module"].to_numpy()
N_ROI = len(roi2mod)
N_MOD = int(roi2mod.max())
print(f"Loaded module file with {N_ROI} ROIs, {N_MOD} modules")


def mean_connectome(file_ids):
    mats = []
    for fid in file_ids:
        fp = CONN_DIR / f"{fid}.npy"
        if not fp.exists():
            print(f"  [WARN] missing connectome: {fp}")
            continue
        mat = np.load(fp)
        if mat.shape != (N_ROI, N_ROI):
            print(f"  [WARN] skipping {fid}: unexpected shape {mat.shape}")
            continue
        mats.append(mat)
    if not mats:
        return None, 0
    mats = np.stack(mats, axis=0)
    return mats.mean(axis=0), mats.shape[0]


def make_edge_file(sex: str, age: str, meta: pd.DataFrame):
    sub = meta[meta["AGE_GROUP"] == age].copy()
    if sub.empty:
        print(f"[{sex} {age}] No subjects, skipping edges.")
        return

    asd_ids = sub.loc[sub["DX_GROUP"] == 1, "FILE_ID"].tolist()
    ctl_ids = sub.loc[sub["DX_GROUP"] == 2, "FILE_ID"].tolist()

    print(f"[{sex} {age}] edge: {len(asd_ids)} ASD, {len(ctl_ids)} CTL")

    if len(asd_ids) < 5 or len(ctl_ids) < 5:
        print(f"  [WARN] too few subjects for reliable edge map, skipping.")
        return

    mean_asd, _ = mean_connectome(asd_ids)
    mean_ctl, _ = mean_connectome(ctl_ids)

    if mean_asd is None or mean_ctl is None:
        print("  [WARN] could not form both group means, skipping.")
        return

    diff = mean_ctl - mean_asd  # CTL - ASD

    out_edge = OUT_DIR / f"{sex}_{age}_CTLminusASD.edge"
    np.savetxt(out_edge, diff, fmt="%.6f")
    print(f"  Saved BrainNet edge file -> {out_edge}")


def make_node_files(sex: str, age: str):
    stats_fp = PCZ_STATS_DIR / f"{sex}_{age}_pc_z_module_stats.csv"
    if not stats_fp.exists():
        print(f"[{sex} {age}] No PC/Z stats file ({stats_fp}), skipping node files.")
        return

    stats = pd.read_csv(stats_fp)
    if not {"module", "d_pc", "d_z"}.issubset(stats.columns):
        print(f"  [WARN] {stats_fp} missing expected columns, skipping.")
        return

    d_pc_map = dict(zip(stats["module"], stats["d_pc"]))
    d_z_map  = dict(zip(stats["module"], stats["d_z"]))

    base = pd.read_csv(NODE_TEMPLATE, header=None, delim_whitespace=True)
    if base.shape[0] != N_ROI:
        print(f"  [WARN] NODE_TEMPLATE rows ({base.shape[0]}) "
              f"!= N_ROI ({N_ROI}), continuing anyway.")

    while base.shape[1] < 6:
        base[base.shape[1]] = 0.0

    base.columns = ["x", "y", "z", "size", "color", "label"]
    modules = roi2mod

    # --- PC ---
    pc_nodes = base.copy()
    pc_nodes["size"] = [float(d_pc_map.get(int(m), 0.0)) for m in modules]
    pc_nodes["color"] = [int(m) for m in modules]
    out_pc = OUT_DIR / f"{sex}_{age}_PC_diff.node"
    pc_nodes.to_csv(out_pc, header=False, index=False, sep="\t")
    print(f"  Saved BrainNet node file (PC) -> {out_pc}")

    # --- Z ---
    z_nodes = base.copy()
    z_nodes["size"] = [float(d_z_map.get(int(m), 0.0)) for m in modules]
    z_nodes["color"] = [int(m) for m in modules]
    out_z = OUT_DIR / f"{sex}_{age}_Z_diff.node"
    z_nodes.to_csv(out_z, header=False, index=False, sep="\t")
    print(f"  Saved BrainNet node file (Z)  -> {out_z}")


def process_sex(sex: str, meta_path: Path):
    print(f"\n {sex.upper()}")
    meta = pd.read_csv(meta_path)
    meta.columns = meta.columns.str.strip()
    meta["FILE_ID"] = meta["FILE_ID"].astype(str).str.strip()
    meta = add_age_group_column(meta)


    for age in AGE_GROUPS:
        sub = meta[meta["AGE_GROUP"] == age]
        if sub.empty:
            print(f"[{sex} {age}] No subjects, skipping.")
            continue

        print(f"\n[{sex} {age}] N = {len(sub)} "
              f"(ASD={sum(sub['DX_GROUP']==1)}, CTL={sum(sub['DX_GROUP']==2)})")

        make_edge_file(sex, age, meta)
        make_node_files(sex, age)


if __name__ == "__main__":
    process_sex("female", FEMALE_META)
    process_sex("male", MALE_META)
    print("\nDone generating BrainNet files.")
