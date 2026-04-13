from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# config
PHENO_CSV = Path(
    "/work/ioannou_lab/elizabeth.szentmiklo/abide-sex-age-connectomics/data/metadata/ABIDE12_phenotypes_combined_fd_0p3.csv"
)

CONNECTOME_ROOT = Path(
    "/work/ioannou_lab/elizabeth.szentmiklo/abide-sex-age-connectomics/results/connectomes/ABIDE12/ABIDE12/fd_0p3/matrices"
)

MODULE_MAP_TXT = Path(
    "/work/ioannou_lab/elizabeth.szentmiklo/abide-sex-age-connectomics/results/group_connectomes/ABIDE12_CC200/ABIDE_modules_asym_min10_fd-0.3.txt"
)

OUT_DIR = Path("module_mean_difference_matrices_fd0p3")

# ABIDE conventions
DX_ASD = 1
DX_CTL = 2

SEX_MALE = 1
SEX_FEMALE = 2

AGE_GROUP_MAP = {
    "child_0_9": "Child",
    "preteen_10_12": "Preteen",
    "teen_13_17": "Teen",
    "adult_18_plus": "Adult",
}

AGE_ORDER = ["Child", "Preteen", "Teen", "Adult"]

EXCLUDED_SUBJECTS: set[str] = set()

FIGSIZE = (11.8, 4.9)
DPI = 300
CMAP = "coolwarm"

MIN_PER_GROUP = 2

# helpers
def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_sub_id(value) -> str:
    s = str(value).strip()
    digits = re.sub(r"\D", "", s)
    if digits == "":
        return s
    return digits.zfill(7)


def extract_sub_id_from_path(path: Path) -> Optional[str]:
    """
    Find a 5-8 digit id anywhere in the full path and normalize to 7 digits.
    """
    full = str(path)
    matches = re.findall(r"(?<!\d)(\d{5,8})(?!\d)", full)
    if not matches:
        return None
    return matches[0].zfill(7)


def sex_to_label(sex_value) -> Optional[str]:
    if pd.isna(sex_value):
        return None
    try:
        sex_value = int(sex_value)
    except Exception:
        return None

    if sex_value == SEX_FEMALE:
        return "Female"
    if sex_value == SEX_MALE:
        return "Male"
    return None


def dx_to_label(dx_value) -> Optional[str]:
    if pd.isna(dx_value):
        return None
    try:
        dx_value = int(dx_value)
    except Exception:
        return None

    if dx_value == DX_CTL:
        return "CTL"
    if dx_value == DX_ASD:
        return "ASD"
    return None


def age_group_to_label(age_group_value) -> Optional[str]:
    if pd.isna(age_group_value):
        return None
    return AGE_GROUP_MAP.get(str(age_group_value).strip(), None)


def find_connectome_files(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Connectome root does not exist: {root}")

    found: dict[str, Path] = {}
    all_npy = sorted(root.rglob("*.npy"))

    print(f"[info] found {len(all_npy)} .npy files under {root}")

    no_id = 0
    dupes = 0

    for path in all_npy:
        sub_id = extract_sub_id_from_path(path)
        if sub_id is None:
            no_id += 1
            continue
        if sub_id in found:
            dupes += 1
            continue
        found[sub_id] = path

    print(f"[info] connectome files with parsed subject ids: {len(found)}")
    print(f"[info] .npy files without parsed subject id: {no_id}")
    print(f"[info] duplicate subject-id .npy files skipped: {dupes}")

    return found


def load_module_map(txt_path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Loads 1-based ROI_index -> Module mapping.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Module map file does not exist: {txt_path}")

    df = pd.read_csv(txt_path, sep=r"\s+|\t+", engine="python")

    cols_lower = {c.lower(): c for c in df.columns}
    roi_col = cols_lower.get("roi_index")
    mod_col = cols_lower.get("module")

    if roi_col is None or mod_col is None:
        raise ValueError("Module map must contain columns 'ROI_index' and 'Module'.")

    df = df[[roi_col, mod_col]].copy()
    df[roi_col] = df[roi_col].astype(int)
    df[mod_col] = df[mod_col].astype(int)

    df["roi_zero"] = df[roi_col] - 1
    df = df.sort_values("roi_zero").reset_index(drop=True)

    if df["roi_zero"].duplicated().any():
        raise ValueError("Duplicate ROI indices found in module map.")

    expected = list(range(200))
    actual = df["roi_zero"].tolist()
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"Module map must cover ROI 1..200 exactly. Missing={missing[:10]}, extra={extra[:10]}"
        )

    unique_modules = sorted(df[mod_col].unique().tolist())
    if unique_modules != [1, 2, 3, 4, 5, 6, 7, 8]:
        raise ValueError(f"Expected module ids 1..8, got {unique_modules}")

    module_per_roi = df[mod_col].to_numpy(dtype=int)

    module_labels = [
        "M1 Somatomotor",
        "M2 Visual-A",
        "M3 Limbic",
        "M4 Frontoparietal",
        "M5 VentralAttention",
        "M6 Visual-B",
        "M7 DefaultMode",
        "M8 DorsalAttention",
    ]

    return module_per_roi, module_labels


def vector_to_symmetric(vec: np.ndarray) -> np.ndarray:
    """
    Convert upper-triangle vector to symmetric matrix.
    n(n-1)/2 = len(vec)
    """
    vec = np.asarray(vec).ravel()
    m = len(vec)

    n = (1 + math.sqrt(1 + 8 * m)) / 2
    n_int = int(round(n))
    if n_int * (n_int - 1) // 2 != m:
        raise ValueError(f"Vector length {m} is not a valid upper triangle size.")

    out = np.zeros((n_int, n_int), dtype=float)
    iu = np.triu_indices(n_int, k=1)
    out[iu] = vec
    out[(iu[1], iu[0])] = vec
    return out


def load_connectome(path: Path) -> np.ndarray:
    arr = np.load(path)

    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return arr.astype(float)

    if arr.ndim == 1:
        return vector_to_symmetric(arr).astype(float)

    if arr.ndim == 2 and 1 in arr.shape:
        return vector_to_symmetric(arr.ravel()).astype(float)

    raise ValueError(f"Unsupported connectome shape {arr.shape} in {path}")


def module_connectivity_matrix(
    conn: np.ndarray,
    module_per_roi: np.ndarray,
    n_modules: int,
) -> np.ndarray:
    if conn.ndim != 2 or conn.shape[0] != conn.shape[1]:
        raise ValueError(f"Connectome must be square 2D, got shape {conn.shape}")

    n_rois = conn.shape[0]
    if n_rois != len(module_per_roi):
        raise ValueError(
            f"Connectome size {n_rois} does not match module map size {len(module_per_roi)}"
        )

    out = np.full((n_modules, n_modules), np.nan, dtype=float)

    for i in range(1, n_modules + 1):
        idx_i = np.where(module_per_roi == i)[0]
        for j in range(1, n_modules + 1):
            idx_j = np.where(module_per_roi == j)[0]

            if i == j:
                block = conn[np.ix_(idx_i, idx_i)]
                iu = np.triu_indices_from(block, k=1)
                vals = block[iu]
            else:
                vals = conn[np.ix_(idx_i, idx_j)].ravel()

            vals = vals[np.isfinite(vals)]
            out[i - 1, j - 1] = np.nan if vals.size == 0 else float(np.mean(vals))

    out = (out + out.T) / 2.0
    return out


def mean_difference_matrix(ctl_stack: np.ndarray, asd_stack: np.ndarray) -> np.ndarray:
    """
    Raw mean connectivity difference matrix.

    Positive values = CTL > ASD
    Negative values = ASD > CTL

    If you want ASD - CTL instead, change the return line to:
        return mean_asd - mean_ctl
    """
    mean_ctl = np.nanmean(ctl_stack, axis=0)
    mean_asd = np.nanmean(asd_stack, axis=0)
    return mean_ctl - mean_asd

# data loading
@dataclass
class GroupData:
    age_label: str
    sex_label: str
    ctl_mats: list[np.ndarray]
    asd_mats: list[np.ndarray]


def build_subject_table() -> pd.DataFrame:
    if not PHENO_CSV.exists():
        raise FileNotFoundError(f"Phenotype CSV does not exist: {PHENO_CSV}")

    pheno = pd.read_csv(PHENO_CSV)

    # important:
    # use explicit uppercase columns to avoid the SEX vs sex collision.
    required = ["SUB_ID", "DX_GROUP", "SEX", "AGE_GROUP"]
    missing = [c for c in required if c not in pheno.columns]
    if missing:
        raise KeyError(f"Phenotype CSV is missing required columns: {missing}")

    pheno = pheno[["SUB_ID", "DX_GROUP", "SEX", "AGE_GROUP"]].copy()
    pheno.columns = ["sub_id_raw", "dx", "sex", "age_group_raw"]

    pheno["sub_id"] = pheno["sub_id_raw"].apply(normalize_sub_id)
    pheno["age_label"] = pheno["age_group_raw"].apply(age_group_to_label)
    pheno["sex_label"] = pheno["sex"].apply(sex_to_label)
    pheno["dx_label"] = pheno["dx"].apply(dx_to_label)

    print(f"[info] phenotype rows total: {len(pheno)}")
    print("[info] selected phenotype columns: SUB_ID, DX_GROUP, SEX, AGE_GROUP")

    pheno = pheno.dropna(subset=["age_label", "sex_label", "dx_label"]).copy()
    pheno = pheno[~pheno["sub_id"].isin(EXCLUDED_SUBJECTS)].copy()

    print(f"[info] phenotype rows after valid age/sex/dx filtering: {len(pheno)}")

    conn_files = find_connectome_files(CONNECTOME_ROOT)
    pheno["conn_path"] = pheno["sub_id"].map(conn_files)

    unmatched = pheno["conn_path"].isna().sum()
    print(f"[info] phenotype rows without matched connectome: {unmatched}")
    if unmatched > 0:
        sample = pheno.loc[pheno["conn_path"].isna(), "sub_id"].astype(str).head(10).tolist()
        print(f"[info] sample unmatched subject ids: {sample}")

    pheno = pheno.dropna(subset=["conn_path"]).copy()
    print(f"[info] matched phenotype rows used: {len(pheno)}")

    return pheno


def build_group_data() -> tuple[dict[tuple[str, str], GroupData], list[str], pd.DataFrame]:
    pheno = build_subject_table()
    module_per_roi, module_labels = load_module_map(MODULE_MAP_TXT)
    n_modules = len(module_labels)

    grouped: dict[tuple[str, str], GroupData] = {}
    kept_rows = []

    bad_shape = 0
    load_fail = 0

    for _, row in pheno.iterrows():
        sub_id = row["sub_id"]
        age_label = row["age_label"]
        sex_label = row["sex_label"]
        dx_label = row["dx_label"]
        conn_path = Path(row["conn_path"])

        try:
            conn = load_connectome(conn_path)
            mod_mat = module_connectivity_matrix(conn, module_per_roi, n_modules)
        except Exception as e:
            print(f"[warn] skipping {sub_id}: {e}")
            if "shape" in str(e).lower():
                bad_shape += 1
            else:
                load_fail += 1
            continue

        key = (age_label, sex_label)
        if key not in grouped:
            grouped[key] = GroupData(
                age_label=age_label,
                sex_label=sex_label,
                ctl_mats=[],
                asd_mats=[],
            )

        if dx_label == "CTL":
            grouped[key].ctl_mats.append(mod_mat)
        elif dx_label == "ASD":
            grouped[key].asd_mats.append(mod_mat)

        kept_rows.append(
            {
                "sub_id": sub_id,
                "age_label": age_label,
                "sex_label": sex_label,
                "dx_label": dx_label,
                "conn_path": str(conn_path),
                "conn_shape": f"{conn.shape[0]}x{conn.shape[1]}",
            }
        )

    print(f"[info] subjects skipped due to shape issues: {bad_shape}")
    print(f"[info] subjects skipped due to other load failures: {load_fail}")

    inventory_df = pd.DataFrame(kept_rows)
    return grouped, module_labels, inventory_df


# plotting
def plot_age_pair(
    age_label: str,
    female_diff: Optional[np.ndarray],
    male_diff: Optional[np.ndarray],
    female_n: tuple[int, int],
    male_n: tuple[int, int],
    module_labels: list[str],
    out_path: Path,
    global_abs_max: float,
):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("white")

    panels = [
        ("Female", female_diff, female_n, axes[0]),
        ("Male", male_diff, male_n, axes[1]),
    ]

    for sex_label, diff_mat, (n_ctl, n_asd), ax in panels:
        if diff_mat is None:
            ax.axis("off")
            ax.set_title(f"{sex_label} {age_label}\n(no usable data)", fontsize=11)
            continue

        im = ax.imshow(
            diff_mat,
            cmap=CMAP,
            vmin=-global_abs_max,
            vmax=global_abs_max,
            interpolation="nearest",
            aspect="equal",
        )

        short_labels = [lbl.split()[0] if " " in lbl else lbl for lbl in module_labels]

        ax.set_xticks(np.arange(len(module_labels)))
        ax.set_yticks(np.arange(len(module_labels)))
        ax.set_xticklabels(short_labels, rotation=0)
        ax.set_yticklabels(short_labels)

        ax.set_title(
            f"{sex_label} {age_label} – network connectivity\n"
            f"(ASD vs Control)\n"
            f"CTL={n_ctl}, ASD={n_asd}",
            fontsize=11,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean connectivity difference (CTL - ASD)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_all_age_groups(
    grouped: dict[tuple[str, str], GroupData],
    module_labels: list[str],
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    diff_mats: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    all_abs = []

    print("\n[info] group counts")
    print("-" * 60)

    for age_label in AGE_ORDER:
        for sex_label in ["Female", "Male"]:
            key = (age_label, sex_label)
            g = grouped.get(key)

            if g is None:
                counts[key] = (0, 0)
                print(f"{age_label:8s} | {sex_label:6s} | CTL=0 ASD=0")
                continue

            n_ctl = len(g.ctl_mats)
            n_asd = len(g.asd_mats)
            counts[key] = (n_ctl, n_asd)

            print(f"{age_label:8s} | {sex_label:6s} | CTL={n_ctl} ASD={n_asd}")

            if n_ctl < MIN_PER_GROUP or n_asd < MIN_PER_GROUP:
                continue

            ctl_stack = np.stack(g.ctl_mats, axis=0)
            asd_stack = np.stack(g.asd_mats, axis=0)

            diff = mean_difference_matrix(ctl_stack, asd_stack)
            diff_mats[key] = diff
            all_abs.append(np.nanmax(np.abs(diff)))

    if not all_abs:
        raise RuntimeError(
            "No valid age/sex groups had enough subjects to compute mean difference matrices.\n"
            "Check the printed diagnostics above and the CSVs written to the output folder."
        )

    global_abs_max = max(all_abs)
    if not np.isfinite(global_abs_max) or global_abs_max == 0:
        global_abs_max = 1.0

    for age_label in AGE_ORDER:
        female_diff = diff_mats.get((age_label, "Female"))
        male_diff = diff_mats.get((age_label, "Male"))
        female_n = counts.get((age_label, "Female"), (0, 0))
        male_n = counts.get((age_label, "Male"), (0, 0))

        if female_diff is None and male_diff is None:
            continue

        out_path = OUT_DIR / f"mean_connectivity_difference_{age_label.lower()}.png"
        plot_age_pair(
            age_label=age_label,
            female_diff=female_diff,
            male_diff=male_diff,
            female_n=female_n,
            male_n=male_n,
            module_labels=module_labels,
            out_path=out_path,
            global_abs_max=global_abs_max,
        )


# main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grouped, module_labels, inventory_df = build_group_data()

    inventory_path = OUT_DIR / "matched_subject_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)
    print(f"[info] wrote matched subject inventory: {inventory_path}")

    if not inventory_df.empty:
        group_counts = (
            inventory_df.groupby(["age_label", "sex_label", "dx_label"])
            .size()
            .reset_index(name="n")
            .sort_values(["age_label", "sex_label", "dx_label"])
        )
    else:
        group_counts = pd.DataFrame(columns=["age_label", "sex_label", "dx_label", "n"])

    counts_path = OUT_DIR / "group_counts.csv"
    group_counts.to_csv(counts_path, index=False)
    print(f"[info] wrote group counts: {counts_path}")

    plot_all_age_groups(grouped, module_labels)
    print(f"[info] saved outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()