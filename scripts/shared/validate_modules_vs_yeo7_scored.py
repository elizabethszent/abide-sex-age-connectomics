import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import nibabel as nib
from nilearn.datasets import fetch_atlas_yeo_2011
from nilearn.image import resample_to_img


ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# Louvain output (ROI_index  Module)
#MODULE_FILE = ROOT / r"results\group_connectomes\CC200_modules_signed_asym1000.txt"
MODULE_FILE = ROOT / r"results\group_connectomes\CC200_modules_ALLSUBJ_signed_asym1000.txt"


# CC200 atlas where voxels are labeled 1..200 (your file)
CC200_ATLAS = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\abide\preprocessing\resources\abide_rois\CC200.nii.gz")

OUT_DIR = ROOT / r"results\qc\module_validation_scored"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Yeo7 labels in the atlas file (typical order)
YEO7_LABELS = {
    1: "Visual",
    2: "Somatomotor",
    3: "DorsalAttention",
    4: "VentralAttention",
    5: "Limbic",
    6: "Frontoparietal",
    7: "DefaultMode",
}


def load_modules_txt(path: Path) -> np.ndarray:
    # expects columns: ROI_index  Module
    mods = pd.read_csv(path, sep=r"\s+")
    mods = mods.sort_values("ROI_index")
    modules = mods["Module"].to_numpy().astype(int)
    if modules.shape[0] != 200:
        raise ValueError(f"Expected 200 module labels, got {modules.shape[0]}")
    return modules


def cc200_roi_voxel_ids(cc200_img: nib.Nifti1Image) -> np.ndarray:
    data = np.asanyarray(cc200_img.dataobj)
    roi_ids = np.unique(data)
    roi_ids = roi_ids[roi_ids != 0]
    roi_ids = roi_ids.astype(int)
    return np.sort(roi_ids)


def yeo7_in_cc200_space(cc200_img: nib.Nifti1Image) -> nib.Nifti1Image:
    yeo = fetch_atlas_yeo_2011()
    # Pick a 7-network volumetric atlas file from the dataset bundle
    # (this is the one you successfully used already)
    # It may vary by nilearn version, so we choose the one you printed:
    # Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz
    # If your nilearn changes in the future, update this selection.
    yeo7_path = None
    for k, v in yeo.items():
        if isinstance(v, str) and "7Networks" in v and v.endswith(".nii.gz"):
            yeo7_path = v
            break
    if yeo7_path is None:
        # fallback: try known key names
        for key in ["thick_7", "liberal_7", "anat_7"]:
            if hasattr(yeo, key):
                yeo7_path = getattr(yeo, key)
                break
    if yeo7_path is None:
        raise RuntimeError("Could not locate a 7-network Yeo atlas path in nilearn fetch_atlas_yeo_2011() output.")

    yeo_img = nib.load(yeo7_path)
    # resample to CC200 voxel grid using nearest-neighbor (labels)
    yeo_rs = resample_to_img(yeo_img, cc200_img, interpolation="nearest")
    return yeo_rs


def build_roi_to_yeo7(cc200_img: nib.Nifti1Image, yeo_rs: nib.Nifti1Image) -> pd.DataFrame:
    cc = np.asanyarray(cc200_img.dataobj).astype(int)
    yy = np.asanyarray(yeo_rs.dataobj).astype(int)

    roi_ids = cc200_roi_voxel_ids(cc200_img)
    if len(roi_ids) < 190:
        raise ValueError(f"CC200 atlas has only {len(roi_ids)} nonzero ROI labels; expected ~200.")

    rows = []
    for roi in roi_ids:
        mask = (cc == roi)
        vals = yy[mask]
        vals = vals[vals != 0]  # ignore background
        if vals.size == 0:
            dom = 0
            dom_name = "Background"
            dom_frac = 0.0
        else:
            dom = int(pd.Series(vals).value_counts().idxmax())
            dom_name = YEO7_LABELS.get(dom, f"Yeo{dom}")
            dom_frac = float((vals == dom).mean())
        rows.append((roi, dom, dom_name, dom_frac))

    df = pd.DataFrame(rows, columns=["ROI_index", "yeo7_id", "yeo7_name", "yeo7_dom_frac_within_roi"])
    return df


def contingency_and_scores(modules: np.ndarray, roi_to_yeo: pd.DataFrame):
    # modules index: ROI_index 1..200 maps to modules[0..199]
    df = roi_to_yeo.copy()
    df["module"] = df["ROI_index"].apply(lambda r: int(modules[r - 1]))
    # if yeo7_id=0 => Background, keep it as "Background"
    df["yeo7_name"] = df["yeo7_name"].astype(str)

    counts = pd.crosstab(df["module"], df["yeo7_name"])
    rowpct = counts.div(counts.sum(axis=1), axis=0)

    # Purity: sum over modules of (max count in that module) / total ROIs
    max_per_module = counts.max(axis=1)
    purity = float(max_per_module.sum() / counts.values.sum())

    dom_frac_per_module = (counts.max(axis=1) / counts.sum(axis=1)).astype(float)
    mean_dom = float(dom_frac_per_module.mean())
    min_dom = float(dom_frac_per_module.min())

    dom_label_per_module = counts.idxmax(axis=1)
    dom_count_per_module = counts.max(axis=1)

    return df, counts, rowpct, purity, mean_dom, min_dom, dom_label_per_module, dom_count_per_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Suffix tag for outputs (e.g., g1p18, g1p19, g1p20)")
    args = ap.parse_args()
    tag = args.tag

    if not MODULE_FILE.exists():
        raise FileNotFoundError(f"Missing module file: {MODULE_FILE}")
    if not CC200_ATLAS.exists():
        raise FileNotFoundError(f"Missing CC200 atlas: {CC200_ATLAS}")

    modules = load_modules_txt(MODULE_FILE)
    K = len(set(modules.tolist()))
    print(f"Loaded Louvain modules: N={len(modules)}, K={K}")

    cc200_img = nib.load(str(CC200_ATLAS))
    roi_ids = cc200_roi_voxel_ids(cc200_img)
    print(f"CC200 atlas nonzero ROI labels found: {len(roi_ids)}")

    yeo_rs = yeo7_in_cc200_space(cc200_img)

    roi_to_yeo = build_roi_to_yeo7(cc200_img, yeo_rs)

    df_map, counts, rowpct, purity, mean_dom, min_dom, dom_label, dom_count = contingency_and_scores(modules, roi_to_yeo)

    # Save outputs with tag so you can compare
    out_map    = OUT_DIR / f"cc200_roi_to_yeo7_and_module_{tag}.csv"
    out_counts = OUT_DIR / f"louvain_module_x_yeo7_counts_{tag}.csv"
    out_rowpct = OUT_DIR / f"louvain_module_x_yeo7_rowpct_{tag}.csv"

    df_map.to_csv(out_map, index=False)
    counts.to_csv(out_counts)
    rowpct.to_csv(out_rowpct)

    print(f"Saved per-ROI mapping -> {out_map}")
    print(f"Saved contingency table (counts) -> {out_counts}")
    print(f"Saved contingency table (row %) -> {out_rowpct}")

    print("\n=== Cleanliness scores ===")
    print(f"Purity (overall): {purity:.3f}")
    print(f"Mean dominant fraction (per module): {mean_dom:.3f}")
    print(f"Min dominant fraction (per module): {min_dom:.3f}")

    print("\n=== Module -> dominant Yeo7 network ===")
    for m in dom_label.index:
        print(f"Module {m}: {dom_label[m]} ({int(dom_count[m])}/{int(counts.loc[m].sum())} ROIs)")

    print("\nDone.")


if __name__ == "__main__":
    main()
