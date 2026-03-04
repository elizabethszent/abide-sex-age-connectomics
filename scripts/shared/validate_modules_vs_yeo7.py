import numpy as np
import pandas as pd
from pathlib import Path

import nibabel as nib
from nilearn import datasets, image


ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

CC200_ATLAS = ROOT / r"atlases\cc200\cc200_roi_atlas.nii.gz"

# Use ONE of these:
MODULE_TXT = ROOT / r"results\group_connectomes\CC200_modules_ALLSUBJ_signed_asym1000.txt"
MODULE_NPY = None 

OUT_DIR = ROOT / r"results\qc\module_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)



def load_modules(module_txt: Path | None, module_npy: Path | None) -> np.ndarray:
    """Returns modules array length N_ROI with values 1..K."""
    if module_npy is not None:
        mods = np.load(module_npy)
        mods = np.asarray(mods).astype(int)
        if mods.ndim != 1:
            raise ValueError(f"Expected 1D modules array, got shape {mods.shape}")
        return mods

    if module_txt is None or not module_txt.exists():
        raise FileNotFoundError("Provide MODULE_TXT or MODULE_NPY")

    df = pd.read_csv(module_txt, sep=r"\s+")
    if "ROI_index" not in df.columns or "Module" not in df.columns:
        raise ValueError(f"{module_txt} must contain columns ROI_index and Module")

    df = df.sort_values("ROI_index")
    return df["Module"].to_numpy().astype(int)


def pick_yeo7_path(yeo_bunch) -> str:
    """
    Robustly find a 7-network volumetric NIfTI path from nilearn's Yeo bundle.
    Works across nilearn versions where keys differ (thin_7, thick_7, etc).
    """
    # 1) Try common attribute names
    candidates = [
        "thin_7", "thick_7",
        "yeo_7", "maps_7", "yeo7",
        "thin7", "thick7",
    ]
    for k in candidates:
        if hasattr(yeo_bunch, k):
            p = getattr(yeo_bunch, k)
            if isinstance(p, str) and (p.endswith(".nii") or p.endswith(".nii.gz")):
                return p

    # 2) Search any string fields inside the bunch for something "7" + nifti
    for key in getattr(yeo_bunch, "keys", lambda: [])():
        v = yeo_bunch[key]
        if isinstance(v, str) and (v.endswith(".nii") or v.endswith(".nii.gz")):
            name = Path(v).name.lower()
            if ("7" in name) and ("17" not in name):
                return v

    base = Path.home() / "nilearn_data" / "yeo_2011"
    if base.exists():
        nii = list(base.rglob("*.nii*"))
        # Prefer files with "7" and "networks" and not "17"
        scored = []
        for p in nii:
            n = p.name.lower()
            score = 0
            if "7" in n and "17" not in n:
                score += 2
            if "network" in n or "networks" in n:
                score += 2
            if "mni" in n:
                score += 1
            scored.append((score, str(p)))
        scored.sort(reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]

    raise RuntimeError(
        "Could not find a Yeo 7-network NIfTI in fetch_atlas_yeo_2011(). "
        "Print the bunch keys and we’ll target the right one."
    )


def yeo7_to_cc200_space(cc200_img: nib.Nifti1Image):
    """Fetch Yeo 7 networks, resample to CC200 atlas space (nearest)."""
    yeo = datasets.fetch_atlas_yeo_2011()
    yeo_7_path = pick_yeo7_path(yeo)
    print(f"Using Yeo7 atlas: {yeo_7_path}")

    yeo_img = nib.load(yeo_7_path)
    yeo_rs = image.resample_to_img(yeo_img, cc200_img, interpolation="nearest")

    label_map = {
        0: "Background",
        1: "Visual",
        2: "Somatomotor",
        3: "DorsalAttention",
        4: "VentralAttention",
        5: "Limbic",
        6: "Frontoparietal",
        7: "DefaultMode",
    }
    return yeo_rs, label_map


def roi_majority_vote(cc200_data: np.ndarray, yeo_data: np.ndarray, roi_id: int) -> int:
    mask = (cc200_data == roi_id)
    if not np.any(mask):
        return 0
    vals = yeo_data[mask].astype(int)
    if vals.size == 0:
        return 0
    nonzero = vals[vals != 0]
    use = nonzero if nonzero.size > 0 else vals
    labels, counts = np.unique(use, return_counts=True)
    return int(labels[np.argmax(counts)])


def main():
    if not CC200_ATLAS.exists():
        raise FileNotFoundError(f"Missing CC200 atlas: {CC200_ATLAS}")

    modules = load_modules(MODULE_TXT if MODULE_NPY is None else None, MODULE_NPY)
    K = int(modules.max())
    print(f"Loaded Louvain modules: N={len(modules)}, K={K}")

    cc200_img = nib.load(str(CC200_ATLAS))
    cc200_data = cc200_img.get_fdata().astype(int)

    roi_ids = np.unique(cc200_data)
    roi_ids = roi_ids[roi_ids != 0]
    print(f"CC200 atlas nonzero ROI labels found: {len(roi_ids)}")
    if len(roi_ids) != len(modules):
        print("[WARN] Atlas ROI count != modules length. This script assumes ROI labels are 1..N.")

    yeo_rs, label_map = yeo7_to_cc200_space(cc200_img)
    yeo_data = yeo_rs.get_fdata().astype(int)

    N = len(modules)
    roi_to_yeo = np.zeros(N, dtype=int)
    for roi in range(1, N + 1):
        roi_to_yeo[roi - 1] = roi_majority_vote(cc200_data, yeo_data, roi)

    per_roi = pd.DataFrame({
        "ROI_index": np.arange(1, N + 1),
        "Louvain_module": modules.astype(int),
        "Yeo7_label": roi_to_yeo.astype(int),
        "Yeo7_name": [label_map.get(int(x), f"Label_{int(x)}") for x in roi_to_yeo]
    })

    out_per_roi = OUT_DIR / "cc200_roi_to_yeo7_and_module.csv"
    per_roi.to_csv(out_per_roi, index=False)
    print(f"Saved per-ROI mapping -> {out_per_roi}")

    ctab = pd.crosstab(per_roi["Louvain_module"], per_roi["Yeo7_name"])
    out_ctab = OUT_DIR / "louvain_module_x_yeo7_counts.csv"
    ctab.to_csv(out_ctab)
    print(f"Saved contingency table (counts) -> {out_ctab}")

    ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100.0
    out_pct = OUT_DIR / "louvain_module_x_yeo7_rowpct.csv"
    ctab_pct.to_csv(out_pct)
    print(f"Saved contingency table (row %) -> {out_pct}")

    print("\n=== Louvain module -> dominant Yeo7 network ===")
    for m in sorted(per_roi["Louvain_module"].unique()):
        sub = per_roi[per_roi["Louvain_module"] == m]
        vc = sub["Yeo7_name"].value_counts()
        
        # Calculate percentage data
        top_network = vc.index[0]
        top_count = vc.iloc[0]
        total_rois = len(sub)
        percentage = (top_count / total_rois) * 100.0
        
        # Print with percentage formatted to 1 decimal place
        print(f"Module {m}: {top_network} ({top_count}/{total_rois} ROIs, {percentage:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()