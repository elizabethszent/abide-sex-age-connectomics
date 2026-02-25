import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn import datasets, image

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# CC200 atlas path (edit if yours differs)
CC200_ATLAS = ROOT / r"atlases\cc200\cc200_roi_atlas.nii.gz"

# Where your module solutions are (e.g., from compare_louvain_methods.py)
MODULE_DIR = ROOT / r"results\louvain_compare"
MODULE_GLOB = "*_best_modules.txt"   # or "*.txt" if you want all

OUT_DIR = ROOT / r"results\qc\module_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 200

def load_modules_txt(module_txt: Path) -> np.ndarray:
    df = pd.read_csv(module_txt, sep=r"\s+")
    if "ROI_index" not in df.columns or "Module" not in df.columns:
        raise ValueError(f"{module_txt} must contain columns ROI_index and Module")
    df = df.sort_values("ROI_index")
    mods = df["Module"].to_numpy().astype(int)
    if mods.ndim != 1:
        raise ValueError(f"Expected 1D modules, got {mods.shape}")
    return mods

def pick_yeo7_path(yeo_bunch) -> str:
    candidates = ["thin_7", "thick_7", "yeo_7", "maps_7", "yeo7", "thin7", "thick7"]
    for k in candidates:
        if hasattr(yeo_bunch, k):
            p = getattr(yeo_bunch, k)
            if isinstance(p, str) and (p.endswith(".nii") or p.endswith(".nii.gz")):
                return p
    # fallback: search any string fields
    for key in getattr(yeo_bunch, "keys", lambda: [])():
        v = yeo_bunch[key]
        if isinstance(v, str) and (v.endswith(".nii") or v.endswith(".nii.gz")):
            name = Path(v).name.lower()
            if ("7" in name) and ("17" not in name):
                return v
    raise RuntimeError("Could not find a Yeo7 NIfTI path from nilearn fetch_atlas_yeo_2011().")

def yeo7_to_cc200_space(cc200_img: nib.Nifti1Image):
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

def roi_majority_vote(cc200_data: np.ndarray, yeo_data: np.ndarray, roi_label: int) -> int:
    mask = (cc200_data == roi_label)
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

    module_files = sorted(MODULE_DIR.glob(MODULE_GLOB))
    if not module_files:
        raise FileNotFoundError(f"No module files found in {MODULE_DIR} matching {MODULE_GLOB}")

    cc200_img = nib.load(str(CC200_ATLAS))
    cc200_data = cc200_img.get_fdata().astype(int)
    roi_labels = np.unique(cc200_data)
    roi_labels = roi_labels[roi_labels != 0]
    roi_labels = np.sort(roi_labels)

    print(f"Atlas ROI labels found (nonzero): {len(roi_labels)}")

    yeo_rs, label_map = yeo7_to_cc200_space(cc200_img)
    yeo_data = yeo_rs.get_fdata().astype(int)

    for mf in module_files:
        label = mf.stem.replace("_best_modules", "")
        print(f"\n=== Validating {label} ===")
        modules = load_modules_txt(mf)

        if len(modules) != len(roi_labels):
            print(f"[WARN] modules length ({len(modules)}) != atlas ROI count ({len(roi_labels)})")
        if len(modules) != EXPECTED_N:
            print(f"[WARN] modules length is {len(modules)} not {EXPECTED_N}")

        N = min(len(modules), len(roi_labels))

        roi_to_yeo = np.zeros(N, dtype=int)
        for i in range(N):
            roi_label_i = int(roi_labels[i])  # robust even if labels aren’t 1..N
            roi_to_yeo[i] = roi_majority_vote(cc200_data, yeo_data, roi_label_i)

        per_roi = pd.DataFrame({
            "ROI_index": np.arange(1, N + 1),
            "ROI_label_in_atlas": roi_labels[:N].astype(int),
            "Louvain_module": modules[:N].astype(int),
            "Yeo7_label": roi_to_yeo.astype(int),
            "Yeo7_name": [label_map.get(int(x), f"Label_{int(x)}") for x in roi_to_yeo],
        })

        out_subdir = OUT_DIR / label
        out_subdir.mkdir(parents=True, exist_ok=True)

        out_per_roi = out_subdir / "cc200_roi_to_yeo7_and_module.csv"
        per_roi.to_csv(out_per_roi, index=False)

        ctab = pd.crosstab(per_roi["Louvain_module"], per_roi["Yeo7_name"])
        (out_subdir / "louvain_module_x_yeo7_counts.csv").write_text(ctab.to_csv())

        ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100.0
        (out_subdir / "louvain_module_x_yeo7_rowpct.csv").write_text(ctab_pct.to_csv())

        # quick console summary
        print("Dominant Yeo7 per module:")
        for m in sorted(per_roi["Louvain_module"].unique()):
            sub = per_roi[per_roi["Louvain_module"] == m]
            vc = sub["Yeo7_name"].value_counts()
            top = vc.index[0]
            print(f"  Module {m}: {top} ({vc.iloc[0]}/{len(sub)} ROIs)")

        print(f"Saved -> {out_subdir}")

if __name__ == "__main__":
    main()