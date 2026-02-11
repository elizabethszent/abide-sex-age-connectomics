import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib

# Optional but very useful for quick figures
try:
    from nilearn import plotting
except ImportError:
    plotting = None

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# Your module output (either txt or npy)
MODULE_TXT = ROOT / r"results\group_connectomes\CC200_modules_ALLSUBJ_signed_asym1000.txt"
MODULE_NPY = ROOT / r"results\group_connectomes\CC200_modules_ALLSUBJ_signed_asym1000.npy"

# Where to save outputs
OUT_DIR = ROOT / r"results\qc\module_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)



# If auto-find fails, uncomment and set manually:
# ATLAS_PATH = ROOT / r"data\parcellation\cc200_atlas.nii.gz"
ATLAS_PATH = Path(
    r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\abide\preprocessing\resources\abide_rois\CC200.nii.gz"
)

def load_modules() -> np.ndarray:
    if MODULE_NPY.exists():
        mods = np.load(MODULE_NPY).astype(int)
        # modules file is already length-200 (ROI order 0..199)
        if mods.shape != (200,):
            raise ValueError(f"Expected modules shape (200,), got {mods.shape}")
        return mods

    if MODULE_TXT.exists():
        # expects header: ROI_index Module
        df = pd.read_csv(MODULE_TXT, sep=r"\s+")
        df = df.sort_values("ROI_index")
        mods = df["Module"].to_numpy().astype(int)
        if mods.shape != (200,):
            raise ValueError(f"Expected 200 rows in {MODULE_TXT}, got {mods.shape}")
        return mods

    raise FileNotFoundError("Could not find module file (.npy or .txt).")

def atlas_roi_ids(atlas_data: np.ndarray) -> np.ndarray:
    vals = np.unique(atlas_data)
    vals = vals[vals != 0]  # 0 is background
    return vals.astype(int)

def build_module_map(atlas_img: nib.Nifti1Image, modules: np.ndarray) -> nib.Nifti1Image:
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_ids = atlas_roi_ids(atlas_data)

    # CC200 atlases are commonly labeled 1..200, but sometimes not.
    # We assume ROI i corresponds to label i (1-based) unless atlas labels say otherwise.
    if len(roi_ids) < 200:
        raise ValueError(f"Atlas has only {len(roi_ids)} nonzero labels; expected ~200.")
    if len(roi_ids) > 200:
        # Some atlases have extra junk labels; we can still map 1..200
        pass

    # Create new volume: each voxel gets its module id (1..K)
    out = np.zeros_like(atlas_data, dtype=np.int16)

    # Map ROI label -> module id
    # assume ROI label is 1..200 -> index 0..199
    for roi_label in range(1, 201):
        mod_id = int(modules[roi_label - 1])
        out[atlas_data == roi_label] = mod_id

    return nib.Nifti1Image(out, affine=atlas_img.affine, header=atlas_img.header)

def roi_centroids_mni(atlas_img: nib.Nifti1Image) -> dict:
    """Return centroid (x,y,z) in MNI for each ROI label 1..200."""
    atlas_data = atlas_img.get_fdata().astype(int)
    aff = atlas_img.affine
    centroids = {}

    for roi_label in range(1, 201):
        vox = np.argwhere(atlas_data == roi_label)
        if vox.size == 0:
            continue
        mean_vox = vox.mean(axis=0)
        # convert voxel coords to world (MNI) coords
        x, y, z, _ = aff @ np.array([mean_vox[0], mean_vox[1], mean_vox[2], 1.0])
        centroids[roi_label] = (float(x), float(y), float(z))
    return centroids

def write_brainnet_node(centroids: dict, modules: np.ndarray, out_path: Path):
    """
    BrainNet .node format (typical):
    x y z  color/size  size  label
    We’ll put module as the “color” field.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for roi_label in range(1, 201):
            if roi_label not in centroids:
                continue
            x, y, z = centroids[roi_label]
            mod = int(modules[roi_label - 1])
            size = 3.0
            label = f"ROI{roi_label}"
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {mod} {size:.2f} {label}\n")

def main():
    modules = load_modules()
    K = int(modules.max())
    print(f"Loaded modules for 200 ROIs. K={K}. Module sizes:")
    uniq, cnts = np.unique(modules, return_counts=True)
    for u, c in zip(uniq, cnts):
        print(f"  module {u}: {c}")

    atlas_path = ATLAS_PATH if ATLAS_PATH is not None else find_cc200_atlas(ROOT)
    print(f"Using atlas: {atlas_path}")

    atlas_img = nib.load(str(atlas_path))

    mod_img = build_module_map(atlas_img, modules)
    out_nii = OUT_DIR / f"CC200_modulemap_K{K}.nii.gz"
    nib.save(mod_img, str(out_nii))
    print(f"Saved module NIfTI -> {out_nii}")

    # Quick PNGs (if nilearn installed)
    if plotting is not None:
        out_png = OUT_DIR / f"gradmeanCC200_modulemap_K{K}_glass.png"
        display = plotting.plot_glass_brain(mod_img, title=f"CC200 Louvain modules (K={K})")
        display.savefig(str(out_png))
        display.close()
        print(f"Saved glass brain PNG -> {out_png}")
    else:
        print("nilearn not installed; skipping PNGs. (pip install nilearn)")

    # Optional: BrainNet node file (lets you view in BrainNet Viewer)
    cents = roi_centroids_mni(atlas_img)
    out_node = OUT_DIR / f"CC200_modules_K{K}.node"
    write_brainnet_node(cents, modules, out_node)
    print(f"Saved BrainNet node file -> {out_node}")

if __name__ == "__main__":
    main()
