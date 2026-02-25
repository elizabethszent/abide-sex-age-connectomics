import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib

from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# Where compare_louvain_methods.py saved best partitions:
MODULE_DIR = ROOT / r"results\louvain_compare"

# Where to save outputs:
OUT_DIR = ROOT / r"results\qc\module_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 200

# We will process these four labels (must match compare script output filenames):
WANTED_LABELS = [
    "OVERALL_sexbalanced_fd-0.2",
    "OVERALL_sexbalanced_fd-0.3",
    "OVERALL_ageSexMatched_fd-0.2",
    "OVERALL_ageSexMatched_fd-0.3",
]

def find_cc200_atlas(root: Path) -> Path:
    # Your real atlas path is here:
    candidates = [
        root / r"atlases\cc200\cc200_roi_atlas.nii.gz",
        root / r"abide\preprocessing\resources\abide_rois\CC200.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = sorted([p for p in root.rglob("*cc200*nii*") if p.is_file()])
    if hits:
        return hits[0]
    raise FileNotFoundError("Could not find a CC200 atlas NIfTI under the project folder.")

def load_modules_for_label(label: str) -> np.ndarray:
    """
    Prefer .npy if present, else read *_best_modules.txt.
    """
    npy = MODULE_DIR / f"{label}_best_modules.npy"
    txt = MODULE_DIR / f"{label}_best_modules.txt"

    if npy.exists():
        mods = np.load(npy).astype(int)
        if mods.ndim != 1:
            raise ValueError(f"{npy.name}: expected 1D array, got {mods.shape}")
        return mods

    if txt.exists():
        df = pd.read_csv(txt, sep=r"\s+")
        if "ROI_index" not in df.columns or "Module" not in df.columns:
            raise ValueError(f"{txt.name}: expected columns ROI_index and Module")
        df = df.sort_values("ROI_index")
        return df["Module"].to_numpy().astype(int)

    raise FileNotFoundError(f"Missing module files for {label}: {npy.name} or {txt.name}")

def atlas_roi_labels(atlas_data: np.ndarray) -> np.ndarray:
    vals = np.unique(atlas_data.astype(int))
    vals = vals[vals != 0]
    return np.sort(vals)

def build_module_map(atlas_img: nib.Nifti1Image, modules: np.ndarray) -> nib.Nifti1Image:
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_labels = atlas_roi_labels(atlas_data)

    if len(roi_labels) < len(modules):
        raise ValueError(f"Atlas has {len(roi_labels)} ROIs but modules has {len(modules)} entries.")

    out = np.zeros_like(atlas_data, dtype=np.int16)

    # Robust mapping: if labels are 1..N use direct, else map by sorted label order.
    is_1_to_n = np.array_equal(roi_labels[: len(modules)], np.arange(1, len(modules) + 1))
    if is_1_to_n:
        for roi_label in range(1, len(modules) + 1):
            out[atlas_data == roi_label] = int(modules[roi_label - 1])
    else:
        print("[WARN] Atlas ROI labels are not 1..N. Mapping by sorted atlas ROI labels.")
        for i in range(len(modules)):
            out[atlas_data == int(roi_labels[i])] = int(modules[i])

    return nib.Nifti1Image(out, atlas_img.affine, atlas_img.header)

def fixed_module_cmap(K: int) -> ListedColormap:
    """
    CATEGORICAL colors (not gradient).
    Index 0 is transparent background.
    Module 1 = BLUE, Module 2 = YELLOW, etc.
    """
    bg = (0, 0, 0, 0)

    palette = [
        (0.10, 0.30, 0.90, 1.0),  # 1 blue
        (1.00, 0.90, 0.10, 1.0),  # 2 yellow
        (0.20, 0.80, 0.20, 1.0),  # 3 green
        (0.90, 0.20, 0.20, 1.0),  # 4 red
        (0.65, 0.35, 0.90, 1.0),  # 5 purple
        (1.00, 0.55, 0.10, 1.0),  # 6 orange
        (0.10, 0.80, 0.85, 1.0),  # 7 cyan
        (0.95, 0.35, 0.75, 1.0),  # 8 pink
        (0.55, 0.55, 0.55, 1.0),  # 9 gray
        (0.60, 0.40, 0.20, 1.0),  # 10 brown
    ]

    if K > len(palette):
        base = plt.get_cmap("tab20")
        for i in range(K - len(palette)):
            palette.append(base(i % 20))

    return ListedColormap([bg] + palette[:K], name=f"fixed_modules_{K}")

def save_legend_png(K: int, cmap: ListedColormap, out_path: Path):
    fig_h = max(2.0, 0.35 * K)
    fig, ax = plt.subplots(figsize=(4.4, fig_h))
    ax.set_axis_off()
    y = K
    for m in range(1, K + 1):
        ax.add_patch(plt.Rectangle((0.1, y - 0.8), 0.35, 0.6, color=cmap(m)))
        ax.text(0.6, y - 0.5, f"Module {m}", va="center", fontsize=10)
        y -= 1
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, K + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)

def main():
    if not MODULE_DIR.exists():
        raise FileNotFoundError(f"Missing module dir: {MODULE_DIR}")

    atlas_path = find_cc200_atlas(ROOT)
    print(f"[INFO] Using atlas: {atlas_path}")
    atlas_img = nib.load(str(atlas_path))

    made_any = False

    for label in WANTED_LABELS:
        print(f"\n=== {label} ===")
        mods = load_modules_for_label(label)

        if len(mods) != EXPECTED_N:
            print(f"[WARN] modules length is {len(mods)} not {EXPECTED_N}")

        K = int(np.max(mods))
        print(f"[INFO] K={K}")

        mod_img = build_module_map(atlas_img, mods)

        out_sub = OUT_DIR / label
        out_sub.mkdir(parents=True, exist_ok=True)

        out_nii = out_sub / f"{label}_CC200_modulemap_K{K}.nii.gz"
        nib.save(mod_img, str(out_nii))

        cmap = fixed_module_cmap(K)
        vmin, vmax = 0.5, K + 0.5  # crisp integer binning

        # No nilearn colorbar (it looks gradient-y). We output a categorical legend PNG instead.
        out_glass = out_sub / f"{label}_K{K}_glass.png"
        disp = plotting.plot_glass_brain(
            mod_img,
            title=f"{label} (K={K})",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            plot_abs=False,
        )
        disp.savefig(str(out_glass), dpi=250)
        disp.close()

        out_slices = out_sub / f"{label}_K{K}_slices.png"
        disp2 = plotting.plot_roi(
            mod_img,
            title=f"{label} (K={K})",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            draw_cross=False,
            display_mode="ortho",
            cut_coords=(0, -20, 20),
            resampling_interpolation="nearest",
        )
        disp2.savefig(str(out_slices), dpi=250)
        disp2.close()

        out_legend = out_sub / f"{label}_K{K}_legend.png"
        save_legend_png(K, cmap, out_legend)

        print(f"Saved:")
        print(f"  {out_nii}")
        print(f"  {out_glass}")
        print(f"  {out_slices}")
        print(f"  {out_legend}")

        made_any = True

    if not made_any:
        print("[WARN] Nothing was generated. Check that your *_best_modules files exist in results/louvain_compare.")

if __name__ == "__main__":
    main()