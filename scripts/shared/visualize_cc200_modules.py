import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib

from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# Prefer newest Louvain output from compare script; fallback to old folder
MODULE_DIR_PRIMARY = ROOT / r"results\louvain_compare"
MODULE_DIR_FALLBACK = ROOT / r"results\group_connectomes"

OUT_DIR = ROOT / r"results\qc\module_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 200


def find_cc200_atlas(root: Path) -> Path:
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


def newest_matching(paths):
    paths = [p for p in paths if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def find_latest_module_file() -> tuple[Path | None, Path | None]:
    npys = sorted(MODULE_DIR_PRIMARY.glob("*_best_modules.npy"))
    txts = sorted(MODULE_DIR_PRIMARY.glob("*_best_modules.txt"))
    best = newest_matching(npys + txts)
    if best is not None:
        return (best, None) if best.suffix == ".npy" else (None, best)

    npys = sorted(MODULE_DIR_FALLBACK.glob("CC200_modules*.npy"))
    txts = sorted(MODULE_DIR_FALLBACK.glob("CC200_modules*.txt"))
    best = newest_matching(npys + txts)
    if best is not None:
        return (best, None) if best.suffix == ".npy" else (None, best)

    return None, None


def load_modules(module_txt: Path | None, module_npy: Path | None) -> np.ndarray:
    if module_npy is not None:
        mods = np.load(module_npy).astype(int)
        if mods.ndim != 1:
            raise ValueError(f"Expected 1D modules, got {mods.shape}")
        return mods

    if module_txt is not None:
        df = pd.read_csv(module_txt, sep=r"\s+")
        if "ROI_index" not in df.columns or "Module" not in df.columns:
            raise ValueError(f"{module_txt} must contain ROI_index and Module columns")
        df = df.sort_values("ROI_index")
        return df["Module"].to_numpy().astype(int)

    raise FileNotFoundError("No module file found (.npy or .txt).")


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

    # If labels are exactly 1..N, map directly; otherwise map by sorted ROI labels.
    is_1_to_n = np.array_equal(roi_labels[: len(modules)], np.arange(1, len(modules) + 1))
    if is_1_to_n:
        for roi_label in range(1, len(modules) + 1):
            out[atlas_data == roi_label] = int(modules[roi_label - 1])
    else:
        print("[WARN] Atlas ROI labels are not 1..N. Mapping modules by sorted ROI label order.")
        for i in range(len(modules)):
            roi_label = int(roi_labels[i])
            out[atlas_data == roi_label] = int(modules[i])

    return nib.Nifti1Image(out, atlas_img.affine, atlas_img.header)


def fixed_module_cmap(K: int) -> ListedColormap:
    """
    Fixed categorical colors (no gradient).
    Index 0 is transparent background.
    Module 1=blue, 2=yellow, etc.
    """
    bg = (0, 0, 0, 0)  # transparent

    # Feel free to change these to your preferred mapping.
    # (RGBA or RGB; matplotlib accepts both)
    palette = [
        (0.121, 0.466, 0.705, 1.0),  # 1 blue
        (1.000, 0.498, 0.054, 1.0),  # 2 orange (often reads like yellow-ish)
        (0.172, 0.627, 0.172, 1.0),  # 3 green
        (0.839, 0.153, 0.157, 1.0),  # 4 red
        (0.580, 0.404, 0.741, 1.0),  # 5 purple
        (0.549, 0.337, 0.294, 1.0),  # 6 brown
        (0.890, 0.467, 0.761, 1.0),  # 7 pink
        (0.498, 0.498, 0.498, 1.0),  # 8 gray
        (0.737, 0.741, 0.133, 1.0),  # 9 yellow-green
        (0.090, 0.745, 0.811, 1.0),  # 10 cyan
    ]

    # If K > palette length, extend using tab20 (still discrete)
    if K > len(palette):
        extra = plt.get_cmap("tab20")
        for i in range(K - len(palette)):
            palette.append(extra(i % 20))

    colors = [bg] + palette[:K]
    return ListedColormap(colors, name=f"modules_fixed_{K}")


def save_legend_png(K: int, cmap: ListedColormap, out_path: Path):
    fig_h = max(2.0, 0.35 * K)
    fig, ax = plt.subplots(figsize=(4.2, fig_h))
    ax.set_axis_off()

    y = K
    for m in range(1, K + 1):
        ax.add_patch(plt.Rectangle((0.1, y - 0.8), 0.35, 0.6, color=cmap(m)))
        ax.text(0.55, y - 0.5, f"Module {m}", va="center", fontsize=10)
        y -= 1

    ax.set_xlim(0, 2.8)
    ax.set_ylim(0, K + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main():
    module_npy, module_txt = find_latest_module_file()
    if module_npy is None and module_txt is None:
        raise FileNotFoundError("No module files found in results/louvain_compare or results/group_connectomes.")

    module_path = module_npy if module_npy is not None else module_txt
    label = module_path.stem.replace("_best_modules", "")
    print(f"[INFO] Using modules: {module_path}")

    modules = load_modules(module_txt, module_npy)
    if len(modules) != EXPECTED_N:
        print(f"[WARN] modules length is {len(modules)} not {EXPECTED_N}")

    K = int(modules.max())
    print(f"[INFO] K={K}")

    atlas_path = find_cc200_atlas(ROOT)
    print(f"[INFO] Using atlas: {atlas_path}")
    atlas_img = nib.load(str(atlas_path))

    mod_img = build_module_map(atlas_img, modules)

    out_nii = OUT_DIR / f"{label}_CC200_modulemap_K{K}.nii.gz"
    nib.save(mod_img, str(out_nii))
    print(f"Saved NIfTI -> {out_nii}")

    cmap = fixed_module_cmap(K)

    # Key trick: set vmin/vmax to half-steps so integers map cleanly,
    # and turn OFF nilearn’s continuous colorbar (use the legend PNG instead).
    vmin = 0.5
    vmax = K + 0.5

    out_glass = OUT_DIR / f"{label}_CC200_modules_K{K}_glass.png"
    disp = plotting.plot_glass_brain(
        mod_img,
        title=f"{label} CC200 modules (K={K})",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,  # avoid gradient bar
        plot_abs=False,
    )
    disp.savefig(str(out_glass), dpi=250)
    disp.close()
    print(f"Saved PNG -> {out_glass}")

    out_slices = OUT_DIR / f"{label}_CC200_modules_K{K}_slices.png"
    disp2 = plotting.plot_roi(
        mod_img,
        title=f"{label} CC200 modules (K={K})",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,  # avoid gradient bar
        draw_cross=False,
        display_mode="ortho",
        cut_coords=(0, -20, 20),
        resampling_interpolation="nearest",  # IMPORTANT for labels
    )
    disp2.savefig(str(out_slices), dpi=250)
    disp2.close()
    print(f"Saved PNG -> {out_slices}")

    out_legend = OUT_DIR / f"{label}_CC200_modules_K{K}_legend.png"
    save_legend_png(K, cmap, out_legend)
    print(f"Saved LEGEND -> {out_legend}")


if __name__ == "__main__":
    main()