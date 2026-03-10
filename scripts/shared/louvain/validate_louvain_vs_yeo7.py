import numpy as np
import pandas as pd
from pathlib import Path

import nibabel as nib
from nilearn import datasets, image


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "results").exists() and (p / "atlases").exists():
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

CC200_ATLAS = ROOT / "atlases" / "cc200" / "cc200_roi_atlas.nii.gz"
MODULE_DIR = ROOT / "results" / "group_connectomes" / "ABIDE12_CC200"
MODULE_PATTERNS = [
    "ABIDE_modules_asym_min10_fd-*.txt",
    "ABIDE_modules_asym_min10_fd-*.npy",
]

OUT_DIR = ROOT / "results" / "qc" / "module_validation" / "ABIDE12_CC200"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 200


def discover_module_files() -> list[Path]:
    chosen = {}

    for p in sorted(MODULE_DIR.glob("ABIDE_modules_asym_min10_fd-*")):
        if p.suffix not in {".txt", ".npy"}:
            continue

        stem = p.stem
        if stem not in chosen:
            chosen[stem] = p
        else:
            if chosen[stem].suffix == ".npy" and p.suffix == ".txt":
                chosen[stem] = p

    return sorted(chosen.values())


def load_modules(module_path: Path) -> np.ndarray:
    if module_path.suffix == ".npy":
        mods = np.load(module_path)
        mods = np.asarray(mods).astype(int).reshape(-1)
        return mods

    if module_path.suffix == ".txt":
        df = pd.read_csv(module_path, sep=r"\s+")
        if "ROI_index" not in df.columns or "Module" not in df.columns:
            raise ValueError(f"{module_path} must contain columns ROI_index and Module")
        df = df.sort_values("ROI_index")
        mods = df["Module"].to_numpy().astype(int)
        return mods

    raise ValueError(f"Unsupported module file type: {module_path}")


def pick_yeo7_path(yeo_bunch) -> str:
    candidates = ["thin_7", "thick_7", "yeo_7", "maps_7", "yeo7", "thin7", "thick7"]

    for k in candidates:
        if hasattr(yeo_bunch, k):
            p = getattr(yeo_bunch, k)
            if isinstance(p, str) and p.endswith((".nii", ".nii.gz")):
                return p

    for key in getattr(yeo_bunch, "keys", lambda: [])():
        v = yeo_bunch[key]
        if isinstance(v, str) and v.endswith((".nii", ".nii.gz")):
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
    mask = cc200_data == roi_label
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
    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] module dir: {MODULE_DIR}")
    print(f"[INFO] output dir: {OUT_DIR}")
    print(f"[INFO] atlas: {CC200_ATLAS}")

    if not CC200_ATLAS.exists():
        raise FileNotFoundError(f"Missing CC200 atlas: {CC200_ATLAS}")

    module_files = discover_module_files()
    if not module_files:
        raise FileNotFoundError(
            f"No module files found in {MODULE_DIR} matching ABIDE_modules_asym_min10_fd-*"
        )

    print(f"[INFO] Found {len(module_files)} module file(s)")

    cc200_img = nib.load(str(CC200_ATLAS))
    cc200_data = cc200_img.get_fdata().astype(int)

    roi_labels = np.unique(cc200_data)
    roi_labels = roi_labels[roi_labels != 0]
    roi_labels = np.sort(roi_labels)

    print(f"Atlas ROI labels found (nonzero): {len(roi_labels)}")

    yeo_rs, label_map = yeo7_to_cc200_space(cc200_img)
    yeo_data = yeo_rs.get_fdata().astype(int)

    for mf in module_files:
        label = mf.stem
        print(f"\n=== validating {label} ===")
        print(f"[INFO] module file: {mf}")

        modules = load_modules(mf)

        if len(modules) != len(roi_labels):
            print(f"[WARN] modules length ({len(modules)}) != atlas ROI count ({len(roi_labels)})")
        if len(modules) != EXPECTED_N:
            print(f"[WARN] modules length is {len(modules)} not {EXPECTED_N}")

        N = min(len(modules), len(roi_labels))

        roi_to_yeo = np.zeros(N, dtype=int)
        for i in range(N):
            roi_label_i = int(roi_labels[i])
            roi_to_yeo[i] = roi_majority_vote(cc200_data, yeo_data, roi_label_i)

        per_roi = pd.DataFrame(
            {
                "ROI_index": np.arange(1, N + 1),
                "ROI_label_in_atlas": roi_labels[:N].astype(int),
                "Louvain_module": modules[:N].astype(int),
                "Yeo7_label": roi_to_yeo.astype(int),
                "Yeo7_name": [label_map.get(int(x), f"Label_{int(x)}") for x in roi_to_yeo],
            }
        )

        out_subdir = OUT_DIR / label
        out_subdir.mkdir(parents=True, exist_ok=True)

        per_roi.to_csv(out_subdir / "cc200_roi_to_yeo7_and_module.csv", index=False)

        ctab = pd.crosstab(per_roi["Louvain_module"], per_roi["Yeo7_name"])
        ctab.to_csv(out_subdir / "louvain_module_x_yeo7_counts.csv")

        ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100.0
        ctab_pct.to_csv(out_subdir / "louvain_module_x_yeo7_rowpct.csv")

        print("Dominant Yeo7 per module:")
        for m in sorted(per_roi["Louvain_module"].unique()):
            sub = per_roi[per_roi["Louvain_module"] == m]
            vc = sub["Yeo7_name"].value_counts()
            top = vc.index[0]
            print(f"  Module {m}: {top} ({vc.iloc[0]}/{len(sub)} ROIs)")

        print(f"Saved -> {out_subdir}")


if __name__ == "__main__":
    main()