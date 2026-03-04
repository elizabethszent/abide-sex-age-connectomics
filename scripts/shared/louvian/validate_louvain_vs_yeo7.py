import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations

import nibabel as nib
from nilearn import datasets, image

ROOT = Path(r"C:\Users\eliza\Connectomics\TERMproject\abide-sex-age-connectomics")

CC200_ATLAS = ROOT / r"atlases\cc200\cc200_roi_atlas.nii.gz"

DEFAULT_MODULE_DIRS = [
    ROOT / r"results\group_connectomes",
    ROOT / r"results\louvain_compare",
    ROOT / r"results\louvain_bestK7_vs_yeo",
]

MODULE_GLOBS = [
    "CC200_modules_*.txt",
    "CC200_modules_*.npy",
    "*_best_modules.txt",
    "*_best_modules.npy",
]

OUT_DIR = ROOT / r"results\qc\module_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 200

YEO7_LABELS = [
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Limbic",
    "Frontoparietal",
    "DefaultMode",
]



def discover_module_files() -> list[Path]:
    files: list[Path] = []
    for d in DEFAULT_MODULE_DIRS:
        if not d.exists():
            continue
        for g in MODULE_GLOBS:
            files.extend(sorted(d.glob(g)))

    seen = set()
    out = []
    for f in files:
        r = f.resolve()
        if r in seen:
            continue
        seen.add(r)
        out.append(f)
    return out


def load_modules(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npy":
        v = np.load(path)
        return np.asarray(v).reshape(-1).astype(int)

    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if not lines:
        raise ValueError(f"Empty module file: {path}")

    if ("ROI_index" in lines[0]) and ("Module" in lines[0]):
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        if "ROI_index" not in df.columns or "Module" not in df.columns:
            raise ValueError(f"{path} must contain columns ROI_index and Module")
        df = df.sort_values("ROI_index")
        return df["Module"].to_numpy().astype(int)

    vals = []
    for ln in lines:
        ln = ln.strip()
        if ln:
            vals.append(int(float(ln)))
    return np.array(vals, dtype=int)


def pick_yeo7_path(yeo_bunch) -> str:
    candidates = ["thin_7", "thick_7", "yeo_7", "maps_7", "thin7", "thick7"]
    for k in candidates:
        if hasattr(yeo_bunch, k):
            p = getattr(yeo_bunch, k)
            if isinstance(p, str) and (p.endswith(".nii") or p.endswith(".nii.gz")):
                return p

    if hasattr(yeo_bunch, "keys"):
        for key in yeo_bunch.keys():
            v = yeo_bunch[key]
            if isinstance(v, str) and (v.endswith(".nii") or v.endswith(".nii.gz")):
                name = Path(v).name.lower()
                if ("7" in name) and ("17" not in name):
                    return v

    raise RuntimeError("Could not find a Yeo7 NIfTI path from fetch_atlas_yeo_2011().")


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



def compute_match_metrics(ctab: pd.DataFrame, n_total: int):
    if n_total <= 0:
        return 0, 0.0, float("nan"), float("nan"), {}

    ctab = ctab.copy()
    for y in YEO7_LABELS:
        if y not in ctab.columns:
            ctab[y] = 0

    purity_hits = int(ctab.max(axis=1).sum())
    purity_pct = 100.0 * purity_hits / n_total

    modules_sorted = list(ctab.index.astype(int))
    K = len(modules_sorted)

    best_hits = float("nan")
    best_map = {}

    if K <= len(YEO7_LABELS):
        counts = {(int(m), y): int(ctab.loc[m, y]) for m in modules_sorted for y in YEO7_LABELS}
        best_total = -1
        best_mapping = None
        for perm in permutations(YEO7_LABELS, r=K):
            total = 0
            for m, y in zip(modules_sorted, perm):
                total += counts[(m, y)]
            if total > best_total:
                best_total = total
                best_mapping = dict(zip(modules_sorted, perm))
        best_hits = int(best_total)
        best_map = best_mapping if best_mapping is not None else {}

    best_pct = (100.0 * best_hits / n_total) if isinstance(best_hits, int) else float("nan")
    return purity_hits, purity_pct, best_hits, best_pct, best_map



def main():
    if not CC200_ATLAS.exists():
        raise FileNotFoundError(f"Missing CC200 atlas: {CC200_ATLAS}")

    module_files = discover_module_files()
    if not module_files:
        raise FileNotFoundError(
            "No module files found. Looked in:\n"
            + "\n".join(str(d) for d in DEFAULT_MODULE_DIRS)
            + "\nwith globs:\n"
            + "\n".join(MODULE_GLOBS)
        )

    print("Module files found:")
    for f in module_files:
        print(f"  - {f}")

    cc200_img = nib.load(str(CC200_ATLAS))
    cc200_data = cc200_img.get_fdata().astype(int)

    roi_labels = np.unique(cc200_data)
    roi_labels = roi_labels[roi_labels != 0]
    roi_labels = np.sort(roi_labels)
    print(f"CC200 atlas nonzero ROI labels found: {len(roi_labels)}")

    yeo_rs, label_map = yeo7_to_cc200_space(cc200_img)
    yeo_data = yeo_rs.get_fdata().astype(int)

    summary_rows = []

    for mf in module_files:
        label = mf.stem
        out_subdir = OUT_DIR / label
        out_subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Validating {label} ===")
        modules = load_modules(mf)

        if modules.ndim != 1:
            raise ValueError(f"Modules not 1D: {mf} -> {modules.shape}")

        if len(modules) != EXPECTED_N:
            print(f"[WARN] modules length is {len(modules)} (expected {EXPECTED_N})")
        if len(roi_labels) != EXPECTED_N:
            print(f"[WARN] atlas ROI count is {len(roi_labels)} (expected {EXPECTED_N})")

        N = min(len(modules), len(roi_labels))

        roi_to_yeo = np.zeros(N, dtype=int)
        for i in range(N):
            roi_label_i = int(roi_labels[i])
            roi_to_yeo[i] = roi_majority_vote(cc200_data, yeo_data, roi_label_i)

        per_roi = pd.DataFrame({
            "ROI_index": np.arange(1, N + 1),
            "ROI_label_in_atlas": roi_labels[:N].astype(int),
            "Louvain_module": modules[:N].astype(int),
            "Yeo7_label": roi_to_yeo.astype(int),
            "Yeo7_name": [label_map.get(int(x), f"Label_{int(x)}") for x in roi_to_yeo],
        })

        out_per_roi = out_subdir / "cc200_roi_to_yeo7_and_module.csv"
        per_roi.to_csv(out_per_roi, index=False)

        ctab = pd.crosstab(per_roi["Louvain_module"], per_roi["Yeo7_name"])
        (out_subdir / "louvain_module_x_yeo7_counts.csv").write_text(ctab.to_csv(), encoding="utf-8")

        ctab_pct = ctab.div(ctab.sum(axis=1), axis=0) * 100.0
        (out_subdir / "louvain_module_x_yeo7_rowpct.csv").write_text(ctab_pct.to_csv(), encoding="utf-8")

        print("\n=== Louvain module -> dominant Yeo7 network ===")
        for m in sorted(per_roi["Louvain_module"].unique()):
            sub = per_roi[per_roi["Louvain_module"] == m]
            vc = sub["Yeo7_name"].value_counts()
            top = vc.index[0]
            hits = int(vc.iloc[0])
            total = int(len(sub))
            pct = 100.0 * hits / total if total > 0 else 0.0
            print(f"Module {m}: {top} ({hits}/{total} ROIs, {pct:.1f}%)")


        purity_hits, purity_pct, best_hits, best_pct, best_map = compute_match_metrics(ctab, n_total=N)
        print("\n=== Overall match % ===")
        print(f"Purity (dominant overlap): {purity_hits}/{N}  ->  {purity_pct:.1f}%")
        if isinstance(best_hits, int):
            print(f"Best 1-to-1 mapping:       {best_hits}/{N}  ->  {best_pct:.1f}%")

        summary_rows.append({
            "label": label,
            "module_file": str(mf),
            "N_rois_used": N,
            "K_modules": int(per_roi["Louvain_module"].nunique()),
            "purity_hits": purity_hits,
            "purity_pct": purity_pct,
            "best_1to1_hits": best_hits,
            "best_1to1_pct": best_pct,
        })

        print(f"\nSaved -> {out_subdir}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        out_summary = OUT_DIR / "module_vs_yeo7_match_summary.csv"
        summary.to_csv(out_summary, index=False)
        print(f"\n[SAVED] {out_summary}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()