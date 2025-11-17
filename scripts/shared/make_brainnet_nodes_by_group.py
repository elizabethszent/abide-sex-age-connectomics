# scripts/shared/make_brainnet_nodes_by_group.py

import numpy as np
from pathlib import Path


BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# Your existing node template (with CC200 coordinates + colors + labels)
TEMPLATE_NODE = BASE / r"results\vis\brainnet\CC200_base.node"

# Group-mean connectivity matrices (one per sex × diagnosis)
# !! EDIT these to the actual filenames you saved earlier !!
GROUP_MATS = {
    "female_ASD":    BASE / r"results\group_connectomes\F_ASD_Zmean.npy",
    "female_Control":BASE / r"results\group_connectomes\F_CTL_Zmean.npy",
    "male_ASD":      BASE / r"results\group_connectomes\M_ASD_Zmean.npy",
    "male_Control":  BASE / r"results\group_connectomes\M_CTL_Zmean.npy",
}

OUT_DIR = BASE / r"results\vis\brainnet\nodes_by_group"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD TEMPLATE NODE FILE ===
# BrainNet .node format: x y z size color label
# We'll keep x,y,z, color, label; replace "size" with group-specific metric
template = np.loadtxt(TEMPLATE_NODE, dtype=str)

if template.shape[1] < 6:
    raise ValueError(
        f"Expected 6 columns in {TEMPLATE_NODE}, found {template.shape[1]}."
        " BrainNet .node should be: x y z size color label."
    )

# Coordinates as float
coords = template[:, 0:3].astype(float)   # (200, 3)
# Original color index (we'll reuse so parcels keep the same colors)
colors = template[:, 4].astype(float)
# Labels (ROI names / indices)
labels = template[:, 5]

n_rois = coords.shape[0]
print(f"Template has {n_rois} nodes")

def compute_node_strength(mat: np.ndarray) -> np.ndarray:
    """
    Simple nodal metric: weighted degree (strength).
    Assumes symmetric [n, n] matrix.
    Negative weights are set to 0.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix is not square: {mat.shape}")
    if mat.shape[0] != n_rois:
        raise ValueError(
            f"Matrix size {mat.shape[0]} does not match node file ({n_rois})"
        )

    # Use absolute or positive part; here positive weights only
    M = np.array(mat, copy=True)
    M[M < 0] = 0.0
    # strength = sum of weights per node
    strength = M.sum(axis=0)
    return strength

def scale_to_1_10(x: np.ndarray) -> np.ndarray:
    """
    Scale a vector to [1, 10] for nice node sizes in BrainNet.
    """
    x = x.astype(float)
    xmin, xmax = x.min(), x.max()
    if np.isclose(xmax, xmin):
        return np.ones_like(x) * 5.0  # all equal: medium size
    return 1.0 + 9.0 * (x - xmin) / (xmax - xmin)

# === PROCESS EACH GROUP ===
for name, mat_path in GROUP_MATS.items():
    if not mat_path.exists():
        print(f"[WARN] Missing matrix for {name}: {mat_path}")
        continue

    print(f"\nProcessing group: {name}")
    mat = np.load(mat_path)

    # Compute nodal strength for this group
    # Compute nodal strength for this group
    strength = compute_node_strength(mat)
    size_col = scale_to_1_10(strength)

    out_path = OUT_DIR / f"{name}_strength.node"

    # Manually write BrainNet .node: x y z size color label
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n_rois):
            x, y, z = coords[i]
            size = size_col[i]
            color = colors[i]
            label = labels[i]
            f.write(f"{x:.3f}\t{y:.3f}\t{z:.3f}\t{size:.3f}\t{color:.0f}\t{label}\n")

    print(f"  -> Saved {out_path}")

