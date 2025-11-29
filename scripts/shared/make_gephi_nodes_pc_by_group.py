import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")


TEMPLATE_NODE = BASE / r"results\vis\brainnet\CC200_base.node"
MODULES_FILE  = BASE / r"results\group_connectomes\CC200_modules.npy"

#use your group Z-mean matrices here
GROUP_MATS = {
    "F_ASD":  BASE / r"results\group_connectomes\F_ASD_Zmean.npy",
    "F_CTL":  BASE / r"results\group_connectomes\F_CTL_Zmean.npy",
    "M_ASD":  BASE / r"results\group_connectomes\M_ASD_Zmean.npy",
    "M_CTL":  BASE / r"results\group_connectomes\M_CTL_Zmean.npy",
}

OUT_DIR = BASE / r"results\vis\gephi_nodes_pc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#participation coefficient
def participation_coefficient(A: np.ndarray, modules: np.ndarray) -> np.ndarray:

    A = np.array(A, copy=True, dtype=float)
    A[A < 0] = 0.0
    np.fill_diagonal(A, 0.0)

    n = A.shape[0]
    modules = np.asarray(modules)
    uniq = np.unique(modules)

    pc = np.zeros(n, dtype=float)
    for i in range(n):
        k_i = A[i].sum()
        if k_i <= 0:
            pc[i] = 0.0
            continue
        sum_sq = 0.0
        for m in uniq:
            mask_m = (modules == m)
            k_is = A[i, mask_m].sum()
            sum_sq += (k_is / k_i) ** 2
        pc[i] = 1.0 - sum_sq
    return pc


template = np.loadtxt(TEMPLATE_NODE, dtype=str)
if template.shape[1] < 6:
    raise ValueError(f"{TEMPLATE_NODE} should have at least 6 columns")

labels = template[:, 5] #CC200_ROI_001, etc.
n_rois = labels.shape[0]
print(f"Template has {n_rois} nodes")

modules = np.load(MODULES_FILE)
if modules.shape[0] != n_rois:
    raise ValueError("modules length does not match node count")
print(f"Modules loaded. K = {int(modules.max())} modules")

#one Gephi node file per group
for short_name, mat_path in GROUP_MATS.items():
    if not mat_path.exists():
        print(f"[WARN] Missing matrix for {short_name}: {mat_path}")
        continue

    print(f"\nProcessing group: {short_name}")
    #if your group matrices are saved as .edge instead of .npy:
    #A = np.loadtxt(mat_path)
    A = np.load(mat_path)   # Z-mean matrix, shape (200,200)

    if A.shape[0] != n_rois or A.shape[1] != n_rois:
        raise ValueError(f"{mat_path} has wrong shape: {A.shape}")

    pc = participation_coefficient(A, modules)

    out_path = OUT_DIR / f"{short_name}_nodes_PC.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Id,Module,pc,Label\n")
        for i in range(n_rois):
            #Id is 0-based (to match your gephi_nodes_PC example)
            node_id = i
            module_id = int(modules[i])
            pc_i = float(pc[i])
            label = labels[i]
            f.write(f"{node_id},{module_id},{pc_i:.6f},{label}\n")

    print(f"  -> wrote {out_path}")
