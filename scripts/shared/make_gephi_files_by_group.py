import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# CC200 base node file 
TEMPLATE_NODE = BASE / r"results\vis\brainnet\CC200_base.node"

# Louvain modules 
MODULES_FILE = BASE / r"results\group_connectomes\CC200_modules.npy"

#Group-mean connectivity matrices 
GROUP_MATS = {
    "female_ASD":     BASE / r"results\group_connectomes\F_ASD_Zmean.npy",
    "female_Control": BASE / r"results\group_connectomes\F_CTL_Zmean.npy",
    "male_ASD":       BASE / r"results\group_connectomes\M_ASD_Zmean.npy",
    "male_Control":   BASE / r"results\group_connectomes\M_CTL_Zmean.npy",
}

# Edge density for Gephi 
EDGE_DENSITY = 0.10  # 0.10 = top 10%; set to None to keep all positive edges

OUT_DIR = BASE / r"results\vis\gephi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#LOAD TEMPLATE NODE FILE
# BrainNet .node: x y z ? ? label
template = np.loadtxt(TEMPLATE_NODE, dtype=str)
if template.shape[1] < 6:
    raise ValueError(
        f"Expected 6 columns in {TEMPLATE_NODE}, found {template.shape[1]}."
    )

coords = template[:, 0:3].astype(float) #(200, 3)
labels = template[:, 5] # CC200_ROI_###

n_rois = coords.shape[0]
print(f"Template has {n_rois} nodes")

#LOAD 7 MODULES
modules = np.load(MODULES_FILE)
if modules.shape[0] != n_rois:
    raise ValueError(
        f"Modules length {modules.shape[0]} does not match node count {n_rois}"
    )

K = int(modules.max())
print(f"Loaded modules, K = {K} distinct modules")

def compute_node_strength(A: np.ndarray) -> np.ndarray:

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix is not square: {A.shape}")
    if A.shape[0] != n_rois:
        raise ValueError(
            f"Matrix size {A.shape[0]} does not match node count {n_rois}"
        )
    M = np.maximum(A, 0.0)
    np.fill_diagonal(M, 0.0)
    return M.sum(axis=0)

def write_gephi_files_for_group(name: str, A: np.ndarray):

    #prepare adjacency: positive weights only, no self-loops
    M = np.maximum(A, 0.0)
    np.fill_diagonal(M, 0.0)

    #node strength for this group
    strength = M.sum(axis=0)

    #nodes CSV
    nodes_path = OUT_DIR / f"{name}_nodes_gephi.csv"
    with open(nodes_path, "w", encoding="utf-8") as f:
        f.write("Id,Label,x,y,z,Module,Strength\n")
        for i in range(n_rois):
            node_id = i + 1
            label = labels[i]
            x, y, z = coords[i]
            module_id = int(modules[i])
            s = float(strength[i])
            f.write(
                f"{node_id},{label},{x:.3f},{y:.3f},{z:.3f},"
                f"{module_id},{s:.6f}\n"
            )
    print(f"  -> Wrote nodes to {nodes_path}")

    #edges CSV
    #use upper triangle only to avoid duplicates
    iu, ju = np.triu_indices(n_rois, k=1)
    w = M[iu, ju]
    pos_mask = w > 0
    w_pos = w[pos_mask]

    if w_pos.size == 0:
        print(f"  [WARN] No positive edges for {name}; edges file will be empty.")
        edges_path = OUT_DIR / f"{name}_edges_gephi.csv"
        with open(edges_path, "w", encoding="utf-8") as f:
            f.write("Source,Target,Type,Weight\n")
        return

    #keep top EDGE_DENSITY fraction of positive edges
    if EDGE_DENSITY is not None:
        keep_frac = float(EDGE_DENSITY)
        if keep_frac <= 0 or keep_frac > 1:
            raise ValueError("EDGE_DENSITY must be in (0,1].")
        thr = np.quantile(w_pos, 1.0 - keep_frac)
        keep_mask = pos_mask & (w >= thr)
    else:
        keep_mask = pos_mask

    edges_path = OUT_DIR / f"{name}_edges_gephi.csv"
    with open(edges_path, "w", encoding="utf-8") as f:
        f.write("Source,Target,Type,Weight\n")
        for i_idx, j_idx, wt in zip(iu[keep_mask], ju[keep_mask], w[keep_mask]):
            source = i_idx + 1     # must match Id in nodes file
            target = j_idx + 1
            f.write(f"{source},{target},Undirected,{wt:.6f}\n")

    n_edges = int(keep_mask.sum())
    print(f"  -> Wrote edges to {edges_path} ({n_edges} edges)")

#MAIN LOOP OVER GROUPS
for name, mat_path in GROUP_MATS.items():
    if not mat_path.exists():
        print(f"[WARN] Missing matrix for {name}: {mat_path}")
        continue

    print(f"\nProcessing group: {name}")
    A = np.load(mat_path)
    write_gephi_files_for_group(name, A)
