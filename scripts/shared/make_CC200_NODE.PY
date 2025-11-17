import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine

# ---------------------------------------------------------
# 1. CHANGE THIS to point to your CC200 atlas NIfTI
# ---------------------------------------------------------
ATLAS_PATH = r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\abide\preprocessing\resources\abide_rois\CC200.nii\CC200.nii"

# 2. CHANGE THIS if you want a different output filename
OUT_NODE = "CC200_base.node"


def get_roi_centers_3d(data, affine):
    """
    data: 3D array with integer labels (0 = background, 1..N = ROIs)
    returns: roi_ids (sorted), centers [N x 3] in MNI space
    """
    roi_ids = sorted(int(i) for i in np.unique(data) if i > 0)
    centers = []
    for roi in roi_ids:
        mask = data == roi
        idx = np.column_stack(np.where(mask))   # voxel indices (i,j,k)
        if idx.size == 0:
            centers.append(np.array([np.nan, np.nan, np.nan]))
            continue
        xyz = apply_affine(affine, idx)        # voxel -> world (MNI) coords
        center = xyz.mean(axis=0)
        centers.append(center)
    return roi_ids, np.array(centers)


def get_roi_centers_4d(data, affine):
    """
    data: 4D array of shape (X,Y,Z,N_ROI), each volume is one ROI mask
    returns: roi_ids (1..N), centers [N x 3] in MNI space
    """
    n_rois = data.shape[3]
    roi_ids = list(range(1, n_rois + 1))
    centers = []
    for r in range(n_rois):
        vol = data[..., r]
        mask = vol > 0
        idx = np.column_stack(np.where(mask))
        if idx.size == 0:
            centers.append(np.array([np.nan, np.nan, np.nan]))
            continue
        xyz = apply_affine(affine, idx)
        center = xyz.mean(axis=0)
        centers.append(center)
    return roi_ids, np.array(centers)


def main():
    print(f"Loading atlas from {ATLAS_PATH}")
    img = nib.load(ATLAS_PATH)
    data = img.get_fdata()
    affine = img.affine

    # Squeeze singleton dimensions (e.g. (X,Y,Z,1) -> (X,Y,Z))
    data = np.squeeze(data)

    if data.ndim == 3:
        print("Detected 3D labeled atlas.")
        roi_ids, centers = get_roi_centers_3d(data, affine)
    elif data.ndim == 4:
        print("Detected 4D ROI-atlas (one volume per ROI).")
        roi_ids, centers = get_roi_centers_4d(data, affine)
    else:
        raise ValueError(f"Unexpected data ndim={data.ndim} for atlas.")

    n_rois = len(roi_ids)
    print(f"Found {n_rois} ROIs")

    # Sanity check: we expect 200 for CC200
    if n_rois != 200:
        print("WARNING: Expected 200 ROIs for CC200 but found", n_rois)

    # ---------------------------------------------------------
    # Write BrainNet .node file
    # Format: x y z col4 col5 label
    # We'll set col4 = 1, col5 = 1 for now.
    # ---------------------------------------------------------
    with open(OUT_NODE, "w") as f:
        for roi_id, (x, y, z) in zip(roi_ids, centers):
            if np.isnan(x):
                # Shouldn't happen for CC200, but just in case
                x = y = z = 0.0
            size = 1.0   # placeholder; will be replaced with degree later
            color = 1.0  # placeholder
            label = f"CC200_ROI_{roi_id:03d}"
            f.write(f"{x:.4f}\t{y:.4f}\t{z:.4f}\t{size:.4f}\t{color:.4f}\t{label}\n")

    print(f"Wrote {OUT_NODE}")


if __name__ == "__main__":
    main()
