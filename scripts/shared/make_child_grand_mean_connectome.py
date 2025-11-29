
from pathlib import Path
import numpy as np
import pandas as pd

#paths 
BASE = Path(r"C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")

F_CHILD = BASE / "data" / "female" / "child_metadata_included.csv"
M_CHILD = BASE / "data" / "male"   / "child_metadata_included.csv"

CONN_DIR = BASE / "data" / "connectomes" / "cpac" / "nofilt_noglobal" / "cc200_z"

OUT_DIR = BASE / "results" / "shared"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "child_grand_mean_connectome.npy"

N_ROI = 200 #CC200

#load metadata 
f_meta = pd.read_csv(F_CHILD)
m_meta = pd.read_csv(M_CHILD)

meta = pd.concat([f_meta[["FILE_ID"]], m_meta[["FILE_ID"]]],ignore_index=True).drop_duplicates()

print(f"Total child subjects (F+M, included in metadata): {len(meta)}")

mats = []
skipped_shape = []
missing = []

for fid in meta["FILE_ID"]:
    fp = CONN_DIR / f"{fid}.npy"
    if not fp.exists():
        missing.append(fid)
        continue

    mat = np.load(fp)

    if mat.shape != (N_ROI, N_ROI):
        skipped_shape.append((fid, mat.shape))
        continue

    mats.append(mat)

print(f"Found {len(mats)} matrices with shape {N_ROI}x{N_ROI}")

if missing:
    print(f"Missing {len(missing)} matrices (first few): {missing[:10]}")

if skipped_shape:
    print("Skipped due to non-200x200 shape (first few):")
    for fid, sh in skipped_shape[:10]:
        print(f"  {fid}: {sh}")

if not mats:
    raise RuntimeError("No valid 200x200 matrices found – check CONN_DIR and FILE_IDs")

mats = np.stack(mats, axis=0)
grand_mean = mats.mean(axis=0)

np.save(OUT_PATH, grand_mean)
print(f"Saved grand-mean child connectome -> {OUT_PATH}")
