# scripts/shared/group_mean_connectomes.py
import numpy as np
import pandas as pd
from pathlib import Path
import os
from collections import Counter

#paths
ROOT = Path(r"C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")
CONN_DIR = ROOT / "data" / "connectomes" / "cpac" / "nofilt_noglobal" / "cc200_z"

F_META = ROOT / "data" / "female" / "female_metadata_included.csv"
M_META = ROOT / "data" / "male"   / "male_metadata_included.csv"

OUT = ROOT / "results" / "group_connectomes"
OUT.mkdir(parents=True, exist_ok=True)

#name, metadata_csv, DX_GROUP value
GROUPS = [
    ("F_ASD",  F_META, 1),
    ("F_CTL",  F_META, 2),
    ("M_ASD",  M_META, 1),
    ("M_CTL",  M_META, 2),
]

for name, meta_path, dx in GROUPS:
    print(f"\n=== {name} ===")

    if not meta_path.exists():
        print(f"[WARN] Missing metadata file {meta_path} for {name}, skipping.")
        continue

    meta = pd.read_csv(meta_path)
    if "FILE_ID" not in meta.columns or "DX_GROUP" not in meta.columns:
        print(f"[WARN] {meta_path} doesn’t have FILE_ID/DX_GROUP, skipping {name}.")
        continue

    subs = (
        meta.loc[meta["DX_GROUP"] == dx, "FILE_ID"]
        .dropna()
        .astype(str)
        .unique()
    )
    print(f"Candidates in metadata: {len(subs)}")

    #first pass: inspect shapes
    shapes = []
    shape_by_sub = {}
    missing = 0

    for fid in subs:
        fp = CONN_DIR / f"{fid}.npy"
        if not fp.exists():
            missing += 1
            continue
        try:
            arr = np.load(fp)
            shapes.append(arr.shape)
            shape_by_sub[fid] = arr.shape
        except Exception as e:
            print(f"[WARN] Could not load {fp}: {e}")

    if not shapes:
        print(f"[INFO] {name}: no usable matrices found (all missing or failed), skipping.")
        continue

    counts = Counter(shapes)
    target_shape = max(counts.items(), key=lambda x: x[1])[0]
    print(f"Shape counts: {dict(counts)}")
    print(f"Using most common shape for averaging: {target_shape} (n={counts[target_shape]})")
    if len(counts) > 1:
        print("[NOTE] Some subjects have odd-sized matrices and will be excluded from the group mean.")

    #Second pass: actually collect only matrices with target_shape
    mats = []
    used_subs = []
    skipped_shape = 0

    for fid in subs:
        fp = CONN_DIR / f"{fid}.npy"
        if not fp.exists():
            continue
        arr = np.load(fp)
        if arr.shape != target_shape:
            skipped_shape += 1
            continue
        mats.append(arr)
        used_subs.append(fid)

    if not mats:
        print(f"[INFO] {name}: no matrices with target shape {target_shape}, skipping.")
        continue

    mats = np.stack(mats, axis=0) #n_subj, n, n
    Z_mean = np.nanmean(mats, axis=0)

    out_path = OUT / f"{name}_Zmean.npy"
    np.save(out_path, Z_mean)

    print(f"{name}:")
    print(f"  subjects used    : {len(used_subs)}")
    print(f"  missing matrices : {missing}")
    print(f"  wrong-shape skip : {skipped_shape}")
    print(f"  saved group mean : {out_path}")
