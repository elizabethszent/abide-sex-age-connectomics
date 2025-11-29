import numpy as np
import pandas as pd
from pathlib import Path
import glob, os

TS_DIR  = "C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/roi_timeseries/cpac/nofilt_noglobal/rois_cc200"
OUT_DIR = "C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject/data/connectomes/cpac/nofilt_noglobal/cc200_z"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

files = glob.glob(os.path.join(TS_DIR, "*_rois_cc200.1D"))
print(f"Found {len(files)} ROI time-series files")

for fp in files:
    sub = Path(fp).stem.replace("_rois_cc200", "")
    df = pd.read_csv(
        fp,
        sep=r"\s+",
        engine="python",
        header=None,
        comment="#",
        dtype=str
    ).apply(pd.to_numeric, errors="coerce")

    #drop all-NaN columns
    df = df.dropna(axis=1, how="all")

    #drop constant columns
    std = df.std(axis=0, numeric_only=True)
    good_cols = std[std > 0].index
    ts = df[good_cols].to_numpy()

    #correlation across ROIs
    C = np.corrcoef(ts.T)
    Z = fisher_z(C)

    np.save(os.path.join(OUT_DIR, f"{sub}.npy"), Z)

print("Done.")
