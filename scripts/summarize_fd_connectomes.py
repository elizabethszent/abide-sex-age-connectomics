#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(f"/work/ioannou_lab/{os.environ['USER']}/abide-sex-age-connectomics")
df = pd.read_csv(REPO / "manifests/master_runs_fd.tsv", sep="\t")

def summarize_npz(path):
    z = np.load(path)
    r = z["r"] if "r" in z else None
    if r is None:
        return {}
    # off-diagonal summary
    m = r.shape[0]
    off = r[~np.eye(m, dtype=bool)]
    return {
        "mean_r": float(off.mean()),
        "std_r": float(off.std()),
        "mean_abs_r": float(np.abs(off).mean()),
    }

rows = []
for _, r in df.iterrows():
    p = r["connectome_npz"]
    s = summarize_npz(p)
    rows.append({
        "dataset": r["dataset"],
        "fd": r["fd"],
        "site": r.get("site"),
        "run_id": r["run_id"],
        "subject_id": r["subject_id"],
        **s
    })

out = pd.DataFrame(rows)
out_path = REPO / "manifests/fd_connectome_summary.tsv"
out.to_csv(out_path, sep="\t", index=False)

print("WROTE:", out_path)
print("Counts:", out.groupby(["dataset","fd"]).size().to_dict())
print("Mean abs r by fd:", out.groupby(["dataset","fd"])["mean_abs_r"].mean().to_dict())
