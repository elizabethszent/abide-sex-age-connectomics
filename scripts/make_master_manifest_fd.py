#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path

REPO = Path(f"/work/ioannou_lab/{os.environ['USER']}/abide-sex-age-connectomics")
MAN = REPO / "manifests"

def explode_fd(df, dataset):
    rows = []
    for _, r in df.iterrows():
        base = {
            "dataset": dataset,
            "run_id": r.get("run_id"),
            "subject_id": r.get("subject_id"),
            "site": r.get("site"),
            "qc_status": r.get("qc_status"),
        }
        # fd0.2
        if r.get("fd02_status") == "ok" and isinstance(r.get("fd02_npz"), str):
            p = r["fd02_npz"]
            if p and Path(p).exists():
                rows.append({**base, "fd": 0.2, "connectome_npz": p})
        # fd0.3
        if r.get("fd03_status") == "ok" and isinstance(r.get("fd03_npz"), str):
            p = r["fd03_npz"]
            if p and Path(p).exists():
                rows.append({**base, "fd": 0.3, "connectome_npz": p})
    return pd.DataFrame(rows)

ab1 = pd.read_csv(MAN / "abide1_runs_with_site.tsv", sep="\t")
ab2 = pd.read_csv(MAN / "abide2_runs_with_site.tsv", sep="\t")

m1 = explode_fd(ab1, "ABIDE1")
m2 = explode_fd(ab2, "ABIDE2")

master = pd.concat([m1, m2], ignore_index=True)

out = MAN / "master_runs_fd.tsv"
master.to_csv(out, sep="\t", index=False)

print("WROTE:", out)
print("ROWS:", len(master))
print(master.groupby(["dataset","fd"]).size().to_dict())
print("unique subjects:", master["subject_id"].nunique())
