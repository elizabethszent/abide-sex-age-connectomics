import os, argparse, glob, sys
from collections import Counter
import numpy as np
import pandas as pd

TS_DIR = os.path.join("data", "roi_timeseries", "cpac", "nofilt_noglobal", "rois_cc200")
META_CSV = os.path.join("data","female" "metrics_merged.csv")
OUT_DIR  = "data/female/processed"

def read_timeseries(file_id):
    fp = os.path.join(TS_DIR, f"{file_id}_rois_cc200.1D")
    if not os.path.exists(fp):
        return None
    ts = pd.read_csv(fp, delim_whitespace=True, header=None)

    ts = ts.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    good_cols = ts.std(axis=0) > 0
    ts = ts.loc[:, good_cols.values]

    roi_idx = np.where(good_cols.values)[0]
    return ts.values, roi_idx

def subject_edges(file_id, top_prop=0.10):
    out = read_timeseries(file_id)
    if out is None:
        return None
    ts, roi_idx = out

    C = np.corrcoef(ts, rowvar=False)
    np.fill_diagonal(C, 0.0)

    iu = np.triu_indices(C.shape[0], k=1)
    vals = np.abs(C[iu])
    if np.all(np.isnan(vals)):
        return []
    thr = np.nanquantile(vals, 1.0 - top_prop)
    keep = vals >= thr
    ui = iu[0][keep]
    uj = iu[1][keep]

    edges = [(int(roi_idx[i]), int(roi_idx[j])) for i, j in zip(ui, uj)]

    edges = [(i, j) if i < j else (j, i) for i, j in edges]
    return edges

def main():
    parser = argparse.ArgumentParser(description="Build group weighted edge list (weights = #subjects with edge).")
    parser.add_argument("--group", choices=["All", "ASD", "Control"], default="All",
                        help="Which cohort to aggregate.")
    parser.add_argument("--prop", type=float, default=0.10,
                        help="Proportional density threshold (default 0.10 = top 10%% |r|).")
    args = parser.parse_args()

    if not os.path.exists(META_CSV):
        sys.exit(f"Missing metadata: {META_CSV}")

    meta = pd.read_csv(META_CSV, usecols=["FILE_ID","DX_GROUP"])
    meta["FILE_ID"]  = meta["FILE_ID"].astype(str).str.strip()
    meta["DX_GROUP"] = meta["DX_GROUP"].astype(int)

    if args.group == "ASD":
        meta = meta.loc[meta["DX_GROUP"] == 1].copy()
    elif args.group == "Control":
        meta = meta.loc[meta["DX_GROUP"] == 2].copy()

    file_ids = meta["FILE_ID"].unique().tolist()
    if not file_ids:
        sys.exit("No subjects found for the selected group.")

    counts = Counter()
    missing = []
    for fid in file_ids:
        edges = subject_edges(fid, top_prop=args.prop)
        if edges is None:
            missing.append(fid)
            continue
        counts.update(edges)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"edgelist_{args.group}.txt")

    #sort like your example: by i then j
    with open(out_path, "w", encoding="utf-8") as f:
        for (i, j) in sorted(counts.keys()):
            w = counts[(i, j)]
            f.write(f"{i} {j} {{'weight': {w}}}\n")

    print(f"Subjects processed: {len(file_ids)} | edges written: {len(counts)}")
    if missing:
        print(f"Missing time-series for {len(missing)} subject(s) (first few): {missing[:5]}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
