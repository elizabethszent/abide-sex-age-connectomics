import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")


CONN_DIR  = ROOT / "connectomes" / "CC200" / "ABIDE1" / "FDpersubject2"
META_DIR  = ROOT / "connectomes" / "CC200" / "ABIDE1" / "FDpersubject2_meta"
PHENO_DIR = ROOT / "phenotypes" / "ABIDE1"

OUT_DIR   = ROOT / "results" / "group_connectomes" / "ABIDE1_CC200"
OUT_DIR.mkdir(parents=True, exist_ok=True)


FD_LIST = ["0.2"]  


def to_subid7(x) -> str | None:
    if pd.isna(x):
        return None
    try:
        return f"{int(float(x)):07d}"
    except Exception:
        return None

def age_group(age: float) -> str | None:
    if pd.isna(age):
        return None
    a = float(age)
    if a <= 9:
        return "child_0_9"
    if a <= 12:
        return "preteen_10_12"
    if a <= 17:
        return "teen_13_17"
    return "adult_18_plus"

def robust_standardize(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = mad if (mad is not None and mad > 1e-12) else (np.nanstd(x) + 1e-12)
    return (x - med) / scale

def load_json(fp: Path) -> dict:
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(fp.read_text(encoding="latin1"))

def meta_for_run(run_stem: str) -> Path:
    # run_stem like: sub-0050002_task-rest_run-1
    return META_DIR / f"{run_stem}.json"

def parse_run_stem(p: Path) -> tuple[str, str, int] | None:
    """
    Returns (subid7, run_stem, run_num) for a .npy connectome
    """
    m = re.search(r"(sub-(\d{7})_task-rest_run-(\d+))\.npy$", p.name)
    if not m:
        return None
    run_stem = m.group(1)
    subid7 = m.group(2)
    run_num = int(m.group(3))
    return subid7, run_stem, run_num

def load_matrix(fp: Path) -> np.ndarray:
    x = np.load(fp)
    x = np.asarray(x)
    if x.shape != (200, 200):
        raise ValueError(f"{fp.name}: expected (200,200), got {x.shape}")
    return x

def detect_input_space(example_mat: np.ndarray) -> str:
    """
    Guess whether .npy matrices are correlation-like (r in [-1,1])
    or fisher-z-like (z unbounded).
    """
    finite = example_mat[np.isfinite(example_mat)]
    if finite.size == 0:
        return "z"  # default
    # If almost everything is within [-1, 1], treat as r
    frac_in_unit = np.mean((finite >= -1.0001) & (finite <= 1.0001))
    return "r" if frac_in_unit > 0.98 else "z"

def r_to_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def z_to_r(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def build_connectome_index() -> dict[str, dict]:
    """
    Map subid7 -> chosen run record:
      {
        "conn_path": Path,
        "run_stem": str,
        "run_num": int,
        "meta_path": Path,
        "mean_fd": float|nan,
        "n_kept": float|nan,
        "n_total": float|nan,
        "frac_kept": float|nan
      }

    Preference order when multiple runs exist:
      1) run-1 if available
      2) otherwise lowest mean_fd (if present in meta)
      3) otherwise lowest run number
    """
    by_sub: dict[str, list[dict]] = {}

    for fp in CONN_DIR.glob("sub-*_task-rest_run-*.npy"):
        parsed = parse_run_stem(fp)
        if not parsed:
            continue
        subid7, run_stem, run_num = parsed
        mp = meta_for_run(run_stem)

        mean_fd = np.nan
        n_kept = np.nan
        n_total = np.nan
        frac_kept = np.nan

        if mp.exists():
            md = load_json(mp)
            # tolerate different key names
            for k in ["mean_fd", "meanFD", "MEAN_FD"]:
                if k in md:
                    mean_fd = float(md[k])
                    break
            for k in ["n_kept", "N_KEPT"]:
                if k in md:
                    n_kept = float(md[k])
                    break
            for k in ["n_total", "N_TOTAL"]:
                if k in md:
                    n_total = float(md[k])
                    break
            if np.isfinite(n_kept) and np.isfinite(n_total) and n_total != 0:
                frac_kept = n_kept / n_total

        rec = {
            "conn_path": fp,
            "run_stem": run_stem,
            "run_num": run_num,
            "meta_path": mp,
            "mean_fd": mean_fd,
            "n_kept": n_kept,
            "n_total": n_total,
            "frac_kept": frac_kept,
        }
        by_sub.setdefault(subid7, []).append(rec)

    chosen: dict[str, dict] = {}
    for subid7, runs in by_sub.items():
        run1 = [r for r in runs if r["run_num"] == 1]
        if run1:
            chosen[subid7] = run1[0]
            continue

        finite_fd = [r for r in runs if np.isfinite(r["mean_fd"])]
        if finite_fd:
            finite_fd.sort(key=lambda r: r["mean_fd"])
            chosen[subid7] = finite_fd[0]
            continue

        #lowest run number
        runs.sort(key=lambda r: r["run_num"])
        chosen[subid7] = runs[0]

    return chosen


def load_abide1_phenotypes(pheno_dir: Path) -> pd.DataFrame:
    patterns = ["Phenotypic_*.csv", "phenotypic_*.csv", "*.csv"]
    files = []
    for pat in patterns:
        files = sorted(pheno_dir.glob(pat))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"No phenotype CSVs found in {pheno_dir}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp, encoding="latin1")
        df.columns = [c.upper() for c in df.columns]

        need = {"SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN"}
        if not need.issubset(df.columns):
            continue

        keep = ["SITE_ID", "SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "FIQ"]
        keep = [c for c in keep if c in df.columns]
        d2 = df[keep].copy()
        d2["subid7"] = d2["SUB_ID"].map(to_subid7)
        d2["age_group"] = d2["AGE_AT_SCAN"].map(age_group)
        dfs.append(d2)

    if not dfs:
        raise RuntimeError(f"No usable phenotype tables in {pheno_dir}")

    ph = pd.concat(dfs, ignore_index=True)

    # numeric cleanup
    for c in ["DX_GROUP", "SEX", "AGE_AT_SCAN"]:
        ph[c] = pd.to_numeric(ph[c], errors="coerce")
    if "FIQ" in ph.columns:
        ph["FIQ"] = pd.to_numeric(ph["FIQ"], errors="coerce")

    ph = ph.dropna(subset=["subid7", "DX_GROUP", "SEX", "AGE_AT_SCAN", "age_group"]).copy()
    ph["subid7"] = ph["subid7"].astype(str)

    #de-duplicate subjects across site files
    ph = ph.drop_duplicates(subset=["subid7"], keep="first").copy()
    return ph


pheno = load_abide1_phenotypes(PHENO_DIR)
print(f"[INFO] Phenotype subjects loaded (unique): {len(pheno)}")

conn_idx = build_connectome_index()
print(f"[INFO] Unique subjects with at least one connectome: {len(conn_idx)}")

#attach chosen connectome and frac_kept
has_conn = pheno["subid7"].isin(conn_idx.keys())
df = pheno[has_conn].copy()
df["conn_path"] = df["subid7"].map(lambda s: conn_idx[s]["conn_path"])
df["frac_kept"] = df["subid7"].map(lambda s: conn_idx[s]["frac_kept"])
df["run_num"]   = df["subid7"].map(lambda s: conn_idx[s]["run_num"])

print(f"[INFO] Phenotype subjects with connectomes available: {len(df)}")
print("[INFO] Chosen run breakdown (subjects):")
print(df["run_num"].value_counts().sort_index().to_string())

# decide r vs z based on the first file
example_mat = load_matrix(df["conn_path"].iloc[0])
space = detect_input_space(example_mat)
print(f"[INFO] Detected input space: {space} (r=correlation, z=fisher-z)")

#for each FD label 
for fd in FD_LIST:

    selected_rows = []

    for (dx, ag), stratum in df.groupby(["DX_GROUP", "age_group"]):
        females = stratum[stratum["SEX"] == 2].copy()
        males   = stratum[stratum["SEX"] == 1].copy()

        nF = len(females)
        nM = len(males)
        if nF == 0 or nM == 0:
            continue

        n = min(nF, nM)

        #females > males (rare), downsample females too
        if nF > n:
            females = females.sample(n=n, random_state=42)

        cols = ["AGE_AT_SCAN", "frac_kept"]
        if "FIQ" in females.columns:
            cols.append("FIQ")

        #fill missing covariates with stratum medians
        for c in cols:
            med = np.nanmedian(stratum[c].to_numpy())
            females[c] = females[c].fillna(med)
            males[c] = males[c].fillna(med)

        F = females[cols].to_numpy(dtype=float)
        M = males[cols].to_numpy(dtype=float)

        # robust standardize each column using combined values
        ZF = []
        ZM = []
        for j in range(len(cols)):
            combined = np.concatenate([F[:, j], M[:, j]])
            z_comb = robust_standardize(combined)
            zF = z_comb[: F.shape[0]]
            zM = z_comb[F.shape[0] :]
            ZF.append(zF)
            ZM.append(zM)
        ZF = np.stack(ZF, axis=1)
        ZM = np.stack(ZM, axis=1)

        centroid = np.mean(ZF, axis=0)

        dists = np.sum((ZM - centroid) ** 2, axis=1)
        males = males.assign(match_score=dists)
        males_sel = males.nsmallest(n, "match_score")

        females = females.assign(match_score=np.nan)
        selected_rows.append(females)
        selected_rows.append(males_sel)

        print(f"[STRATUM] DX={int(dx)} age={ag}: females={len(females)} males_selected={len(males_sel)} (males_avail={nM})")

    if not selected_rows:
        raise RuntimeError("No matched subjects found.")

    sel = pd.concat(selected_rows, ignore_index=True)
    sel = sel.drop_duplicates(subset=["subid7"])  # safety

    sex_counts = sel["SEX"].value_counts().to_dict()
    print(f"[INFO] Selected total={len(sel)} sex_counts={sex_counts}")

    # nan
    sum_mat = np.zeros((200, 200), dtype=np.float64)
    cnt_mat = np.zeros((200, 200), dtype=np.int32)

    for fp in sel["conn_path"]:
        mat = load_matrix(fp).astype(np.float64)
        mask = np.isfinite(mat)
        sum_mat[mask] += mat[mask]
        cnt_mat[mask] += 1

    mean_mat = np.full((200, 200), np.nan, dtype=np.float64)
    valid = cnt_mat > 0
    mean_mat[valid] = sum_mat[valid] / cnt_mat[valid]

    # produce BOTH mean_z and mean_r reliably
    if space == "z":
        mean_z = mean_mat
        mean_r = z_to_r(mean_z)
    else:
        mean_r = mean_mat
        mean_z = r_to_z(mean_r)

    out_z   = OUT_DIR / f"OVERALL_ageSexMatched_fd-{fd}_mean_z.npy"
    out_r   = OUT_DIR / f"OVERALL_ageSexMatched_fd-{fd}_mean_r.npy"
    out_csv = OUT_DIR / f"OVERALL_ageSexMatched_fd-{fd}_selected_subjects.csv"
    out_ids = OUT_DIR / f"OVERALL_ageSexMatched_fd-{fd}_used_subids.txt"

    np.save(out_z, mean_z)
    np.save(out_r, mean_r)
    sel.to_csv(out_csv, index=False)
    out_ids.write_text("\n".join(sel["subid7"].tolist()), encoding="utf-8")

    print(f"[DONE] Saved:")
    print(f"  {out_z}")
    print(f"  {out_r}")
    print(f"  {out_csv}")
    print(f"  {out_ids}")