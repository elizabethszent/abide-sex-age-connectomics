import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
        if (p / "scripts").exists() and (p / "phenotypes").exists():
            return p
    return start



BINS = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]

SEX_MAP = {1: "Male", 2: "Female"}
DX_MAP = {1: "ASD", 2: "Control"}


def age_group_series(age: pd.Series) -> pd.Series:
    return pd.cut(
        age,
        bins=BINS,
        labels=LABELS,
        right=False,
        include_lowest=True,
    )


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def load_json_robust(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="latin1"))


def to_int_safe(x):
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def robust_standardize(x: np.ndarray) -> np.ndarray:
    """
    Safe standardization that never warns on all-NaN / constant vectors.
    Returns zeros where it cannot standardize.
    """
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=float)

    med = np.nanmedian(finite)
    mad = np.nanmedian(np.abs(finite - med))

    if not np.isfinite(mad) or mad <= 1e-12:
        sd = np.nanstd(finite)
        scale = sd if (np.isfinite(sd) and sd > 1e-12) else 1.0
    else:
        scale = mad

    z = (x - med) / scale
    z[~np.isfinite(z)] = 0.0
    return z


def detect_space(mat: np.ndarray) -> str:
    """
    r: almost all values in [-1,1]
    z: otherwise
    """
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return "z"
    frac_in_unit = np.mean((finite >= -1.0001) & (finite <= 1.0001))
    return "r" if frac_in_unit > 0.98 else "z"


def r_to_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def load_abide1_phenotypes(pheno_dir: Path) -> pd.DataFrame:
    files = sorted(list(pheno_dir.glob("Phenotypic_*.csv"))) + sorted(list(pheno_dir.glob("phenotypic_*.csv")))
    if not files:
        raise FileNotFoundError(f"ABIDE1: No phenotypic_*.csv found in {pheno_dir}")

    dfs = []
    for fp in files:
        df = read_csv_robust(fp)
        df.columns = [c.upper().strip() for c in df.columns]

        need = {"SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"}
        if not need.issubset(df.columns):
            continue

        keep = ["SITE_ID", "SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]
        if "FIQ" in df.columns:
            keep.append("FIQ")
        keep = [c for c in keep if c in df.columns]

        d = df[keep].copy()
        d["DATASET"] = "ABIDE1"
        dfs.append(d)

    if not dfs:
        raise RuntimeError(f"ABIDE1: No usable phenotype tables in {pheno_dir}")

    ph = pd.concat(dfs, ignore_index=True)

    for c in ["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]:
        ph[c] = pd.to_numeric(ph[c], errors="coerce")

    ph = ph.dropna(subset=["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]).copy()
    ph["SUB_ID"] = ph["SUB_ID"].astype(int)
    ph["SEX"] = ph["SEX"].astype(int)
    ph["DX_GROUP"] = ph["DX_GROUP"].astype(int)

    if "FIQ" in ph.columns:
        ph["FIQ"] = pd.to_numeric(ph["FIQ"], errors="coerce")

    ph["age_group"] = age_group_series(ph["AGE_AT_SCAN"])
    ph = ph.dropna(subset=["age_group"]).copy()

    ph = ph.drop_duplicates(subset=["SUB_ID"], keep="first").copy()
    return ph


def choose_best_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in cols:
            return cols[cand.upper()]
    return None


def infer_age_col(df: pd.DataFrame) -> str | None:
    # common ABIDE2 variants
    candidates = [
        "AGE_AT_SCAN",
        "AGE_AT_SCAN_YRS",
        "AGE_AT_SCAN_YEARS",
        "AGE",
        "AGE_YRS",
        "AGE_YEARS",
    ]
    c = choose_best_col(df, candidates)
    if c:
        return c


    best = None
    best_score = -1.0
    for col in df.columns:
        name = str(col).upper()
        if "AGE" not in name:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        frac = float(s.notna().mean())
        if frac < 0.2:
            continue
        med = float(np.nanmedian(s.to_numpy())) if s.notna().any() else np.nan
        if not np.isfinite(med) or med <= 0 or med > 90:
            continue
        score = frac
        if score > best_score:
            best_score = score
            best = col
    return best


def load_abide2_phenotypes(pheno_dir: Path) -> pd.DataFrame:
    # prefer these two, but tolerate anything
    comp = pheno_dir / "abide2_composite_pheno.csv"
    longi = pheno_dir / "abide2_composite_pheno_longitudinal.csv"

    files = []
    if comp.exists():
        files.append(comp)
    if longi.exists():
        files.append(longi)

    if not files:
        files = sorted(pheno_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"ABIDE2: No phenotype CSVs found in {pheno_dir}")

    dfs = []
    for fp in files:
        df = read_csv_robust(fp)
        df.columns = [c.strip() for c in df.columns]

        sub_col = choose_best_col(df, ["SUB_ID"])
        sex_col = choose_best_col(df, ["SEX"])
        dx_col = choose_best_col(df, ["DX_GROUP"])
        site_col = choose_best_col(df, ["SITE_ID"])
        age_col = infer_age_col(df)

        need = [sub_col, sex_col, dx_col, age_col]
        if any(c is None for c in need):
            continue

        keep = [sub_col, sex_col, dx_col, age_col]
        if site_col:
            keep.append(site_col)

        # optional IQ
        fiq_col = choose_best_col(df, ["FIQ", "FIQ_TOTAL", "FIQ_SCORE"])
        if fiq_col:
            keep.append(fiq_col)

        d = df[keep].copy()
        d = d.rename(columns={
            sub_col: "SUB_ID",
            sex_col: "SEX",
            dx_col: "DX_GROUP",
            age_col: "AGE_AT_SCAN",
        })
        if site_col:
            d = d.rename(columns={site_col: "SITE_ID"})
        if fiq_col:
            d = d.rename(columns={fiq_col: "FIQ"})

        d["DATASET"] = "ABIDE2"
        d["source_file"] = fp.name
        dfs.append(d)

    if not dfs:
        raise RuntimeError(
            f"ABIDE2: Could not find required columns in {pheno_dir}. "
            f"Need at least SUB_ID, SEX, DX_GROUP, and some AGE column."
        )

    ph = pd.concat(dfs, ignore_index=True)

    for c in ["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]:
        ph[c] = pd.to_numeric(ph[c], errors="coerce")

    ph = ph.dropna(subset=["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]).copy()
    ph["SUB_ID"] = ph["SUB_ID"].astype(int)
    ph["SEX"] = ph["SEX"].astype(int)
    ph["DX_GROUP"] = ph["DX_GROUP"].astype(int)

    if "FIQ" in ph.columns:
        ph["FIQ"] = pd.to_numeric(ph["FIQ"], errors="coerce")

    ph["age_group"] = age_group_series(ph["AGE_AT_SCAN"])
    ph = ph.dropna(subset=["age_group"]).copy()

    # de-dupe by SUB_ID (prefer composite over longitudinal simply by file order)
    ph = ph.drop_duplicates(subset=["SUB_ID"], keep="first").copy()
    return ph


def build_abide1_index(conn_dir: Path, meta_dir: Path) -> dict[int, dict]:
    """
    Returns subject_int -> record
    Prefer run-1, else lowest mean_fd, else lowest run.
    """
    rx_sub = re.compile(r"sub-(\d+)")
    rx_run = re.compile(r"run-(\d+)")

    by_sub: dict[int, list[dict]] = {}

    for fp in conn_dir.glob("sub-*_task-rest_run-*.npy"):
        ms = rx_sub.search(fp.name)
        mr = rx_run.search(fp.name)
        if not ms:
            continue
        subj_int = int(ms.group(1))
        run_num = int(mr.group(1)) if mr else 999

        # meta json name matches stem
        meta_fp = meta_dir / (fp.stem + ".json")

        mean_fd = np.nan
        n_kept = np.nan
        n_total = np.nan
        frac_kept = np.nan

        if meta_fp.exists():
            md = load_json_robust(meta_fp)
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

        rec = dict(
            dataset="ABIDE1",
            subject_int=subj_int,
            conn_path=fp,
            run_num=run_num,
            meta_path=meta_fp,
            mean_fd=mean_fd,
            frac_kept=frac_kept,
        )
        by_sub.setdefault(subj_int, []).append(rec)

    chosen: dict[int, dict] = {}
    for subj_int, runs in by_sub.items():
        # 1) run-1
        run1 = [r for r in runs if r["run_num"] == 1]
        if run1:
            chosen[subj_int] = run1[0]
            continue
        # 2) lowest mean_fd
        finite = [r for r in runs if np.isfinite(r["mean_fd"])]
        if finite:
            finite.sort(key=lambda r: r["mean_fd"])
            chosen[subj_int] = finite[0]
            continue
        # 3) lowest run_num
        runs.sort(key=lambda r: r["run_num"])
        chosen[subj_int] = runs[0]

    return chosen


def build_abide2_index(conn_dir: Path, meta_dir: Path) -> dict[int, dict]:
    """
    ABIDE2 subject-level npy files: sub-28743.npy (also some bad sub-50002.npy contamination)
    Meta json: meta_real/sub-28743.json
    """
    rx_sub = re.compile(r"sub-(\d+)")
    by_sub: dict[int, list[dict]] = {}

    for fp in conn_dir.glob("sub-*.npy"):
        ms = rx_sub.search(fp.name)
        if not ms:
            continue
        subj_int = int(ms.group(1))

        meta_fp = meta_dir / f"sub-{subj_int}.json"

        mean_fd = np.nan
        n_kept = np.nan
        n_total = np.nan
        frac_kept = np.nan

        if meta_fp.exists():
            md = load_json_robust(meta_fp)
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

        rec = dict(
            dataset="ABIDE2",
            subject_int=subj_int,
            conn_path=fp,
            run_num=np.nan,
            meta_path=meta_fp,
            mean_fd=mean_fd,
            frac_kept=frac_kept,
        )
        by_sub.setdefault(subj_int, []).append(rec)

    chosen: dict[int, dict] = {}
    for subj_int, recs in by_sub.items():
        finite = [r for r in recs if np.isfinite(r["mean_fd"])]
        if finite:
            finite.sort(key=lambda r: r["mean_fd"])
            chosen[subj_int] = finite[0]
        else:
            chosen[subj_int] = recs[0]

    return chosen


# ----------------------------
# Selection + mean
# ----------------------------
def load_matrix(fp: Path) -> np.ndarray:
    x = np.load(fp)
    x = np.asarray(x, dtype=float)
    if x.shape != (200, 200):
        raise ValueError(f"{fp.name}: expected (200,200), got {x.shape}")
    return x


def build_mean_z(selected: pd.DataFrame) -> np.ndarray:
    sum_mat = np.zeros((200, 200), dtype=np.float64)
    cnt_mat = np.zeros((200, 200), dtype=np.int32)

    for fp in selected["conn_path"]:
        mat = load_matrix(fp)

        # normalize per-file to z-space if needed
        space = detect_space(mat)
        z = r_to_z(mat) if space == "r" else mat

        mask = np.isfinite(z)
        sum_mat[mask] += z[mask]
        cnt_mat[mask] += 1

    mean_z = np.full((200, 200), np.nan, dtype=np.float64)
    valid = cnt_mat > 0
    mean_z[valid] = sum_mat[valid] / cnt_mat[valid]
    return mean_z


def age_sex_match(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Match males to females per (DX_GROUP, age_group) using centroid distance.
    """
    rng = np.random.default_rng(seed)
    selected_rows = []

    for (dx, ag), stratum in df.groupby(["DX_GROUP", "age_group"]):
        females = stratum[stratum["SEX"] == 2].copy()
        males = stratum[stratum["SEX"] == 1].copy()

        nF = len(females)
        nM = len(males)
        if nF == 0 or nM == 0:
            continue

        n = min(nF, nM)

        if nF > n:
            females = females.sample(n=n, random_state=seed)

        cols = ["AGE_AT_SCAN", "frac_kept"]
        if "FIQ" in stratum.columns:
            cols.append("FIQ")

        # fill missing covariates using stratum median; if all missing -> 0
        for c in cols:
            v = pd.to_numeric(stratum[c], errors="coerce").to_numpy(dtype=float)
            med = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
            females[c] = pd.to_numeric(females[c], errors="coerce").fillna(med)
            males[c] = pd.to_numeric(males[c], errors="coerce").fillna(med)

        F = females[cols].to_numpy(dtype=float)
        M = males[cols].to_numpy(dtype=float)

        ZF_cols = []
        ZM_cols = []
        for j in range(len(cols)):
            combined = np.concatenate([F[:, j], M[:, j]])
            z_comb = robust_standardize(combined)
            ZF_cols.append(z_comb[: F.shape[0]])
            ZM_cols.append(z_comb[F.shape[0] :])

        ZF = np.stack(ZF_cols, axis=1)
        ZM = np.stack(ZM_cols, axis=1)

        centroid = np.mean(ZF, axis=0)
        dists = np.sum((ZM - centroid) ** 2, axis=1)

        males = males.assign(match_score=dists)
        males_sel = males.nsmallest(n, "match_score")

        females = females.assign(match_score=np.nan)
        selected_rows.append(females)
        selected_rows.append(males_sel)

        print(f"[STRATUM] DX={int(dx)} age={ag}: females={len(females)} males_selected={len(males_sel)} (males_avail={nM})")

    if not selected_rows:
        raise RuntimeError("No matched strata found (check DX_GROUP/SEX/AGE bins).")

    sel = pd.concat(selected_rows, ignore_index=True).drop_duplicates(subset=["DATASET", "SUB_ID"])
    return sel


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="0.2")
    ap.add_argument("--mode", choices=["all", "agesexmatched"], default="agesexmatched")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=None, help="Optional repo root override")
    args = ap.parse_args()

    root = find_repo_root(Path(args.root) if args.root else Path.cwd())
    fd = args.fd

    # INPUT PATHS (your repo layout)
    abide1_conn = root / "connectomes" / "CC200" / "ABIDE1" / "FDpersubject2"
    abide1_meta = root / "connectomes" / "CC200" / "ABIDE1" / "FDpersubject2_meta"
    abide1_pheno = root / "phenotypes" / "ABIDE1"

    abide2_conn = root / "outputs" / "cc200" / "abide2" / "subjects" / "npy_real"
    abide2_meta = root / "outputs" / "cc200" / "abide2" / "subjects" / "meta_real"
    abide2_pheno = root / "phenotypes" / "ABIDE2"

    out_dir = root / "results" / "group_connectomes" / "ABIDE12_CC200"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] repo_root = {root}")
    print(f"[INFO] mode={args.mode} fd-{fd}")
    print(f"[INFO] ABIDE1 conn={abide1_conn}")
    print(f"[INFO] ABIDE2 conn={abide2_conn}")

    # Phenotypes
    ph1 = load_abide1_phenotypes(abide1_pheno)
    ph2 = load_abide2_phenotypes(abide2_pheno)

    print(f"[INFO] ABIDE1 phenotypes unique subjects: {len(ph1)}")
    print(f"[INFO] ABIDE2 phenotypes unique subjects: {len(ph2)}")

    # Connectome indices
    idx1 = build_abide1_index(abide1_conn, abide1_meta)
    idx2 = build_abide2_index(abide2_conn, abide2_meta)

    print(f"[INFO] ABIDE1 subjects with connectomes (unique): {len(idx1)}")
    print(f"[INFO] ABIDE2 subjects with connectomes (unique): {len(idx2)}")

    # Attach connectomes to phenos (and drop connectomes that don't have phenotypes)
    ph1 = ph1[ph1["SUB_ID"].isin(idx1.keys())].copy()
    ph1["conn_path"] = ph1["SUB_ID"].map(lambda s: idx1[int(s)]["conn_path"])
    ph1["frac_kept"] = ph1["SUB_ID"].map(lambda s: idx1[int(s)]["frac_kept"])
    ph1["run_num"] = ph1["SUB_ID"].map(lambda s: idx1[int(s)]["run_num"])

    ph2 = ph2[ph2["SUB_ID"].isin(idx2.keys())].copy()
    ph2["conn_path"] = ph2["SUB_ID"].map(lambda s: idx2[int(s)]["conn_path"])
    ph2["frac_kept"] = ph2["SUB_ID"].map(lambda s: idx2[int(s)]["frac_kept"])
    ph2["run_num"] = np.nan

    # If ABIDE2 has contamination (e.g. sub-50002.npy), it will be dropped here automatically
    dropped_abide2 = sorted(set(idx2.keys()) - set(ph2["SUB_ID"].tolist()))
    if dropped_abide2:
        print(f"[WARN] Dropped {len(dropped_abide2)} ABIDE2 connectomes with no ABIDE2 phenotype row (first 15): {dropped_abide2[:15]}")

    df = pd.concat([ph1, ph2], ignore_index=True)

    # Pretty labels
    df["sex_name"] = df["SEX"].map(SEX_MAP).fillna(df["SEX"].astype(str))
    df["dx_name"] = df["DX_GROUP"].map(DX_MAP).fillna(df["DX_GROUP"].astype(str))

    print(f"[INFO] ABIDE12 subjects with phenotypes+connectomes: {len(df)}")
    print("[INFO] Counts by dataset:")
    print(df["DATASET"].value_counts().to_string())

    # Selection
    if args.mode == "all":
        sel = df.copy()
        sel["match_score"] = np.nan
        print(f"[INFO] Using ALL subjects (no matching). N={len(sel)}")
    else:
        sel = age_sex_match(df, seed=args.seed)
        print(f"[INFO] Selected total={len(sel)} sex_counts={sel['SEX'].value_counts().to_dict()}")

    # Build mean in Z-space then convert to R
    mean_z = build_mean_z(sel)
    mean_r = np.tanh(mean_z)

    out_z = out_dir / f"OVERALL_{args.mode}_fd-{fd}_mean_z.npy"
    out_r = out_dir / f"OVERALL_{args.mode}_fd-{fd}_mean_r.npy"
    out_csv = out_dir / f"OVERALL_{args.mode}_fd-{fd}_selected_subjects.csv"
    out_ids = out_dir / f"OVERALL_{args.mode}_fd-{fd}_used_subids.txt"

    np.save(out_z, mean_z)
    np.save(out_r, mean_r)
    sel_out = sel.copy()
    sel_out["conn_path"] = sel_out["conn_path"].astype(str)
    sel_out.to_csv(out_csv, index=False)

    # dataset-qualified ids so ABIDE1/2 never collide
    used = [f"{row.DATASET}:{int(row.SUB_ID)}" for row in sel_out[["DATASET", "SUB_ID"]].itertuples(index=False)]
    out_ids.write_text("\n".join(used), encoding="utf-8")

    print("[DONE] Saved:")
    print(f"  {out_z}")
    print(f"  {out_r}")
    print(f"  {out_csv}")
    print(f"  {out_ids}")


if __name__ == "__main__":
    main()