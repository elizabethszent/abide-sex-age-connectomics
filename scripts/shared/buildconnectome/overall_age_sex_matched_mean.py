# scripts/shared/buildconnectome/overall_age_sex_matched_mean_abide2.py

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================
# PATHS (ABIDE2)
# ==========================
ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# ABIDE2 SUBJECT-LEVEL connectomes
CONN_DIR = ROOT / "outputs" / "cc200" / "abide2" / "subjects" / "npy_real"

# ABIDE2 per-subject or per-run meta (optional, but recommended)
META_DIR = ROOT / "outputs" / "cc200" / "abide2" / "subjects" / "meta_real"

# ABIDE2 phenotypes (2 files)
PHENO_DIR = ROOT / "phenotypes" / "ABIDE2"
PHENO_COMPOSITE = PHENO_DIR / "abide2_composite_pheno.csv"
PHENO_LONG      = PHENO_DIR / "abide2_composite_pheno_longitudinal.csv"

OUT_DIR = ROOT / "results" / "group_connectomes" / "ABIDE2_CC200"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FD_LIST = ["0.2"]  # just a label for outputs (your connectomes are already QC’d at meanFD<=0.2)

# Age bins
BINS   = [0, 10, 13, 18, 200]
LABELS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]

SEX_MAP = {1: "Male", 2: "Female"}
DX_MAP  = {1: "ASD",  2: "Control"}

# ==========================
# Helpers
# ==========================
def clean_colname(c: str) -> str:
    # replace non-breaking spaces and weird whitespace
    c = str(c).replace("\xa0", " ").strip()
    c = re.sub(r"\s+", " ", c)
    return c.upper()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]
    return df

def read_csv_robust(fp: Path) -> pd.DataFrame:
    # ABIDE2 pheno sometimes isn’t pure utf-8
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(fp, encoding="utf-8", encoding_errors="replace")

def pick_col(df: pd.DataFrame, candidates: list[str], startswith_ok: bool = False) -> str | None:
    cols = list(df.columns)
    cand_up = [clean_colname(c) for c in candidates]

    for c in cand_up:
        if c in cols:
            return c

    if startswith_ok:
        for c in cols:
            for pref in cand_up:
                if c.startswith(pref):
                    return c
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
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = mad if (np.isfinite(mad) and mad > 1e-12) else (np.nanstd(x) + 1e-12)
    return (x - med) / scale

def load_json(fp: Path) -> dict:
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(fp.read_text(encoding="latin1"))

def load_matrix(fp: Path) -> np.ndarray:
    x = np.load(fp)
    x = np.asarray(x)
    if x.shape != (200, 200):
        raise ValueError(f"{fp.name}: expected (200,200), got {x.shape}")
    return x

def detect_input_space(example_mat: np.ndarray) -> str:
    finite = example_mat[np.isfinite(example_mat)]
    if finite.size == 0:
        return "z"
    frac_in_unit = np.mean((finite >= -1.0001) & (finite <= 1.0001))
    return "r" if frac_in_unit > 0.98 else "z"

def r_to_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def z_to_r(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

# ==========================
# Connectome indexing (ABIDE2)
# ==========================
def parse_subject_and_run(p: Path) -> tuple[int, int | None]:
    """
    Extract subject int and run int (if present) from filename.
    Supports:
      sub-28743.npy
      sub-28743_task-rest_run-1.npy
      sub-28743_ses-1_task-rest_run-2.npy
    """
    ms = re.search(r"sub-(\d+)", p.name)
    subj = int(ms.group(1)) if ms else None
    mr = re.search(r"run-(\d+)", p.name)
    run = int(mr.group(1)) if mr else None
    if subj is None:
        raise ValueError(f"Could not parse subject from: {p.name}")
    return subj, run

def meta_for_connectome(fp: Path) -> Path:
    # subject-level meta is usually same stem
    return META_DIR / f"{fp.stem}.json"

def build_connectome_index() -> dict[int, dict]:
    """
    Map subject_int -> chosen record (pref run-1, else lowest mean_fd, else lowest run number).
    Also reads frac_kept if present in meta.
    """
    if not CONN_DIR.exists():
        raise FileNotFoundError(f"CONN_DIR not found: {CONN_DIR}")

    by_sub: dict[int, list[dict]] = {}
    files = list(CONN_DIR.glob("*.npy"))

    for fp in files:
        try:
            subj_int, run = parse_subject_and_run(fp)
        except Exception:
            continue

        mp = meta_for_connectome(fp)

        mean_fd = np.nan
        n_kept = np.nan
        n_total = np.nan
        frac_kept = np.nan

        if mp.exists():
            md = load_json(mp)
            # tolerate common key variants
            for k in ["mean_fd", "meanFD", "MEAN_FD", "FUNC_MEAN_FD", "func_mean_fd"]:
                if k in md:
                    try:
                        mean_fd = float(md[k])
                    except Exception:
                        pass
                    break
            for k in ["n_kept", "N_KEPT"]:
                if k in md:
                    try:
                        n_kept = float(md[k])
                    except Exception:
                        pass
                    break
            for k in ["n_total", "N_TOTAL"]:
                if k in md:
                    try:
                        n_total = float(md[k])
                    except Exception:
                        pass
                    break
            if np.isfinite(n_kept) and np.isfinite(n_total) and n_total != 0:
                frac_kept = n_kept / n_total

        rec = {
            "conn_path": fp,
            "meta_path": mp,
            "subject_int": subj_int,
            "run_num": run if run is not None else 999,
            "mean_fd": mean_fd,
            "n_kept": n_kept,
            "n_total": n_total,
            "frac_kept": frac_kept,
        }
        by_sub.setdefault(subj_int, []).append(rec)

    chosen: dict[int, dict] = {}
    for subj_int, runs in by_sub.items():
        # 1) prefer run-1 if present
        run1 = [r for r in runs if r["run_num"] == 1]
        if run1:
            chosen[subj_int] = run1[0]
            continue

        # 2) prefer lowest mean_fd if available
        finite_fd = [r for r in runs if np.isfinite(r["mean_fd"])]
        if finite_fd:
            finite_fd.sort(key=lambda r: r["mean_fd"])
            chosen[subj_int] = finite_fd[0]
            continue

        # 3) otherwise lowest run number
        runs.sort(key=lambda r: r["run_num"])
        chosen[subj_int] = runs[0]

    return chosen

# ==========================
# Phenotype loading (ABIDE2: composite + longitudinal)
# ==========================
def extract_pheno_minimal(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = normalize_columns(df)

    col_sub  = pick_col(df, ["SUB_ID", "SUBID", "SUBJECT_ID"], startswith_ok=True)
    col_sex  = pick_col(df, ["SEX"], startswith_ok=True)
    col_dx   = pick_col(df, ["DX_GROUP", "DX"], startswith_ok=True)
    col_age  = pick_col(df, ["AGE_AT_SCAN", "AGE"], startswith_ok=True)
    col_site = pick_col(df, ["SITE_ID", "SITE"], startswith_ok=True)
    col_fiq  = pick_col(df, ["FIQ", "FIQ_TOTAL"], startswith_ok=True)

    need_missing = [("SUB_ID", col_sub), ("SEX", col_sex), ("DX_GROUP", col_dx), ("AGE_AT_SCAN", col_age)]
    missing = [name for name, col in need_missing if col is None]
    if missing:
        raise RuntimeError(
            f"{source_name}: missing required columns {missing}. "
            f"(I found columns like: {list(df.columns)[:25]} ...)"
        )

    keep_map = {
        "SUB_ID": col_sub,
        "SEX": col_sex,
        "DX_GROUP": col_dx,
        "AGE_AT_SCAN": col_age,
    }
    if col_site is not None:
        keep_map["SITE_ID"] = col_site
    if col_fiq is not None:
        keep_map["FIQ"] = col_fiq

    out = df[list(keep_map.values())].copy()
    out.columns = list(keep_map.keys())
    out["source_file"] = source_name

    # numeric cleanup
    out["SUB_ID"] = pd.to_numeric(out["SUB_ID"], errors="coerce")
    out["SEX"] = pd.to_numeric(out["SEX"], errors="coerce")
    out["DX_GROUP"] = pd.to_numeric(out["DX_GROUP"], errors="coerce")
    out["AGE_AT_SCAN"] = pd.to_numeric(out["AGE_AT_SCAN"], errors="coerce")
    if "FIQ" in out.columns:
        out["FIQ"] = pd.to_numeric(out["FIQ"], errors="coerce")

    out = out.dropna(subset=["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]).copy()
    out["SUB_ID"] = out["SUB_ID"].astype(int)
    out["SEX"] = out["SEX"].astype(int)
    out["DX_GROUP"] = out["DX_GROUP"].astype(int)

    out["age_group"] = out["AGE_AT_SCAN"].map(age_group)
    out = out.dropna(subset=["age_group"]).copy()

    return out

def dedupe_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a file has multiple rows per SUB_ID (longitudinal), collapse to 1 row per subject:
    take first non-null value per column.
    """
    def first_nonnull(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    cols = [c for c in df.columns if c not in ["source_file"]]
    agg = {c: first_nonnull for c in cols}
    out = df.groupby("SUB_ID", as_index=False).agg(agg)
    return out

def load_abide2_phenotypes() -> pd.DataFrame:
    if not PHENO_COMPOSITE.exists():
        raise FileNotFoundError(f"Missing: {PHENO_COMPOSITE}")
    if not PHENO_LONG.exists():
        raise FileNotFoundError(f"Missing: {PHENO_LONG}")

    comp = extract_pheno_minimal(read_csv_robust(PHENO_COMPOSITE), PHENO_COMPOSITE.name)
    long = extract_pheno_minimal(read_csv_robust(PHENO_LONG), PHENO_LONG.name)

    comp = dedupe_subjects(comp)
    long = dedupe_subjects(long)

    # merge: prefer composite, fill from longitudinal
    merged = comp.merge(long, on="SUB_ID", how="outer", suffixes=("_comp", "_long"))

    def pick_field(base: str):
        a = merged.get(f"{base}_comp")
        b = merged.get(f"{base}_long")
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return a.combine_first(b)

    out = pd.DataFrame({
        "SUB_ID": merged["SUB_ID"].astype(int),
        "SEX": pick_field("SEX"),
        "DX_GROUP": pick_field("DX_GROUP"),
        "AGE_AT_SCAN": pick_field("AGE_AT_SCAN"),
        "SITE_ID": pick_field("SITE_ID") if "SITE_ID_comp" in merged.columns or "SITE_ID_long" in merged.columns else None,
        "FIQ": pick_field("FIQ") if "FIQ_comp" in merged.columns or "FIQ_long" in merged.columns else None,
    })

    # cleanup after combine_first
    out["SEX"] = pd.to_numeric(out["SEX"], errors="coerce")
    out["DX_GROUP"] = pd.to_numeric(out["DX_GROUP"], errors="coerce")
    out["AGE_AT_SCAN"] = pd.to_numeric(out["AGE_AT_SCAN"], errors="coerce")
    if "FIQ" in out.columns and out["FIQ"] is not None:
        out["FIQ"] = pd.to_numeric(out["FIQ"], errors="coerce")

    out = out.dropna(subset=["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]).copy()
    out["SUB_ID"] = out["SUB_ID"].astype(int)
    out["SEX"] = out["SEX"].astype(int)
    out["DX_GROUP"] = out["DX_GROUP"].astype(int)

    out["age_group"] = out["AGE_AT_SCAN"].map(age_group)
    out = out.dropna(subset=["age_group"]).copy()

    # one row per subject
    out = out.drop_duplicates(subset=["SUB_ID"], keep="first").copy()
    return out

# ==========================
# Main
# ==========================
def main():
    pheno = load_abide2_phenotypes()
    print(f"[INFO] ABIDE2 phenotype subjects loaded (unique): {len(pheno)}")

    conn_idx = build_connectome_index()
    print(f"[INFO] Unique subjects with at least one connectome: {len(conn_idx)}")

    # attach chosen connectome and frac_kept
    has_conn = pheno["SUB_ID"].isin(conn_idx.keys())
    df = pheno[has_conn].copy()

    df["conn_path"] = df["SUB_ID"].map(lambda s: conn_idx[int(s)]["conn_path"])
    df["frac_kept"] = df["SUB_ID"].map(lambda s: conn_idx[int(s)]["frac_kept"])
    df["run_num"]   = df["SUB_ID"].map(lambda s: conn_idx[int(s)]["run_num"])
    df["mean_fd"]   = df["SUB_ID"].map(lambda s: conn_idx[int(s)]["mean_fd"])

    print(f"[INFO] Phenotype subjects with connectomes available: {len(df)}")
    print("[INFO] Chosen run breakdown (subjects):")
    print(df["run_num"].value_counts().sort_index().to_string())

    # warn if connectomes contain subjects not in phenotype
    ph_ids = set(pheno["SUB_ID"].tolist())
    conn_ids = set(conn_idx.keys())
    missing_pheno = sorted(list(conn_ids - ph_ids))
    if missing_pheno:
        print(f"[WARN] {len(missing_pheno)} connectome subjects missing phenotypes (first 15): {missing_pheno[:15]}")

    if df.empty:
        raise RuntimeError("No ABIDE2 subjects matched between phenotypes and connectomes.")

    # decide r vs z based on the first file
    example_mat = load_matrix(df["conn_path"].iloc[0])
    space = detect_input_space(example_mat)
    print(f"[INFO] Detected input space: {space} (r=correlation, z=fisher-z)")

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

            # if females > males (rare), downsample females
            if nF > n:
                females = females.sample(n=n, random_state=42)

            cols = ["AGE_AT_SCAN", "frac_kept"]
            if "FIQ" in females.columns and females["FIQ"].notna().any():
                cols.append("FIQ")

            # fill missing covariates with stratum medians
            for c in cols:
                med = np.nanmedian(stratum[c].to_numpy(dtype=float))
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
            raise RuntimeError("No matched subjects found (check phenotypes/connectomes).")

        sel = pd.concat(selected_rows, ignore_index=True)
        sel = sel.drop_duplicates(subset=["SUB_ID"])  # safety

        sex_counts = sel["SEX"].value_counts().to_dict()
        print(f"[INFO] Selected total={len(sel)} sex_counts={sex_counts}")

        # streaming nan-mean
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

        # add a friendlier sub-XXXX id column
        sel_out = sel.copy()
        sel_out["sub_id_str"] = sel_out["SUB_ID"].map(lambda x: f"sub-{int(x)}")
        sel_out.to_csv(out_csv, index=False)

        out_ids.write_text("\n".join(sel_out["sub_id_str"].tolist()), encoding="utf-8")

        print(f"[DONE] Saved:")
        print(f"  {out_z}")
        print(f"  {out_r}")
        print(f"  {out_csv}")
        print(f"  {out_ids}")

if __name__ == "__main__":
    main()