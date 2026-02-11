import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.stats import ttest_rel, wilcoxon, ttest_ind, mannwhitneyu


# =========================
# CONFIG
# =========================
ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

CONN_DIR = ROOT / r"data\connectomes\cpac\nofilt_noglobal\cc200_z"

# Use your cleaned phenotypic CSV if you have it
PHENO_CSV = ROOT / r"data\Phenotypic_V1_0b_preprocessed1_clean.csv"
# If not:
# PHENO_CSV = ROOT / r"data\Phenotypic_V1_0b_preprocessed1.csv"

OUT_DIR = ROOT / r"results\qc\loo_similarity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 200

# ABIDE coding
ASD_CODE = 1
CTL_CODE = 2
MALE_CODE = 1
FEMALE_CODE = 2

# Correlation masking thresholds
MIN_FINITE_FRAC = 0.80   # require at least 80% finite edges
MIN_FINITE_EDGES = 5000  # absolute guardrail


# =========================
# HELPERS
# =========================
def load_connectome(fid: str) -> Optional[np.ndarray]:
    fp = CONN_DIR / f"{fid}.npy"
    if not fp.exists():
        return None
    mat = np.load(fp)
    if mat.shape != (EXPECTED_N, EXPECTED_N):
        return None
    return mat


def vec_upper(mat: np.ndarray) -> np.ndarray:
    """Vectorize upper triangle (excluding diagonal)."""
    iu = np.triu_indices(mat.shape[0], k=1)
    v = mat[iu].astype(np.float64)
    return v


def safe_corr_masked(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson corr on finite intersection only.
    Returns NaN if too few valid edges or zero variance.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size == 0:
        return np.nan

    m = np.isfinite(a) & np.isfinite(b)
    n_ok = int(m.sum())
    if n_ok < MIN_FINITE_EDGES or (n_ok / a.size) < MIN_FINITE_FRAC:
        return np.nan

    aa = a[m]
    bb = b[m]

    sa = aa.std()
    sb = bb.std()
    if not np.isfinite(sa) or not np.isfinite(sb) or sa == 0 or sb == 0:
        return np.nan

    return float(np.corrcoef(aa, bb)[0, 1])


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["FILE_ID"] = df["FILE_ID"].astype(str).str.strip()
    return df


def group_label(sex_code: int, dx_code: int) -> Optional[str]:
    if sex_code == FEMALE_CODE and dx_code == ASD_CODE:
        return "F_ASD"
    if sex_code == FEMALE_CODE and dx_code == CTL_CODE:
        return "F_CTL"
    if sex_code == MALE_CODE and dx_code == ASD_CODE:
        return "M_ASD"
    if sex_code == MALE_CODE and dx_code == CTL_CODE:
        return "M_CTL"
    return None


def add_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match your standard bins used elsewhere.
    right=False means [0,13) is child, [13,18) teen, etc.
    """
    bins = [0, 13, 18, 30, 45, 120]
    labels = ["child", "teen", "young_adult", "adult", "older"]
    df = df.copy()
    if "AGE_AT_SCAN" in df.columns:
        df["AGE_GROUP"] = pd.cut(df["AGE_AT_SCAN"], bins=bins, labels=labels, right=False)
    else:
        df["AGE_GROUP"] = np.nan
    return df


def summarize_pairwise(per_subject: pd.DataFrame,
                       comparisons: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """
    Within-subject comparisons (paired): col_a vs col_b.
    """
    rows = []
    for name, col_a, col_b in comparisons:
        if col_a not in per_subject.columns or col_b not in per_subject.columns:
            rows.append({
                "comparison": name, "n": 0,
                "mean_a": np.nan, "mean_b": np.nan,
                "mean_diff(a-b)": np.nan,
                "p_ttest": np.nan, "p_wilcoxon": np.nan,
            })
            continue

        sub = per_subject[[col_a, col_b]].dropna()
        a = sub[col_a].to_numpy()
        b = sub[col_b].to_numpy()
        n = int(len(sub))

        if n < 5:
            rows.append({
                "comparison": name, "n": n,
                "mean_a": float(np.mean(a)) if n else np.nan,
                "mean_b": float(np.mean(b)) if n else np.nan,
                "mean_diff(a-b)": float(np.mean(a - b)) if n else np.nan,
                "p_ttest": np.nan, "p_wilcoxon": np.nan,
            })
            continue

        try:
            p_t = float(ttest_rel(a, b, nan_policy="omit").pvalue)
        except Exception:
            p_t = np.nan

        try:
            diffs = a - b
            if np.allclose(diffs, 0):
                p_w = 1.0
            else:
                p_w = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
        except Exception:
            p_w = np.nan

        rows.append({
            "comparison": name, "n": n,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff(a-b)": float(np.mean(a - b)),
            "p_ttest": p_t,
            "p_wilcoxon": p_w,
        })

    return pd.DataFrame(rows)


def welch_and_mw(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Between-group tests:
      - Welch t-test (unequal variance)
      - Mann–Whitney U (two-sided)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan)

    try:
        p_t = float(ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)
    except Exception:
        p_t = np.nan

    try:
        p_mw = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        p_mw = np.nan

    return (p_t, p_mw)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa = np.var(a, ddof=1)
    sb = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a)-1)*sa + (len(b)-1)*sb) / (len(a)+len(b)-2))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


# =========================
# MAIN
# =========================
def main():
    if not PHENO_CSV.exists():
        raise FileNotFoundError(f"Missing PHENO_CSV: {PHENO_CSV}")
    if not CONN_DIR.exists():
        raise FileNotFoundError(f"Missing CONN_DIR: {CONN_DIR}")

    ph = pd.read_csv(PHENO_CSV)
    ph = standardize_columns(ph)
    ph = add_age_group_column(ph)

    required = {"FILE_ID", "SEX", "DX_GROUP"}
    missing = required - set(ph.columns)
    if missing:
        raise ValueError(f"Phenotypic CSV missing columns: {missing}")

    ph["group"] = [
        group_label(int(s), int(d))
        for s, d in zip(ph["SEX"].astype(int), ph["DX_GROUP"].astype(int))
    ]

    # drop placeholders
    ph = ph[ph["FILE_ID"].notna()].copy()
    ph = ph[ph["FILE_ID"].astype(str).str.strip() != ""].copy()
    ph = ph[ph["FILE_ID"] != "no_filename"].copy()
    ph = ph[ph["group"].notna()].copy()

    # load connectomes & vectors
    groups: Dict[str, List[str]] = {"F_ASD": [], "F_CTL": [], "M_ASD": [], "M_CTL": []}
    edges: Dict[str, np.ndarray] = {}
    fid_to_group: Dict[str, str] = {}
    fid_to_age_group: Dict[str, str] = {}

    for _, r in ph.iterrows():
        fid = str(r["FILE_ID"]).strip()
        g = r["group"]
        mat = load_connectome(fid)
        if mat is None:
            continue
        v = vec_upper(mat)
        edges[fid] = v
        groups[g].append(fid)
        fid_to_group[fid] = g
        fid_to_age_group[fid] = str(r.get("AGE_GROUP", ""))

    print(f"Loaded usable subjects: {len(edges)}")
    print("Counts per group used:")
    for g in ["F_ASD", "F_CTL", "M_ASD", "M_CTL"]:
        print(f"  {g}: n={len(groups[g])}")

    # QC: non-finite edges
    nonfinite_counts = {fid: int((~np.isfinite(v)).sum()) for fid, v in edges.items()}
    n_with_any = sum(1 for _, c in nonfinite_counts.items() if c > 0)
    if n_with_any > 0:
        worst = sorted(nonfinite_counts.items(), key=lambda x: -x[1])[:10]
        print(f"[QC] Subjects with any non-finite edges: {n_with_any}/{len(edges)}")
        print("[QC] Worst 10 non-finite edge counts:")
        for fid, c in worst:
            if c == 0:
                break
            print(f"  {fid}: {c}")
    else:
        print("[QC] No non-finite edges found in any subject vectors.")

    # build group template in edge-space with nanmean
    def group_template(group_fids: List[str], exclude: Optional[str] = None) -> Optional[np.ndarray]:
        fids = group_fids if exclude is None else [x for x in group_fids if x != exclude]
        if len(fids) < 2:
            return None
        X = np.stack([edges[x] for x in fids], axis=0)  # (n, E)
        templ = np.nanmean(X, axis=0)
        return templ

    # compute similarities (long)
    records = []
    for fid, subj_vec in edges.items():
        subj_group = fid_to_group.get(fid, None)
        if subj_group is None:
            continue

        for tgt in ["F_ASD", "F_CTL", "M_ASD", "M_CTL"]:
            exclude = fid if tgt == subj_group else None
            templ = group_template(groups[tgt], exclude=exclude)
            sim = safe_corr_masked(subj_vec, templ) if templ is not None else np.nan

            records.append({
                "FILE_ID": fid,
                "subject_group": subj_group,
                "AGE_GROUP": fid_to_age_group.get(fid, ""),
                "template_group": tgt,
                "similarity_r": sim,
            })

    long_df = pd.DataFrame(records)
    out_per_subject = OUT_DIR / "loo_similarity_per_subject.csv"
    long_df.to_csv(out_per_subject, index=False)
    print(f"Saved per-subject similarities -> {out_per_subject}")

    # WIDE table
    wide = (
        long_df.pivot_table(
            index=["FILE_ID", "subject_group", "AGE_GROUP"],
            columns="template_group",
            values="similarity_r",
            aggfunc="mean",
        )
        .rename(columns={g: f"sim_to_{g}" for g in ["F_ASD", "F_CTL", "M_ASD", "M_CTL"]})
        .reset_index()
    )

    usable_rows = wide.dropna(subset=["sim_to_F_ASD", "sim_to_F_CTL", "sim_to_M_ASD", "sim_to_M_CTL"], how="all")
    print(f"[QC] Wide rows with at least one non-NaN similarity: {len(usable_rows)}/{len(wide)}")

    # -------------------------
    # (A) Your original within-subject “sex-template preference” checks
    # -------------------------
    comparisons = [
        ("sim_to_M_CTL vs sim_to_F_CTL", "sim_to_M_CTL", "sim_to_F_CTL"),
        ("sim_to_F_CTL vs sim_to_M_CTL", "sim_to_F_CTL", "sim_to_M_CTL"),
    ]

    summary_rows = []
    for sg in ["F_ASD", "F_CTL", "M_ASD", "M_CTL"]:
        sub_w = wide[wide["subject_group"] == sg].copy()
        summ = summarize_pairwise(sub_w, comparisons)
        summ.insert(0, "subject_group", sg)
        summary_rows.append(summ)

    summary = pd.concat(summary_rows, ignore_index=True)

    out_summary = OUT_DIR / "loo_similarity_pairwise_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Saved pairwise test summary -> {out_summary}")

    print("\n=== Pairwise within-subject checks (higher corr = more similar) ===")
    for _, rr in summary.iterrows():
        md = rr["mean_diff(a-b)"]
        print(
            f"- [{rr['subject_group']}] {rr['comparison']} | n={int(rr['n'])} | "
            f"mean_a={rr['mean_a']:.3f} mean_b={rr['mean_b']:.3f} mean_diff={md:.3f} | "
            f"p_t={rr['p_ttest']:.3g} p_w={rr['p_wilcoxon']:.3g}"
        )

    # -------------------------
    # (B) The test you actually wanted:
    #     F_ASD → M_CTL  versus  M_ASD → F_CTL
    # -------------------------
    A = wide.loc[wide["subject_group"] == "F_ASD", "sim_to_M_CTL"].to_numpy()
    B = wide.loc[wide["subject_group"] == "M_ASD", "sim_to_F_CTL"].to_numpy()
    A = A[np.isfinite(A)]
    B = B[np.isfinite(B)]

    p_t, p_mw = welch_and_mw(A, B)
    d = cohens_d(A, B)
    mean_diff = float(np.mean(A) - np.mean(B)) if len(A) and len(B) else np.nan

    custom = pd.DataFrame([{
        "test": "F_ASD→M_CTL vs M_ASD→F_CTL",
        "n_A": int(len(A)),
        "mean_A": float(np.mean(A)) if len(A) else np.nan,
        "sd_A": float(np.std(A, ddof=1)) if len(A) > 1 else np.nan,
        "n_B": int(len(B)),
        "mean_B": float(np.mean(B)) if len(B) else np.nan,
        "sd_B": float(np.std(B, ddof=1)) if len(B) > 1 else np.nan,
        "mean_diff(A-B)": mean_diff,
        "cohen_d": d,
        "p_welch_t": p_t,
        "p_mannwhitney": p_mw,
    }])

    out_custom = OUT_DIR / "loo_similarity_custom_comparisons.csv"
    custom.to_csv(out_custom, index=False)
    print(f"\nSaved custom comparison -> {out_custom}")

    print("\n=== CUSTOM TEST (between groups) ===")
    print("A = F_ASD subjects: similarity to M_CTL template (F_ASD→M_CTL)")
    print("B = M_ASD subjects: similarity to F_CTL template (M_ASD→F_CTL)")
    print(f"n_A={len(A)} mean_A={np.mean(A):.6f} sd_A={np.std(A, ddof=1):.6f}" if len(A) else "n_A=0")
    print(f"n_B={len(B)} mean_B={np.mean(B):.6f} sd_B={np.std(B, ddof=1):.6f}" if len(B) else "n_B=0")
    print(f"mean_diff(A-B)={mean_diff:.6f} Cohen_d={d:.3f} p_welch={p_t} p_mw={p_mw}")

    # -------------------------
    # (C) Age-binned version of that custom test
    # -------------------------
    age_rows = []
    for ag in ["child", "teen", "young_adult", "adult", "older"]:
        A_ag = wide.loc[(wide["subject_group"] == "F_ASD") & (wide["AGE_GROUP"] == ag), "sim_to_M_CTL"].to_numpy()
        B_ag = wide.loc[(wide["subject_group"] == "M_ASD") & (wide["AGE_GROUP"] == ag), "sim_to_F_CTL"].to_numpy()
        A_ag = A_ag[np.isfinite(A_ag)]
        B_ag = B_ag[np.isfinite(B_ag)]

        if len(A_ag) < 5 or len(B_ag) < 5:
            age_rows.append({
                "AGE_GROUP": ag,
                "n_A": int(len(A_ag)),
                "mean_A": float(np.mean(A_ag)) if len(A_ag) else np.nan,
                "n_B": int(len(B_ag)),
                "mean_B": float(np.mean(B_ag)) if len(B_ag) else np.nan,
                "mean_diff(A-B)": float(np.mean(A_ag) - np.mean(B_ag)) if len(A_ag) and len(B_ag) else np.nan,
                "cohen_d": cohens_d(A_ag, B_ag),
                "p_welch_t": np.nan,
                "p_mannwhitney": np.nan,
                "note": "Not enough data (need >=5 each)"
            })
            continue

        p_t_ag, p_mw_ag = welch_and_mw(A_ag, B_ag)
        d_ag = cohens_d(A_ag, B_ag)
        age_rows.append({
            "AGE_GROUP": ag,
            "n_A": int(len(A_ag)),
            "mean_A": float(np.mean(A_ag)),
            "n_B": int(len(B_ag)),
            "mean_B": float(np.mean(B_ag)),
            "mean_diff(A-B)": float(np.mean(A_ag) - np.mean(B_ag)),
            "cohen_d": d_ag,
            "p_welch_t": p_t_ag,
            "p_mannwhitney": p_mw_ag,
            "note": ""
        })

    by_age = pd.DataFrame(age_rows)
    out_by_age = OUT_DIR / "loo_similarity_custom_comparisons_by_age.csv"
    by_age.to_csv(out_by_age, index=False)
    print(f"\nSaved custom comparison by age -> {out_by_age}")

    print("\n=== CUSTOM TEST BY AGE (A=F_ASD→M_CTL vs B=M_ASD→F_CTL) ===")
    for _, r in by_age.iterrows():
        ag = r["AGE_GROUP"]
        if r["note"]:
            print(f"[{ag}] nA={r['n_A']} nB={r['n_B']}  -> {r['note']}")
        else:
            print(
                f"[{ag}] nA={r['n_A']} meanA={r['mean_A']:.6f} | "
                f"nB={r['n_B']} meanB={r['mean_B']:.6f} | "
                f"diff={r['mean_diff(A-B)']:.6f} d={r['cohen_d']:.3f} | "
                f"p_welch={r['p_welch_t']:.3g} p_mw={r['p_mannwhitney']:.3g}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
