# scripts/random/loo_age_binned_custom_test.py

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# INPUTS
LOO_SIM_PATH = ROOT / r"results\qc\loo_similarity\loo_similarity_per_subject.csv"
PHENO_PATH   = ROOT / r"data\Phenotypic_V1_0b_preprocessed1_clean.csv"  # <-- use your clean pheno if you have it
# If you don't have the clean file, switch to:
# PHENO_PATH = ROOT / r"data\Phenotypic_V1_0b_preprocessed1.csv"

OUT_DIR = ROOT / r"results\qc\loo_similarity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_DIR / "loo_similarity_custom_age_binned_summary.csv"

# AGE BINS (same as your other scripts)
BINS   = [0, 13, 18, 30, 45, 120]
LABELS = ["child", "teen", "young_adult", "adult", "older"]

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AGE_GROUP"] = pd.cut(df["AGE_AT_SCAN"], bins=BINS, labels=LABELS, right=False)
    return df

def cohen_d_ind(a: np.ndarray, b: np.ndarray) -> float:
    # pooled SD Cohen's d for two independent samples
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa2 = np.var(a, ddof=1)
    sb2 = np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    if sp == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / sp

def load_loo_to_wide(path: Path) -> pd.DataFrame:
    """
    Supports:
      (A) long: FILE_ID, subject_group, template_group, similarity_r
      (B) wide: FILE_ID, subject_group, sim_to_F_ASD, sim_to_F_CTL, sim_to_M_ASD, sim_to_M_CTL
    Returns wide with sim_to_* columns.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # normalize FILE_ID col name
    if "FILE_ID" not in df.columns:
        raise ValueError(f"{path} missing FILE_ID column.")

    df["FILE_ID"] = df["FILE_ID"].astype(str).str.strip()

    # Case A: long format
    if "template_group" in df.columns and "similarity_r" in df.columns:
        df["template_group"] = df["template_group"].astype(str).str.strip()
        wide = (
            df.pivot_table(
                index=["FILE_ID", "subject_group"],
                columns="template_group",
                values="similarity_r",
                aggfunc="mean",
            )
            .reset_index()
        )
        # rename template columns to sim_to_*
        for c in list(wide.columns):
            if c not in ["FILE_ID", "subject_group"]:
                wide = wide.rename(columns={c: f"sim_to_{c}"})
        return wide

    # Case B: already wide
    sim_cols = [c for c in df.columns if c.startswith("sim_to_")]
    if not sim_cols:
        raise ValueError(
            f"{path} doesn't look long or wide. "
            f"Expected either (template_group, similarity_r) or sim_to_* columns."
        )
    needed = {"FILE_ID", "subject_group"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} missing required columns: {needed - set(df.columns)}")
    return df

def main():
    if not LOO_SIM_PATH.exists():
        raise FileNotFoundError(f"Missing LOO similarity file: {LOO_SIM_PATH}")
    if not PHENO_PATH.exists():
        raise FileNotFoundError(f"Missing phenotypic file: {PHENO_PATH}")

    loo = load_loo_to_wide(LOO_SIM_PATH)

    # make sure required sim columns exist
    required_sim = ["sim_to_M_CTL", "sim_to_F_CTL"]
    missing = [c for c in required_sim if c not in loo.columns]
    if missing:
        raise ValueError(f"LOO file is missing required columns: {missing}")

    pheno = pd.read_csv(PHENO_PATH)
    pheno.columns = pheno.columns.str.strip()

    # handle the possibility pheno uses SUBJECT_ID or FILE_ID differently
    if "FILE_ID" not in pheno.columns:
        raise ValueError(f"Phenotypic file missing FILE_ID column: {PHENO_PATH}")

    if "AGE_AT_SCAN" not in pheno.columns:
        raise ValueError(f"Phenotypic file missing AGE_AT_SCAN column: {PHENO_PATH}")

    pheno["FILE_ID"] = pheno["FILE_ID"].astype(str).str.strip()
    pheno = pheno[["FILE_ID", "AGE_AT_SCAN"]].drop_duplicates("FILE_ID")
    pheno = add_age_group(pheno)

    df = loo.merge(pheno, on="FILE_ID", how="left")

    # Report missing ages
    missing_age = df["AGE_AT_SCAN"].isna().sum()
    if missing_age > 0:
        print(f"[WARN] {missing_age} subjects in LOO file missing AGE_AT_SCAN after merge.")
    df = df.dropna(subset=["AGE_GROUP"])

    results = []

    age_groups_present = [g for g in LABELS if g in df["AGE_GROUP"].unique().tolist()]
    print("\n=== Age-binned test you asked for ===")
    print("A: F_ASD → sim_to_M_CTL")
    print("B: M_ASD → sim_to_F_CTL\n")

    for ag in age_groups_present:
        sub = df[df["AGE_GROUP"] == ag].copy()

        A = sub.loc[sub["subject_group"] == "F_ASD", "sim_to_M_CTL"].dropna().to_numpy()
        B = sub.loc[sub["subject_group"] == "M_ASD", "sim_to_F_CTL"].dropna().to_numpy()

        nA, nB = len(A), len(B)

        if nA < 2 or nB < 2:
            print(f"[{ag}] Not enough data: nA={nA}, nB={nB}")
            results.append({
                "AGE_GROUP": ag,
                "n_A(F_ASD→M_CTL)": nA,
                "n_B(M_ASD→F_CTL)": nB,
                "mean_A": np.mean(A) if nA else np.nan,
                "mean_B": np.mean(B) if nB else np.nan,
                "mean_diff(A-B)": (np.mean(A) - np.mean(B)) if (nA and nB) else np.nan,
                "sd_A": np.std(A, ddof=1) if nA > 1 else np.nan,
                "sd_B": np.std(B, ddof=1) if nB > 1 else np.nan,
                "cohen_d": np.nan,
                "p_welch_t": np.nan,
                "p_mannwhitney": np.nan
            })
            continue

        meanA, meanB = float(np.mean(A)), float(np.mean(B))
        sdA, sdB = float(np.std(A, ddof=1)), float(np.std(B, ddof=1))
        diff = meanA - meanB
        d = cohen_d_ind(A, B)

        # Welch t-test (unequal variances)
        t_res = stats.ttest_ind(A, B, equal_var=False, nan_policy="omit")
        p_t = float(t_res.pvalue)

        # Mann–Whitney U (two-sided)
        # Use 'method="auto"' for recent scipy; fallback if older.
        try:
            mw_res = stats.mannwhitneyu(A, B, alternative="two-sided", method="auto")
        except TypeError:
            mw_res = stats.mannwhitneyu(A, B, alternative="two-sided")
        p_mw = float(mw_res.pvalue)

        print(f"[{ag}] nA={nA} meanA={meanA:.6f} | nB={nB} meanB={meanB:.6f} | diff={diff:.6f} | d={d:.3f} | p_t={p_t:.4g} p_mw={p_mw:.4g}")

        results.append({
            "AGE_GROUP": ag,
            "n_A(F_ASD→M_CTL)": nA,
            "n_B(M_ASD→F_CTL)": nB,
            "mean_A": meanA,
            "mean_B": meanB,
            "mean_diff(A-B)": diff,
            "sd_A": sdA,
            "sd_B": sdB,
            "cohen_d": d,
            "p_welch_t": p_t,
            "p_mannwhitney": p_mw
        })

    out = pd.DataFrame(results)
    out.to_csv(OUT_SUMMARY, index=False)
    print(f"\nSaved age-binned summary -> {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
