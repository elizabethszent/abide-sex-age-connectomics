import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

# Optional stats tests
from scipy.stats import ttest_ind, mannwhitneyu

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# This is the file your LOO script already saved
PER_SUBJECT_CSV = ROOT / r"results\qc\loo_similarity\loo_similarity_per_subject.csv"

# Output summary file (optional)
OUT_SUMMARY_CSV = ROOT / r"results\qc\loo_similarity\loo_similarity_custom_comparisons.csv"

# If present, we can stratify by age
AGE_BINS   = [0, 13, 18, 30, 45, 120]
AGE_LABELS = ["child", "teen", "young_adult", "adult", "older"]


def cohen_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa = np.nanstd(a, ddof=1)
    sb = np.nanstd(b, ddof=1)
    sp = sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / (na + nb - 2))
    if sp == 0:
        return np.nan
    return (np.nanmean(a) - np.nanmean(b)) / sp


def load_wide(per_subject_csv: Path) -> pd.DataFrame:
    """
    Expect per-subject file to contain at least:
      FILE_ID, subject_group, template_group, similarity_r
    And possibly AGE_AT_SCAN, SEX, DX_GROUP, etc (either repeated or not).
    We'll pivot to one row per subject with columns:
      sim_to_F_ASD, sim_to_F_CTL, sim_to_M_ASD, sim_to_M_CTL
    """
    df = pd.read_csv(per_subject_csv)
    df.columns = [c.strip() for c in df.columns]

    required = {"FILE_ID", "subject_group", "template_group", "similarity_r"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{per_subject_csv} missing columns: {missing}")

    # Clean whitespace
    for c in ["FILE_ID", "subject_group", "template_group"]:
        df[c] = df[c].astype(str).str.strip()

    # similarity_r must be numeric
    df["similarity_r"] = pd.to_numeric(df["similarity_r"], errors="coerce")

    # Keep any metadata columns that exist (same per subject). We'll take first after pivot.
    meta_cols = [c for c in df.columns if c not in ["template_group", "similarity_r"]]

    wide = df.pivot_table(
        index=meta_cols,
        columns="template_group",
        values="similarity_r",
        aggfunc="mean",
    ).reset_index()

    # Rename template columns to sim_to_*
    template_cols = [c for c in wide.columns if c not in meta_cols]
    wide = wide.rename(columns={c: f"sim_to_{c}" for c in template_cols})

    return wide


def run_two_sample(a: np.ndarray, b: np.ndarray):
    """Return a dict with summary + tests for A vs B."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    out = {
        "n_A": len(a),
        "n_B": len(b),
        "mean_A": np.mean(a) if len(a) else np.nan,
        "mean_B": np.mean(b) if len(b) else np.nan,
        "sd_A": np.std(a, ddof=1) if len(a) > 1 else np.nan,
        "sd_B": np.std(b, ddof=1) if len(b) > 1 else np.nan,
        "mean_diff_A_minus_B": (np.mean(a) - np.mean(b)) if (len(a) and len(b)) else np.nan,
        "cohen_d": cohen_d_independent(a, b),
        "p_ttest": np.nan,
        "p_mannwhitney": np.nan,
    }

    if len(a) >= 2 and len(b) >= 2:
        # Welch t-test
        out["p_ttest"] = float(ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)

        # Mann–Whitney (two-sided)
        out["p_mannwhitney"] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)

    return out


def main():
    if not PER_SUBJECT_CSV.exists():
        raise FileNotFoundError(f"Missing per-subject similarities: {PER_SUBJECT_CSV}")

    wide = load_wide(PER_SUBJECT_CSV)

    # Ensure the columns we need exist
    need_cols = {"FILE_ID", "subject_group", "sim_to_M_CTL", "sim_to_F_CTL"}
    missing = need_cols - set(wide.columns)
    if missing:
        raise ValueError(
            f"Wide table missing columns {missing}.\n"
            f"Available columns: {list(wide.columns)}"
        )

    # ===== Your specific hypothesis test =====
    # A: F_ASD subjects' similarity to M_CTL template
    A = wide.loc[wide["subject_group"] == "F_ASD", "sim_to_M_CTL"].to_numpy()

    # B: M_ASD subjects' similarity to F_CTL template
    B = wide.loc[wide["subject_group"] == "M_ASD", "sim_to_F_CTL"].to_numpy()

    res = run_two_sample(A, B)

    print("\n=== TEST YOU ASKED FOR ===")
    print("Compare:")
    print("  A = F_ASD subjects: similarity to M_CTL template  (F_ASD → M_CTL)")
    print("  B = M_ASD subjects: similarity to F_CTL template  (M_ASD → F_CTL)")
    print("")
    print(f"n_A={res['n_A']}  mean_A={res['mean_A']:.6f}  sd_A={res['sd_A']:.6f}")
    print(f"n_B={res['n_B']}  mean_B={res['mean_B']:.6f}  sd_B={res['sd_B']:.6f}")
    print(f"mean_diff(A-B)={res['mean_diff_A_minus_B']:.6f}  Cohen_d={res['cohen_d']:.3f}")
    print(f"Welch t-test p={res['p_ttest']}")
    print(f"Mann–Whitney p={res['p_mannwhitney']}")

    rows_out = [{
        "comparison": "F_ASD→M_CTL  vs  M_ASD→F_CTL",
        **res
    }]

    # ===== Optional: do this by age bin if AGE_AT_SCAN exists =====
    if "AGE_AT_SCAN" in wide.columns:
        wide2 = wide.copy()
        wide2["AGE_GROUP"] = pd.cut(
            pd.to_numeric(wide2["AGE_AT_SCAN"], errors="coerce"),
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=False,
        )

        print("\n=== Same test, split by AGE_GROUP ===")
        for ag in ["child", "teen", "young_adult", "adult", "older"]:
            sub_ag = wide2[wide2["AGE_GROUP"] == ag]
            if sub_ag.empty:
                continue

            A_ag = sub_ag.loc[sub_ag["subject_group"] == "F_ASD", "sim_to_M_CTL"].to_numpy()
            B_ag = sub_ag.loc[sub_ag["subject_group"] == "M_ASD", "sim_to_F_CTL"].to_numpy()
            res_ag = run_two_sample(A_ag, B_ag)

            print(f"\n[{ag}] n_A={res_ag['n_A']} n_B={res_ag['n_B']}")
            if res_ag["n_A"] == 0 or res_ag["n_B"] == 0:
                print("  (Not enough data in this bin)")
                continue
            print(f"  mean_A={res_ag['mean_A']:.6f} mean_B={res_ag['mean_B']:.6f} diff={res_ag['mean_diff_A_minus_B']:.6f}")
            print(f"  d={res_ag['cohen_d']:.3f}  p_t={res_ag['p_ttest']}  p_mw={res_ag['p_mannwhitney']}")

            rows_out.append({
                "comparison": f"[{ag}] F_ASD→M_CTL  vs  M_ASD→F_CTL",
                **res_ag
            })

    out_df = pd.DataFrame(rows_out)
    OUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\nSaved summary -> {OUT_SUMMARY_CSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
