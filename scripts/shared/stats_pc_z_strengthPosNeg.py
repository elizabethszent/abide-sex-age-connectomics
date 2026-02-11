import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT    = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
IN_DIR  = ROOT / "results" / "hubs"
OUT_DIR = ROOT / "results" / "hubs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

SEXES      = ["female", "male"]
AGE_GROUPS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
N_MOD      = 7

# These columns must exist in your per-node file
METRICS = [
    ("PC", "PC"),
    ("Z", "z"),
    ("Strength_pos", "strength_pos"),
    ("Strength_neg", "strength_neg"),
]

def safe_ols_pval(term_model, term_name: str) -> float:
    """Return p-value for term, or NaN if unavailable."""
    try:
        p = term_model.pvalues.get(term_name, np.nan)
        if p is None:
            return np.nan
        p = float(p)
        if not np.isfinite(p):
            return np.nan
        return p
    except Exception:
        return np.nan

def run_stats_for_group(sex: str, age_group: str):
    in_path = IN_DIR / f"{sex}_{age_group}_pc_z_strengthPosNeg_revised.csv"
    if not in_path.exists():
        print(f"[SKIP] {in_path} not found")
        return

    df = pd.read_csv(in_path)

    group_cols = ["FILE_ID", "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd"]

    subj_counts = (
        df[group_cols].drop_duplicates()["DX_GROUP"]
        .value_counts()
        .rename({1: "ASD", 2: "CTL"})
    )

    print(f"\n{sex.upper()} | {age_group.upper()}")
    print("  Subjects by DX:", subj_counts.to_dict())

    wide_parts = []
    for metric_name, node_col in METRICS:
        if node_col not in df.columns:
            print(f"  [WARN] missing column '{node_col}', skipping {metric_name}")
            continue

        wide = (
            df.groupby(group_cols + ["module"])[node_col]
              .median()
              .unstack("module")
        )
        wide.columns = [f"{metric_name}_M{m}" for m in wide.columns]
        wide_parts.append(wide)

    if not wide_parts:
        print("  No usable metrics found.")
        return

    subj_df = pd.concat(wide_parts, axis=1).reset_index()

    results = []

    for metric_name, _ in METRICS:
        for m in range(1, N_MOD + 1):
            col = f"{metric_name}_M{m}"
            if col not in subj_df.columns:
                continue

            sub = subj_df[[col, "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd"]].dropna()
            if sub["DX_GROUP"].nunique() != 2:
                # one group missing after dropna
                results.append({
                    "sex": sex, "age_group": age_group, "metric": metric_name, "module": m,
                    "beta_CTL_minus_ASD": np.nan, "p_uncorrected": np.nan,
                    "mean_ASD": np.nan, "mean_CTL": np.nan, "n_ASD": 0, "n_CTL": 0,
                    "note": "missing_group_after_dropna"
                })
                continue

            # If dependent variable is constant -> OLS p-values can be NaN
            if sub[col].nunique(dropna=True) <= 1:
                asd = sub[sub["DX_GROUP"] == 1][col]
                ctl = sub[sub["DX_GROUP"] == 2][col]
                results.append({
                    "sex": sex, "age_group": age_group, "metric": metric_name, "module": m,
                    "beta_CTL_minus_ASD": 0.0, "p_uncorrected": np.nan,
                    "mean_ASD": float(asd.mean()), "mean_CTL": float(ctl.mean()),
                    "n_ASD": int(len(asd)), "n_CTL": int(len(ctl)),
                    "note": "constant_outcome"
                })
                continue

            model = smf.ols(f"{col} ~ C(DX_GROUP) + AGE_AT_SCAN + func_mean_fd", data=sub).fit()
            term = "C(DX_GROUP)[T.2]"  # CTL vs ASD

            beta = float(model.params.get(term, np.nan))
            p    = safe_ols_pval(model, term)

            asd = sub[sub["DX_GROUP"] == 1][col]
            ctl = sub[sub["DX_GROUP"] == 2][col]

            results.append({
                "sex": sex,
                "age_group": age_group,
                "metric": metric_name,
                "module": m,
                "beta_CTL_minus_ASD": beta,
                "p_uncorrected": p,
                "mean_ASD": float(asd.mean()),
                "mean_CTL": float(ctl.mean()),
                "n_ASD": int(len(asd)),
                "n_CTL": int(len(ctl)),
                "note": ""
            })

    stats_df = pd.DataFrame(results)

    # ---- FDR only over finite p-values ----
    stats_df["p_FDR"] = np.nan
    stats_df["FDR_significant"] = False

    valid_mask = np.isfinite(stats_df["p_uncorrected"].to_numpy())
    p_valid = stats_df.loc[valid_mask, "p_uncorrected"].to_numpy()

    if p_valid.size > 0:
        reject, p_fdr, _, _ = multipletests(p_valid, alpha=0.05, method="fdr_bh")
        stats_df.loc[valid_mask, "p_FDR"] = p_fdr
        stats_df.loc[valid_mask, "FDR_significant"] = reject

    out_stats = OUT_DIR / f"{sex}_{age_group}_pc_z_strengthPosNeg_module_stats_revised.csv"
    stats_df.to_csv(out_stats, index=False)
    print(f"  Saved stats -> {out_stats}")

for sex in SEXES:
    for age in AGE_GROUPS:
        run_stats_for_group(sex, age)

print("\nDone.")
