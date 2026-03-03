# stats_module_level_sitecov_4scenarios.py

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# Where the *_node_metrics.csv files are
# (from the newer pipeline: results\hubs\pc_z_strength_sitecov\*_node_metrics.csv)
IN_DIR = ROOT / "results" / "hubs" / "pc_z_strength_sitecov"

# Where to write results
OUT_DIR = ROOT / "results" / "hubs" / "module_stats_sitecov"
OUT_DIR.mkdir(exist_ok=True, parents=True)

SEXES = ["female", "male"]
AGE_GROUPS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
N_MOD = 7  # you forced K=7

METRICS = [
    ("PC", "PC"),
    ("Z", "z"),
    ("Strength_pos", "strength_pos"),
    ("Strength_neg", "strength_neg"),
]

# -------------------------
# helpers
# -------------------------
def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan

def add_sex_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    ABIDE convention: SEX 1=male, 2=female
    """
    df = df.copy()
    df["sex_label"] = np.where(df["SEX"] == 2, "female",
                        np.where(df["SEX"] == 1, "male", "unknown"))
    return df

def run_one_group(df: pd.DataFrame, scenario: str, sex: str, age_group: str):
    subdf = df[(df["sex_label"] == sex) & (df["AGE_GROUP"] == age_group)].copy()
    if subdf.empty:
        print(f"[SKIP] {scenario} | {sex} | {age_group}: no rows")
        return

    # columns we require
    required = {"SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SITE_ID", "module", "node"}
    if not required.issubset(subdf.columns):
        missing = required - set(subdf.columns)
        raise ValueError(f"{scenario}: missing required columns: {missing}")

    # subject counts
    subj_counts = (
        subdf[["SUB_ID", "DX_GROUP"]].drop_duplicates()["DX_GROUP"]
        .value_counts()
        .rename({1: "ASD", 2: "CTL"})
        .to_dict()
    )

    print(f"\n{scenario} | {sex.upper()} | {age_group.upper()}")
    print("  Subjects by DX:", subj_counts)

    # build subject-level wide table: (SUB_ID,DX,AGE,SITE) x module features
    group_cols = ["SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SITE_ID"]

    wide_parts = []
    for metric_name, colname in METRICS:
        if colname not in subdf.columns:
            print(f"  [WARN] missing column '{colname}', skipping {metric_name}")
            continue

        wide = (
            subdf.groupby(group_cols + ["module"])[colname]
                 .median()
                 .unstack("module")
        )

        # ensure columns are 1..7 if present
        wide.columns = [f"{metric_name}_M{int(m)}" for m in wide.columns]
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

            tmp = subj_df[[col, "DX_GROUP", "AGE_AT_SCAN", "SITE_ID"]].dropna()

            # need both groups
            if tmp["DX_GROUP"].nunique() != 2:
                results.append({
                    "scenario": scenario, "sex": sex, "age_group": age_group,
                    "metric": metric_name, "module": m,
                    "beta_CTL_minus_ASD": np.nan,
                    "p_DX": np.nan,
                    "p_SITE": np.nan,
                    "mean_ASD": np.nan, "mean_CTL": np.nan,
                    "n_ASD": 0, "n_CTL": 0,
                    "note": "missing_group_after_dropna"
                })
                continue

            # constant outcome check
            if tmp[col].nunique(dropna=True) <= 1:
                asd = tmp[tmp["DX_GROUP"] == 1][col]
                ctl = tmp[tmp["DX_GROUP"] == 2][col]
                results.append({
                    "scenario": scenario, "sex": sex, "age_group": age_group,
                    "metric": metric_name, "module": m,
                    "beta_CTL_minus_ASD": 0.0,
                    "p_DX": np.nan,
                    "p_SITE": np.nan,
                    "mean_ASD": safe_float(asd.mean()),
                    "mean_CTL": safe_float(ctl.mean()),
                    "n_ASD": int(len(asd)), "n_CTL": int(len(ctl)),
                    "note": "constant_outcome"
                })
                continue

            # --- models ---
            # Reduced: no site
            m_reduced = smf.ols(
                f"{col} ~ C(DX_GROUP) + AGE_AT_SCAN",
                data=tmp
            ).fit()

            # Full: add site covariate
            m_full = smf.ols(
                f"{col} ~ C(DX_GROUP) + AGE_AT_SCAN + C(SITE_ID)",
                data=tmp
            ).fit()

            # DX effect (CTL vs ASD)
            term = "C(DX_GROUP)[T.2]"
            beta = safe_float(m_full.params.get(term, np.nan))
            p_dx = safe_float(m_full.pvalues.get(term, np.nan))

            # Site significance: nested model comparison (reduced vs full)
            try:
                an = anova_lm(m_reduced, m_full)
                # second row corresponds to adding predictors in m_full
                p_site = safe_float(an.iloc[1]["Pr(>F)"])
            except Exception:
                p_site = np.nan

            asd = tmp[tmp["DX_GROUP"] == 1][col]
            ctl = tmp[tmp["DX_GROUP"] == 2][col]

            results.append({
                "scenario": scenario,
                "sex": sex,
                "age_group": age_group,
                "metric": metric_name,
                "module": m,
                "beta_CTL_minus_ASD": beta,
                "p_DX": p_dx,
                "p_SITE": p_site,
                "mean_ASD": safe_float(asd.mean()),
                "mean_CTL": safe_float(ctl.mean()),
                "n_ASD": int(len(asd)),
                "n_CTL": int(len(ctl)),
                "note": ""
            })

    stats_df = pd.DataFrame(results)

    # ---- FDR over p_DX (within this scenario/sex/age) ----
    stats_df["p_DX_FDR"] = np.nan
    stats_df["DX_FDR_significant"] = False

    valid = np.isfinite(stats_df["p_DX"].to_numpy())
    pvals = stats_df.loc[valid, "p_DX"].to_numpy()

    if pvals.size > 0:
        reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        stats_df.loc[valid, "p_DX_FDR"] = p_fdr
        stats_df.loc[valid, "DX_FDR_significant"] = reject

    out_path = OUT_DIR / f"{scenario}__{sex}__{age_group}__module_stats_sitecov.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

def main():
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {IN_DIR}")

    files = sorted(IN_DIR.glob("*_node_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No *_node_metrics.csv found in {IN_DIR}")

    for fp in files:
        scenario = fp.stem.replace("_node_metrics", "")
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip()

        df = add_sex_label(df)

        # quick sanity
        if "AGE_GROUP" not in df.columns:
            raise ValueError(
                f"{fp.name} is missing AGE_GROUP. Re-run the metric export so it includes AGE_GROUP."
            )

        for sex in SEXES:
            for age_group in AGE_GROUPS:
                run_one_group(df, scenario, sex, age_group)

    print("\nDone.")

if __name__ == "__main__":
    main()