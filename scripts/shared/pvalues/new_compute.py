import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

NODE_DIR = ROOT / "results" / "hubs" / "pc_z_strength_sitecov"
OUT_DIR  = ROOT / "results" / "hubs" / "module_stats_sitecov"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "OVERALL_sexbalanced_fd-0.2",
    "OVERALL_sexbalanced_fd-0.3",
    "OVERALL_ageSexMatched_fd-0.2",
    "OVERALL_ageSexMatched_fd-0.3",
]

SEXES = {
    "female": 2,  # ABIDE coding
    "male": 1,
}
AGE_GROUPS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
N_MOD = 7

# node-metrics columns -> output metric names
METRICS = [
    ("PC", "PC"),
    ("PC_pos", "PC_pos"),
    ("PC_neg", "PC_neg"),
    ("Z", "z"),
    ("Z_pos", "z_pos"),
    ("Z_neg", "z_neg"),
    ("Strength_pos", "strength_pos"),
    ("Strength_neg", "strength_neg"),
]

COV_FORMULA = "C(DX_GROUP) + AGE_AT_SCAN + C(SITE_ID)"
DX_TERM = "C(DX_GROUP)[T.2]"   # CTL - ASD (since ASD=1 is baseline)

def load_node_metrics(scenario: str) -> pd.DataFrame:
    fp = NODE_DIR / f"{scenario}_node_metrics.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing node metrics: {fp}")
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip()
    # normalize types
    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()
    df["AGE_GROUP"] = df["AGE_GROUP"].astype(str).str.strip()
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
    df["module"] = pd.to_numeric(df["module"], errors="coerce")
    df = df.dropna(subset=["DX_GROUP", "SEX", "AGE_AT_SCAN", "module"])
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)
    df["module"] = df["module"].astype(int)
    return df

def site_df_resid_ok(model) -> bool:
    try:
        return float(model.df_resid) > 0
    except Exception:
        return False

def joint_site_pvalue(model) -> float:
    # params for site are like C(SITE_ID)[T.XXXX]
    pnames = list(model.params.index)
    site_idx = [i for i, n in enumerate(pnames) if n.startswith("C(SITE_ID)[T.")]
    if not site_idx:
        return np.nan
    R = np.zeros((len(site_idx), len(pnames)))
    for r, j in enumerate(site_idx):
        R[r, j] = 1.0
    try:
        return float(model.f_test(R).pvalue)
    except Exception:
        return np.nan

def compute_module_level(sub_nodes: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Convert node-level -> subject-level per-module median.
    Returns wide table with columns M1..M7 and covariates.
    """
    subj_cols = ["SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SITE_ID"]
    g = (sub_nodes
         .groupby(subj_cols + ["module"], dropna=False)[value_col]
         .median()
         .unstack("module"))
    # ensure all modules exist as columns
    for m in range(1, N_MOD + 1):
        if m not in g.columns:
            g[m] = np.nan
    g = g[[m for m in range(1, N_MOD + 1)]]
    g.columns = [f"M{m}" for m in range(1, N_MOD + 1)]
    return g.reset_index()

def run_one_file(scenario: str, sex_name: str, age_group: str):
    df = load_node_metrics(scenario)

    # filter sex + age
    sex_code = SEXES[sex_name]
    df = df[(df["SEX"] == sex_code) & (df["AGE_GROUP"] == age_group)].copy()
    if df.empty:
        return

    out_rows = []

    for out_metric, node_col in METRICS:
        if node_col not in df.columns:
            continue

        wide = compute_module_level(df, node_col)

        for m in range(1, N_MOD + 1):
            col = f"M{m}"
            sub = wide[[col, "DX_GROUP", "AGE_AT_SCAN", "SITE_ID"]].dropna()

            n_asd = int((sub["DX_GROUP"] == 1).sum())
            n_ctl = int((sub["DX_GROUP"] == 2).sum())

            # defaults
            beta = np.nan
            p_dx = np.nan
            p_site = np.nan
            r2 = np.nan
            note = ""

            # basic feasibility
            if n_asd == 0 or n_ctl == 0:
                note = "missing_dx_group_after_dropna"
            elif sub[col].nunique(dropna=True) <= 1:
                note = "constant_outcome"
                # beta=0 is misleading; keep NaN
            else:
                # Fit with site covariate
                try:
                    model = smf.ols(f"{col} ~ {COV_FORMULA}", data=sub).fit()

                    # If too many parameters vs n, stats become NaN (df_resid<=0)
                    if not site_df_resid_ok(model):
                        note = "df_resid<=0_too_many_params_for_n"
                    else:
                        beta = float(model.params.get(DX_TERM, np.nan))
                        p_dx = float(model.pvalues.get(DX_TERM, np.nan))
                        p_site = joint_site_pvalue(model)
                        r2 = float(model.rsquared)
                except Exception as e:
                    note = f"model_fail:{type(e).__name__}"

            mean_asd = float(sub.loc[sub["DX_GROUP"] == 1, col].mean()) if n_asd else np.nan
            mean_ctl = float(sub.loc[sub["DX_GROUP"] == 2, col].mean()) if n_ctl else np.nan

            out_rows.append({
                "scenario": scenario,
                "sex": sex_name,
                "age_group": age_group,
                "metric": out_metric,
                "module": m,
                "mean_ASD": mean_asd,
                "mean_CTL": mean_ctl,
                "beta_CTL_minus_ASD": beta,
                "p_DX": p_dx,
                "p_SITE": p_site,
                "r2": r2,
                "n_ASD": n_asd,
                "n_CTL": n_ctl,
                "note": note,
            })

    out_df = pd.DataFrame(out_rows)

    # FDR correction separately within each metric
    out_df["p_DX_FDR"] = np.nan
    out_df["DX_FDR_significant"] = False

    for metric in out_df["metric"].unique():
        mask = (out_df["metric"] == metric) & np.isfinite(out_df["p_DX"])
        pvals = out_df.loc[mask, "p_DX"].to_numpy()
        if pvals.size:
            reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            out_df.loc[mask, "p_DX_FDR"] = p_fdr
            out_df.loc[mask, "DX_FDR_significant"] = reject

    out_path = OUT_DIR / f"{scenario}__{sex_name}__{age_group}__module_stats_sitecov.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")

def main():
    for scenario in SCENARIOS:
        for sex in SEXES.keys():
            for age in AGE_GROUPS:
                run_one_file(scenario, sex, age)

    print("\n[DONE] module_stats_sitecov rebuilt (with pos/neg PC/Z).")

if __name__ == "__main__":
    main()