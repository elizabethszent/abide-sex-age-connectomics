import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT    = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")
IN_DIR  = ROOT / "results/hubs"
OUT_DIR = ROOT / "results/hubs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

SEXES      = ["female", "male"]
AGE_GROUPS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
N_MOD      = 7

# Which node-level metrics to analyze -> we compute per-subject median per module
METRICS = [
    ("PC", "PC"),
    ("Z", "z"),
    ("Strength", "strength"),
]


def run_stats_for_group(sex: str, age_group: str):
    in_path = IN_DIR / f"{sex}_{age_group}_pc_z_strength_revised.csv"
    if not in_path.exists():
        print(f"[SKIP] {in_path} not found")
        return

    df = pd.read_csv(in_path)

    # sanity check: should all be same age_group
    if df["AGE_GROUP"].nunique() > 1:
        print(f"[WARN] {sex} {age_group}: multiple AGE_GROUP values present")

    group_cols = ["FILE_ID", "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd"]

    print(f"\n{sex.upper()} | {age_group.upper()}")
    print(f"  Node-rows: {len(df)}")
    # subject counts
    subj_counts = df[group_cols].drop_duplicates()["DX_GROUP"].value_counts().rename({1: "ASD", 2: "CTL"})
    print("  Subjects by DX:", subj_counts.to_dict())

    # Build a per-subject dataframe that has one column per module for each metric
    wide_parts = []
    for metric_name, node_col in METRICS:
        if node_col not in df.columns:
            print(f"  [WARN] missing column '{node_col}' in {in_path}, skipping {metric_name}")
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

    print(f"  Subjects (usable): {len(subj_df)}")
    print(subj_df["DX_GROUP"].value_counts().rename({1: "ASD", 2: "CTL"}))

    results = []
    p_all = []

    # Run OLS for every metric x module
    for metric_name, _ in METRICS:
        for m in range(1, N_MOD + 1):
            col = f"{metric_name}_M{m}"
            if col not in subj_df.columns:
                continue

            sub = subj_df[[col, "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd"]].dropna()
            if sub["DX_GROUP"].nunique() != 2:
                continue

            model = smf.ols(f"{col} ~ C(DX_GROUP) + AGE_AT_SCAN + func_mean_fd", data=sub).fit()

            term = "C(DX_GROUP)[T.2]"  # CTL vs ASD (same as your original)
            beta = model.params.get(term, float("nan"))
            p    = model.pvalues.get(term, float("nan"))

            asd = sub[sub["DX_GROUP"] == 1][col]
            ctl = sub[sub["DX_GROUP"] == 2][col]

            results.append({
                "sex": sex,
                "age_group": age_group,
                "metric": metric_name,
                "module": m,
                "beta_CTL_minus_ASD": beta,
                "p_uncorrected": p,
                "mean_ASD": asd.mean(),
                "mean_CTL": ctl.mean(),
                "n_ASD": len(asd),
                "n_CTL": len(ctl),
            })
            p_all.append(p)

    if not results:
        print("  No valid module tests.")
        return

    stats_df = pd.DataFrame(results)

    # FDR across ALL tests for this sex+age_group (PC + Z + Strength across 7 modules)
    reject, p_fdr, _, _ = multipletests(p_all, alpha=0.05, method="fdr_bh")
    stats_df["p_FDR"] = p_fdr
    stats_df["FDR_significant"] = reject

    out_stats = OUT_DIR / f"{sex}_{age_group}_pc_z_strength_module_stats_revised.csv"
    stats_df.to_csv(out_stats, index=False)
    print(f"  Saved stats -> {out_stats}")


for sex in SEXES:
    for age in AGE_GROUPS:
        run_stats_for_group(sex, age)

print("\nDone.")
