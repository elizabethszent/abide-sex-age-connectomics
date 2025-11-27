import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT   = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")
IN_DIR = ROOT / "results/hubs"
OUT_DIR = ROOT / "results/hubs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

SEXES      = ["female", "male"]
AGE_GROUPS = ["child", "teen", "young_adult"]
N_MOD      = 7


def run_stats_for_group(sex: str, age_group: str):
    in_path = IN_DIR / f"{sex}_{age_group}_pc_z.csv"
    if not in_path.exists():
        print(f"[SKIP] {in_path} not found")
        return

    df = pd.read_csv(in_path)

    #sanity check should all be same age_group
    if df["AGE_GROUP"].nunique() > 1:
        print(f"[WARN] {sex} {age_group}: multiple AGE_GROUP values present")

    #median PC and z per module per subject
    group_cols = ["FILE_ID", "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd"]
    pc_mod = (
        df.groupby(group_cols + ["module"])["PC"]
          .median()
          .unstack("module")
    )
    pc_mod.columns = [f"PC_M{m}" for m in pc_mod.columns]

    z_mod = (
        df.groupby(group_cols + ["module"])["z"]
          .median()
          .unstack("module")
    )
    z_mod.columns = [f"Z_M{m}" for m in z_mod.columns]

    subj_df = pc_mod.join(z_mod).reset_index()

    print(f"\n {sex.upper()} | {age_group.upper()}")
    print(f"  Subjects: {len(subj_df)}")
    print(subj_df["DX_GROUP"].value_counts().rename({1: "ASD", 2: "CTL"}))

    results = []

    #we’ll collect all p-values (PC+Z) for FDR
    p_all = []

    #PC stats
    for m in range(1, N_MOD + 1):
        col = f"PC_M{m}"
        if col not in subj_df.columns:
            continue

        sub = subj_df[[col, "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd"]].dropna()
        if sub["DX_GROUP"].nunique() != 2:
            continue

        model = smf.ols(f"{col} ~ C(DX_GROUP) + AGE_AT_SCAN + func_mean_fd", data=sub).fit()
        term = "C(DX_GROUP)[T.2]" #CTL vs ASD
        beta = model.params.get(term, float("nan"))
        p    = model.pvalues.get(term, float("nan"))

        asd = sub[sub["DX_GROUP"] == 1][col]
        ctl = sub[sub["DX_GROUP"] == 2][col]

        results.append({
            "sex": sex,
            "age_group": age_group,
            "metric": "PC",
            "module": m,
            "beta_CTL_minus_ASD": beta,
            "p_uncorrected": p,
            "mean_ASD": asd.mean(),
            "mean_CTL": ctl.mean(),
            "n_ASD": len(asd),
            "n_CTL": len(ctl),
        })
        p_all.append(p)

    #z stats
    for m in range(1, N_MOD + 1):
        col = f"Z_M{m}"
        if col not in subj_df.columns:
            continue

        sub = subj_df[[col, "DX_GROUP", "AGE_AT_SCAN", "func_mean_fd"]].dropna()
        if sub["DX_GROUP"].nunique() != 2:
            continue

        model = smf.ols(f"{col} ~ C(DX_GROUP) + AGE_AT_SCAN + func_mean_fd", data=sub).fit()
        term = "C(DX_GROUP)[T.2]"
        beta = model.params.get(term, float("nan"))
        p    = model.pvalues.get(term, float("nan"))

        asd = sub[sub["DX_GROUP"] == 1][col]
        ctl = sub[sub["DX_GROUP"] == 2][col]

        results.append({
            "sex": sex,
            "age_group": age_group,
            "metric": "Z",
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
        print("  No valid modules.")
        return

    stats_df = pd.DataFrame(results)

    #FDR across all PC+Z tests for this sex+age_group
    reject, p_fdr, _, _ = multipletests(p_all, alpha=0.05, method="fdr_bh")
    stats_df["p_FDR"] = p_fdr
    stats_df["FDR_significant"] = reject

    out_stats = OUT_DIR / f"{sex}_{age_group}_pc_z_module_stats.csv"
    stats_df.to_csv(out_stats, index=False)
    print(f"  Saved stats -> {out_stats}")


for sex in SEXES:
    for age in AGE_GROUPS:
        run_stats_for_group(sex, age)

print("\nDone.")
