import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        has_results = (p / "results").exists()
        has_meta = (p / "phenotypes").exists() or (p / "data").exists()
        if has_results and has_meta:
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

NODE_FILE = ROOT / "results" / "hubs" / "pc_z_strength_sitecov" / "OVERALL_ageSexMatched_fd-0.3_node_metrics.csv"
OUT_DIR = ROOT / "results" / "qc" / "robustness" / "family_A_teen_female_PCneg_subject_50127"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO = "OVERALL_ageSexMatched_fd-0.3"
SEX_CODE = 2
DX_LABELS = {1: "ASD", 2: "CTL"}
AGE_GROUP = "teen_13_17"
METRIC_COL = "PC_neg"
MODULES = [1, 2, 3, 4, 7]
DROP_SUBJECT = 50127


def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def first_nonnull(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def load_node_metrics() -> pd.DataFrame:
    if not NODE_FILE.exists():
        raise FileNotFoundError(f"Missing node-metrics file: {NODE_FILE}")

    df = pd.read_csv(NODE_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    required = {
        "SUB_ID",
        "DX_GROUP",
        "SEX",
        "AGE_AT_SCAN",
        "AGE_GROUP",
        "SITE_ID",
        "module",
        METRIC_COL,
        "scenario",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{NODE_FILE.name} missing required columns: {sorted(missing)}")

    df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce")
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
    df["module"] = pd.to_numeric(df["module"], errors="coerce")
    df[METRIC_COL] = pd.to_numeric(df[METRIC_COL], errors="coerce")
    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()
    df["AGE_GROUP"] = df["AGE_GROUP"].astype(str).str.strip()

    if "FIQ" in df.columns:
        df["FIQ"] = pd.to_numeric(df["FIQ"], errors="coerce")
    else:
        df["FIQ"] = np.nan

    if "RIGHT_HANDED" in df.columns:
        df["RIGHT_HANDED"] = pd.to_numeric(df["RIGHT_HANDED"], errors="coerce")
    else:
        df["RIGHT_HANDED"] = np.nan

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "module", METRIC_COL]).copy()
    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)
    df["module"] = df["module"].astype(int)

    return df


def build_subject_level(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["scenario"] == SCENARIO)
        & (df["SEX"] == SEX_CODE)
        & (df["AGE_GROUP"] == AGE_GROUP)
        & (df["module"].isin(MODULES))
    ].copy()

    group_cols = [
        "SUB_ID",
        "DX_GROUP",
        "AGE_AT_SCAN",
        "SITE_ID",
        "module",
    ]

    subj = (
        sub.groupby(group_cols, dropna=False)[METRIC_COL]
        .median()
        .reset_index(name="value")
    )

    cov_cols = ["SUB_ID"]
    if "FIQ" in sub.columns:
        cov_cols.append("FIQ")
    if "RIGHT_HANDED" in sub.columns:
        cov_cols.append("RIGHT_HANDED")

    cov_df = (
        sub[cov_cols]
        .groupby("SUB_ID", as_index=False)
        .agg({c: first_nonnull for c in cov_cols if c != "SUB_ID"})
    )

    subj = subj.merge(cov_df, on="SUB_ID", how="left")
    subj["dx_label"] = subj["DX_GROUP"].map(DX_LABELS)

    return subj


def joint_term_pvalue(model, prefix: str) -> float:
    pnames = list(model.params.index)
    idx = [i for i, name in enumerate(pnames) if name.startswith(prefix)]
    if not idx:
        return np.nan

    R = np.zeros((len(idx), len(pnames)))
    for r, j in enumerate(idx):
        R[r, j] = 1.0

    try:
        return safe_float(model.f_test(R).pvalue)
    except Exception:
        return np.nan


def fit_one_model(tmp: pd.DataFrame, include_iq: bool, include_rh: bool) -> dict:
    out = {
        "beta_CTL_minus_ASD": np.nan,
        "p_DX": np.nan,
        "p_SITE": np.nan,
        "p_IQ": np.nan,
        "p_RIGHT_HANDED": np.nan,
        "n_model": 0,
        "n_model_ASD": 0,
        "n_model_CTL": 0,
        "note": "",
    }

    needed = ["value", "DX_GROUP", "AGE_AT_SCAN", "SITE_ID"]
    if include_iq:
        needed.append("FIQ")
    if include_rh:
        needed.append("RIGHT_HANDED")

    tmp_model = tmp[needed].dropna().copy()

    out["n_model"] = int(len(tmp_model))
    out["n_model_ASD"] = int((tmp_model["DX_GROUP"] == 1).sum())
    out["n_model_CTL"] = int((tmp_model["DX_GROUP"] == 2).sum())

    if len(tmp_model) == 0:
        out["note"] = "no_rows_after_dropna"
        return out

    if tmp_model["DX_GROUP"].nunique() != 2:
        out["note"] = "missing_group_after_dropna"
        return out

    if tmp_model["value"].nunique(dropna=True) <= 1:
        out["beta_CTL_minus_ASD"] = 0.0
        out["note"] = "constant_outcome"
        return out

    if include_iq and tmp_model["FIQ"].nunique(dropna=True) <= 1:
        out["note"] = "FIQ_not_variable"
        return out

    if include_rh and tmp_model["RIGHT_HANDED"].nunique(dropna=True) <= 1:
        out["note"] = "RIGHT_HANDED_not_variable"
        return out

    terms = ["C(DX_GROUP)", "AGE_AT_SCAN"]

    use_site = tmp_model["SITE_ID"].nunique() >= 2
    if use_site:
        terms.append("C(SITE_ID)")
    else:
        out["note"] = "single_site_after_dropna"

    if include_iq:
        terms.append("FIQ")

    if include_rh:
        terms.append("C(RIGHT_HANDED)")

    formula = "value ~ " + " + ".join(terms)

    try:
        model = smf.ols(formula, data=tmp_model).fit()

        out["beta_CTL_minus_ASD"] = safe_float(model.params.get("C(DX_GROUP)[T.2]", np.nan))
        out["p_DX"] = safe_float(model.pvalues.get("C(DX_GROUP)[T.2]", np.nan))

        if use_site:
            out["p_SITE"] = joint_term_pvalue(model, "C(SITE_ID)[T.")

        if include_iq:
            out["p_IQ"] = safe_float(model.pvalues.get("FIQ", np.nan))

        if include_rh:
            out["p_RIGHT_HANDED"] = joint_term_pvalue(model, "C(RIGHT_HANDED)[T.")

    except Exception as e:
        out["note"] = f"model_fail:{type(e).__name__}"

    return out


def run_version(subj: pd.DataFrame, version_name: str) -> pd.DataFrame:
    rows = []

    for module in MODULES:
        tmp = subj[subj["module"] == module].copy()

        n_asd = int((tmp["DX_GROUP"] == 1).sum())
        n_ctl = int((tmp["DX_GROUP"] == 2).sum())

        mean_asd = safe_float(tmp.loc[tmp["DX_GROUP"] == 1, "value"].mean()) if n_asd else np.nan
        mean_ctl = safe_float(tmp.loc[tmp["DX_GROUP"] == 2, "value"].mean()) if n_ctl else np.nan

        m1 = fit_one_model(tmp, include_iq=False, include_rh=False)
        m2 = fit_one_model(tmp, include_iq=True, include_rh=False)
        m3 = fit_one_model(tmp, include_iq=True, include_rh=True)

        rows.append(
            {
                "version": version_name,
                "module": module,
                "mean_ASD": mean_asd,
                "mean_CTL": mean_ctl,
                "n_ASD": n_asd,
                "n_CTL": n_ctl,

                "beta_CTL_minus_ASD_m1": m1["beta_CTL_minus_ASD"],
                "p_DX_m1": m1["p_DX"],
                "p_SITE_m1": m1["p_SITE"],
                "p_IQ_m1": np.nan,
                "p_RIGHT_HANDED_m1": np.nan,
                "n_model_m1": m1["n_model"],
                "n_model_ASD_m1": m1["n_model_ASD"],
                "n_model_CTL_m1": m1["n_model_CTL"],
                "note_m1": m1["note"],

                "beta_CTL_minus_ASD_m2": m2["beta_CTL_minus_ASD"],
                "p_DX_m2": m2["p_DX"],
                "p_SITE_m2": m2["p_SITE"],
                "p_IQ_m2": m2["p_IQ"],
                "p_RIGHT_HANDED_m2": np.nan,
                "n_model_m2": m2["n_model"],
                "n_model_ASD_m2": m2["n_model_ASD"],
                "n_model_CTL_m2": m2["n_model_CTL"],
                "note_m2": m2["note"],

                "beta_CTL_minus_ASD_m3": m3["beta_CTL_minus_ASD"],
                "p_DX_m3": m3["p_DX"],
                "p_SITE_m3": m3["p_SITE"],
                "p_IQ_m3": m3["p_IQ"],
                "p_RIGHT_HANDED_m3": m3["p_RIGHT_HANDED"],
                "n_model_m3": m3["n_model"],
                "n_model_ASD_m3": m3["n_model_ASD"],
                "n_model_CTL_m3": m3["n_model_CTL"],
                "note_m3": m3["note"],
            }
        )

    out = pd.DataFrame(rows)

    for model in ["m1", "m2", "m3"]:
        out[f"p_DX_FDR_{model}"] = np.nan
        out[f"DX_FDR_significant_{model}"] = False

        pvals = out[f"p_DX_{model}"].to_numpy(dtype=float)
        valid = np.isfinite(pvals)

        if valid.sum() > 0:
            reject, p_fdr, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
            out.loc[valid, f"p_DX_FDR_{model}"] = p_fdr
            out.loc[valid, f"DX_FDR_significant_{model}"] = reject

    return out


def comparison_table(full_df: pd.DataFrame, excl_df: pd.DataFrame) -> pd.DataFrame:
    full_df = full_df.copy()
    excl_df = excl_df.copy()

    full_df = full_df.add_suffix("_full").rename(columns={"module_full": "module"})
    excl_df = excl_df.add_suffix("_drop50127").rename(columns={"module_drop50127": "module"})

    comp = full_df.merge(excl_df, on="module", how="outer")

    for model in ["m1", "m2", "m3"]:
        b1 = f"beta_CTL_minus_ASD_{model}_full"
        b2 = f"beta_CTL_minus_ASD_{model}_drop50127"
        if b1 in comp.columns and b2 in comp.columns:
            comp[f"delta_beta_{model}"] = comp[b2] - comp[b1]

    return comp.sort_values("module")


def make_plots(subj_full: pd.DataFrame, subj_drop: pd.DataFrame):
    for module in MODULES:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

        for ax, data, title in [
            (axes[0], subj_full[subj_full["module"] == module].copy(), "full sample"),
            (axes[1], subj_drop[subj_drop["module"] == module].copy(), "excluding 50127"),
        ]:
            ctl = data.loc[data["DX_GROUP"] == 2, "value"].to_numpy()
            asd = data.loc[data["DX_GROUP"] == 1, "value"].to_numpy()

            ax.boxplot([ctl, asd], positions=[1, 2], widths=0.5, showfliers=False)

            rng = np.random.default_rng(42)
            if len(ctl):
                ax.scatter(1 + rng.uniform(-0.07, 0.07, size=len(ctl)), ctl, alpha=0.8, s=20)
            if len(asd):
                ax.scatter(2 + rng.uniform(-0.07, 0.07, size=len(asd)), asd, alpha=0.8, s=20)

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["CTL", "ASD"])
            ax.set_title(f"module {module} | {title}")
            ax.set_ylabel("PC_neg")

        fig.suptitle("Family A robustness check: teen female PC_neg", y=1.02)
        fig.tight_layout()
        out_path = OUT_DIR / f"familyA_PCneg_module_{module}__full_vs_drop50127.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    df = load_node_metrics()
    subj = build_subject_level(df)

    subj_full = subj.copy()
    subj_drop = subj[subj["SUB_ID"] != DROP_SUBJECT].copy()

    subj_full.to_csv(OUT_DIR / "familyA_subject_level_full.csv", index=False)
    subj_drop.to_csv(OUT_DIR / "familyA_subject_level_excluding_50127.csv", index=False)

    stats_full = run_version(subj_full, "full")
    stats_drop = run_version(subj_drop, "drop50127")

    stats_full.to_csv(OUT_DIR / "familyA_stats_full.csv", index=False)
    stats_drop.to_csv(OUT_DIR / "familyA_stats_excluding_50127.csv", index=False)

    comp = comparison_table(stats_full, stats_drop)
    comp.to_csv(OUT_DIR / "familyA_stats_comparison_full_vs_drop50127.csv", index=False)

    make_plots(subj_full, subj_drop)

    print(f"[SAVED] {OUT_DIR / 'familyA_subject_level_full.csv'}")
    print(f"[SAVED] {OUT_DIR / 'familyA_subject_level_excluding_50127.csv'}")
    print(f"[SAVED] {OUT_DIR / 'familyA_stats_full.csv'}")
    print(f"[SAVED] {OUT_DIR / 'familyA_stats_excluding_50127.csv'}")
    print(f"[SAVED] {OUT_DIR / 'familyA_stats_comparison_full_vs_drop50127.csv'}")
    print(f"[SAVED] familyA_PCneg_module_*__full_vs_drop50127.png")

    print("\nQuick comparison:")
    show_cols = [
        "module",
        "beta_CTL_minus_ASD_m1_full",
        "p_DX_m1_full",
        "p_DX_FDR_m1_full",
        "beta_CTL_minus_ASD_m1_drop50127",
        "p_DX_m1_drop50127",
        "p_DX_FDR_m1_drop50127",
        "delta_beta_m1",
    ]
    show_cols = [c for c in show_cols if c in comp.columns]
    print(comp[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()