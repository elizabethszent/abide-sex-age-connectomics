import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        has_results = (p / "results").exists()
        has_data = (p / "data").exists() or (p / "phenotypes").exists()
        if has_results and has_data:
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

NODE_FILE = (
    ROOT
    / "results"
    / "hubs"
    / "pc_z_strength_sitecov"
    / "OVERALL_ageSexMatched_fd-0.3_node_metrics.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "qc"
    / "robustness_dxsex"
    / "teen_pcneg_multi_sensitivity"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO = "OVERALL_ageSexMatched_fd-0.3"
AGE_GROUP = "teen_13_17"
METRIC_NAME = "PC_neg"
FEMALE_SUBJECT_TO_DROP = 50127

# set this to an int if you already know the male subject id
MALE_SUBJECT_OVERRIDE = None

MODULES = [1, 2, 3, 4, 7, 8]

METRIC_ALIASES = {
    "PC": ["PC", "pc"],
    "PC_pos": ["PC_pos", "pc_pos"],
    "PC_neg": ["PC_neg", "pc_neg"],
    "Z": ["z", "Z"],
    "Z_pos": ["z_pos", "Z_pos"],
    "Z_neg": ["z_neg", "Z_neg"],
    "Strength_pos": ["strength_pos", "Strength_pos"],
    "Strength_neg": ["strength_neg", "Strength_neg"],
}

DX_TERM = "C(DX_GROUP)[T.2]"
SEX_TERM = "C(SEX)[T.2]"
DXxSEX_TERM = "C(DX_GROUP)[T.2]:C(SEX)[T.2]"

GROUP_ORDER = [
    ("CTL_male", 2, 1),
    ("ASD_male", 1, 1),
    ("CTL_female", 2, 2),
    ("ASD_female", 1, 2),
]


def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def resolve_metric_column(df: pd.DataFrame, metric_name: str) -> str:
    aliases = METRIC_ALIASES[metric_name]
    for col in aliases:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a column for {metric_name}. Tried {aliases}")


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


def load_node_metrics() -> pd.DataFrame:
    if not NODE_FILE.exists():
        raise FileNotFoundError(f"Missing node metrics file: {NODE_FILE}")

    df = pd.read_csv(NODE_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    numeric_cols = [
        "SUB_ID",
        "DX_GROUP",
        "SEX",
        "AGE_AT_SCAN",
        "module",
        "node",
        "FIQ",
        "RIGHT_HANDED",
        "func_mean_fd",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "SITE_ID" in df.columns:
        df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

    if "AGE_GROUP" in df.columns:
        df["AGE_GROUP"] = df["AGE_GROUP"].astype(str).str.strip()

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "module", "node"]).copy()
    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)
    df["module"] = df["module"].astype(int)
    df["node"] = df["node"].astype(int)

    return df


def build_subject_level_df(df: pd.DataFrame) -> pd.DataFrame:
    metric_col = resolve_metric_column(df, METRIC_NAME)

    sub = df[
        (df["AGE_GROUP"] == AGE_GROUP)
        & (df["module"].isin(MODULES))
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    group_cols = [
        "SUB_ID",
        "DX_GROUP",
        "SEX",
        "AGE_AT_SCAN",
        "AGE_GROUP",
        "SITE_ID",
        "module",
    ]

    extra_cols = []
    for col in ["FIQ", "RIGHT_HANDED", "func_mean_fd"]:
        if col in sub.columns:
            extra_cols.append(col)
    group_cols += extra_cols

    subj = (
        sub.groupby(group_cols, dropna=False)[metric_col]
        .median()
        .reset_index(name="value")
    )

    return subj


def identify_male_candidate(subj: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    male_asd = subj[(subj["DX_GROUP"] == 1) & (subj["SEX"] == 1)].copy()
    if male_asd.empty:
        raise RuntimeError("No ASD male rows found in teen subject-level dataframe.")

    piv = male_asd.pivot_table(index="SUB_ID", columns="module", values="value", aggfunc="first")
    piv = piv.reindex(columns=MODULES)

    summary = pd.DataFrame(index=piv.index)
    summary["n_modules_present"] = piv.notna().sum(axis=1)
    summary["mean_across_modules"] = piv.mean(axis=1, skipna=True)
    summary["median_across_modules"] = piv.median(axis=1, skipna=True)
    summary["min_across_modules"] = piv.min(axis=1, skipna=True)
    summary["max_across_modules"] = piv.max(axis=1, skipna=True)

    for m in MODULES:
        summary[f"module_{m}"] = piv[m]

    summary = summary.reset_index().sort_values(
        ["n_modules_present", "mean_across_modules", "median_across_modules", "min_across_modules"],
        ascending=[False, True, True, True],
    )

    # prefer subjects present in all modules
    full_cov = summary[summary["n_modules_present"] == len(MODULES)].copy()
    if not full_cov.empty:
        candidate = int(full_cov.iloc[0]["SUB_ID"])
    else:
        candidate = int(summary.iloc[0]["SUB_ID"])

    return candidate, summary


def compute_group_summary(tmp_basic: pd.DataFrame) -> dict:
    out = {}

    for label, dx_val, sex_val in GROUP_ORDER:
        vals = tmp_basic.loc[
            (tmp_basic["DX_GROUP"] == dx_val) & (tmp_basic["SEX"] == sex_val),
            "value",
        ]
        out[f"mean_{label}"] = safe_float(vals.mean()) if len(vals) else np.nan
        out[f"median_{label}"] = safe_float(vals.median()) if len(vals) else np.nan
        out[f"n_{label}"] = int(len(vals))

    out["n_ASD"] = int((tmp_basic["DX_GROUP"] == 1).sum())
    out["n_CTL"] = int((tmp_basic["DX_GROUP"] == 2).sum())
    out["n_male"] = int((tmp_basic["SEX"] == 1).sum())
    out["n_female"] = int((tmp_basic["SEX"] == 2).sum())
    out["n_total"] = int(len(tmp_basic))

    out["dx_effect_male"] = (
        out["mean_CTL_male"] - out["mean_ASD_male"]
        if pd.notna(out["mean_CTL_male"]) and pd.notna(out["mean_ASD_male"])
        else np.nan
    )
    out["dx_effect_female"] = (
        out["mean_CTL_female"] - out["mean_ASD_female"]
        if pd.notna(out["mean_CTL_female"]) and pd.notna(out["mean_ASD_female"])
        else np.nan
    )

    return out


def fit_one_model(tmp: pd.DataFrame, include_iq: bool, include_rh: bool) -> dict:
    needed = ["value", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]
    if include_iq:
        needed.append("FIQ")
    if include_rh:
        needed.append("RIGHT_HANDED")

    tmp_model = tmp[needed].dropna().copy()

    n_model = int(len(tmp_model))
    n_model_asd = int((tmp_model["DX_GROUP"] == 1).sum())
    n_model_ctl = int((tmp_model["DX_GROUP"] == 2).sum())
    n_model_male = int((tmp_model["SEX"] == 1).sum())
    n_model_female = int((tmp_model["SEX"] == 2).sum())

    cell_counts = tmp_model.groupby(["DX_GROUP", "SEX"]).size().to_dict() if n_model else {}
    n_asd_male = int(cell_counts.get((1, 1), 0))
    n_asd_female = int(cell_counts.get((1, 2), 0))
    n_ctl_male = int(cell_counts.get((2, 1), 0))
    n_ctl_female = int(cell_counts.get((2, 2), 0))

    out = {
        "beta_DX": np.nan,
        "p_DX": np.nan,
        "beta_SEX": np.nan,
        "p_SEX": np.nan,
        "beta_DXxSEX": np.nan,
        "p_DXxSEX": np.nan,
        "p_SITE": np.nan,
        "p_IQ": np.nan,
        "p_RIGHT_HANDED": np.nan,
        "note": "",
        "n_model": n_model,
        "n_model_ASD": n_model_asd,
        "n_model_CTL": n_model_ctl,
        "n_model_male": n_model_male,
        "n_model_female": n_model_female,
        "n_model_ASD_male": n_asd_male,
        "n_model_ASD_female": n_asd_female,
        "n_model_CTL_male": n_ctl_male,
        "n_model_CTL_female": n_ctl_female,
    }

    if n_model == 0:
        out["note"] = "no_rows_after_dropna"
        return out

    if tmp_model["DX_GROUP"].nunique() != 2:
        out["note"] = "missing_dx_group_after_dropna"
        return out

    if tmp_model["SEX"].nunique() != 2:
        out["note"] = "missing_sex_group_after_dropna"
        return out

    if min(n_asd_male, n_asd_female, n_ctl_male, n_ctl_female) == 0:
        out["note"] = "missing_dx_sex_cell_after_dropna"
        return out

    if tmp_model["value"].nunique(dropna=True) <= 1:
        out["beta_DX"] = 0.0
        out["beta_SEX"] = 0.0
        out["beta_DXxSEX"] = 0.0
        out["note"] = "constant_outcome"
        return out

    if include_iq and "FIQ" in tmp_model.columns and tmp_model["FIQ"].nunique(dropna=True) <= 1:
        out["note"] = "FIQ_not_variable_after_dropna"
        return out

    if include_rh and "RIGHT_HANDED" in tmp_model.columns and tmp_model["RIGHT_HANDED"].nunique(dropna=True) <= 1:
        out["note"] = "RIGHT_HANDED_not_variable_after_dropna"
        return out

    terms = ["C(DX_GROUP) * C(SEX)", "AGE_AT_SCAN"]

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

        out["beta_DX"] = safe_float(model.params.get(DX_TERM, np.nan))
        out["p_DX"] = safe_float(model.pvalues.get(DX_TERM, np.nan))

        out["beta_SEX"] = safe_float(model.params.get(SEX_TERM, np.nan))
        out["p_SEX"] = safe_float(model.pvalues.get(SEX_TERM, np.nan))

        out["beta_DXxSEX"] = safe_float(model.params.get(DXxSEX_TERM, np.nan))
        out["p_DXxSEX"] = safe_float(model.pvalues.get(DXxSEX_TERM, np.nan))

        if use_site:
            out["p_SITE"] = joint_term_pvalue(model, "C(SITE_ID)[T.")

        if include_iq:
            out["p_IQ"] = safe_float(model.pvalues.get("FIQ", np.nan))

        if include_rh:
            out["p_RIGHT_HANDED"] = joint_term_pvalue(model, "C(RIGHT_HANDED)[T.")

    except Exception as e:
        if out["note"]:
            out["note"] = f"{out['note']};model_fail:{type(e).__name__}"
        else:
            out["note"] = f"model_fail:{type(e).__name__}"

    return out


def run_version(subj: pd.DataFrame, version_name: str) -> pd.DataFrame:
    rows = []

    for module in MODULES:
        tmp = subj[subj["module"] == module].copy()
        if tmp.empty:
            continue

        tmp_basic = tmp.dropna(subset=["value", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]).copy()

        row = {
            "version": version_name,
            "module": module,
        }
        row.update(compute_group_summary(tmp_basic))

        m1 = fit_one_model(tmp, include_iq=False, include_rh=False)
        m2 = fit_one_model(tmp, include_iq=True, include_rh=False)
        m3 = fit_one_model(tmp, include_iq=True, include_rh=True)

        for prefix, fitted in [("m1", m1), ("m2", m2), ("m3", m3)]:
            for k, v in fitted.items():
                row[f"{k}_{prefix}"] = v

        rows.append(row)

    out = pd.DataFrame(rows)

    for model_key in ["m1", "m2", "m3"]:
        out[f"p_DXxSEX_FDR_{model_key}"] = np.nan
        out[f"DXxSEX_FDR_significant_{model_key}"] = False

        pvals = out[f"p_DXxSEX_{model_key}"].to_numpy(dtype=float)
        valid = np.isfinite(pvals)

        if valid.sum() > 0:
            reject, p_fdr, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
            out.loc[valid, f"p_DXxSEX_FDR_{model_key}"] = p_fdr
            out.loc[valid, f"DXxSEX_FDR_significant_{model_key}"] = reject

    return out.sort_values("module").reset_index(drop=True)


def comparison_table(version_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None

    for version_name, df in version_tables.items():
        tmp = df.copy().add_suffix(f"_{version_name}")
        tmp = tmp.rename(columns={f"module_{version_name}": "module"})

        if merged is None:
            merged = tmp
        else:
            merged = merged.merge(tmp, on="module", how="outer")

    for model_key in ["m1", "m2", "m3"]:
        full_beta = f"beta_DXxSEX_{model_key}_full"
        full_p = f"p_DXxSEX_{model_key}_full"

        for compare_version in ["drop50127", "drop_male", "drop_both"]:
            beta_col = f"beta_DXxSEX_{model_key}_{compare_version}"
            p_col = f"p_DXxSEX_{model_key}_{compare_version}"

            if full_beta in merged.columns and beta_col in merged.columns:
                merged[f"delta_beta_DXxSEX_{model_key}_{compare_version}"] = merged[beta_col] - merged[full_beta]

            if full_p in merged.columns and p_col in merged.columns:
                merged[f"delta_p_DXxSEX_{model_key}_{compare_version}"] = merged[p_col] - merged[full_p]

    return merged.sort_values("module").reset_index(drop=True)


def make_plots(version_subjects: dict[str, pd.DataFrame]):
    rng = np.random.default_rng(42)
    version_order = ["full", "drop50127", "drop_male", "drop_both"]

    for module in MODULES:
        ncols = 2
        nrows = math.ceil(len(version_order) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.8 * nrows), sharey=True)
        axes = axes.flatten()

        for ax, version_name in zip(axes, version_order):
            data = version_subjects[version_name]
            data = data[data["module"] == module].copy()

            series_list = []
            labels = []

            for label, dx_val, sex_val in GROUP_ORDER:
                vals = data.loc[
                    (data["DX_GROUP"] == dx_val) & (data["SEX"] == sex_val),
                    "value",
                ].to_numpy()
                series_list.append(vals)
                labels.append(label.replace("_", "\n"))

            ax.boxplot(series_list, positions=[1, 2, 3, 4], widths=0.5, showfliers=False)

            for i, vals in enumerate(series_list, start=1):
                if len(vals):
                    xs = i + rng.uniform(-0.07, 0.07, size=len(vals))
                    ax.scatter(xs, vals, alpha=0.8, s=18)

            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(labels)
            ax.set_ylabel(METRIC_NAME)
            ax.set_title(f"module {module} | {version_name}")

        for ax in axes[len(version_order):]:
            ax.axis("off")

        fig.suptitle(f"DX×SEX sensitivity | {SCENARIO} | {AGE_GROUP} | {METRIC_NAME}", y=1.02)
        fig.tight_layout()
        out_path = OUT_DIR / f"module_{module}__multi_sensitivity.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] node file: {NODE_FILE}")
    print(f"[INFO] output dir: {OUT_DIR}")

    df = load_node_metrics()
    subj = build_subject_level_df(df)

    if subj.empty:
        raise RuntimeError("No subject-level rows found for this family.")

    candidate_summary_path = OUT_DIR / "male_candidate_summary.csv"

    if MALE_SUBJECT_OVERRIDE is None:
        male_candidate, male_summary = identify_male_candidate(subj)
    else:
        male_candidate = int(MALE_SUBJECT_OVERRIDE)
        _, male_summary = identify_male_candidate(subj)

    male_summary.to_csv(candidate_summary_path, index=False)

    print(f"\n[INFO] female subject to drop: {FEMALE_SUBJECT_TO_DROP}")
    print(f"[INFO] male candidate to drop: {male_candidate}")
    print(f"[SAVED] {candidate_summary_path}")

    version_subjects = {
        "full": subj.copy(),
        "drop50127": subj[subj["SUB_ID"] != FEMALE_SUBJECT_TO_DROP].copy(),
        "drop_male": subj[subj["SUB_ID"] != male_candidate].copy(),
        "drop_both": subj[
            (~subj["SUB_ID"].isin([FEMALE_SUBJECT_TO_DROP, male_candidate]))
        ].copy(),
    }

    for name, sdf in version_subjects.items():
        out_path = OUT_DIR / f"subject_level_{name}.csv"
        sdf.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path}")
        print(
            f"[CHECK] {name}: subjects={sdf['SUB_ID'].nunique()} | "
            f"contains_50127={(sdf['SUB_ID'] == FEMALE_SUBJECT_TO_DROP).any()} | "
            f"contains_male={(sdf['SUB_ID'] == male_candidate).any()}"
        )

    version_tables = {}
    for name, sdf in version_subjects.items():
        stats = run_version(sdf, name)
        version_tables[name] = stats
        out_path = OUT_DIR / f"stats_{name}.csv"
        stats.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path}")

    comp = comparison_table(version_tables)
    comp_path = OUT_DIR / "stats_comparison_all_versions.csv"
    comp.to_csv(comp_path, index=False)
    print(f"[SAVED] {comp_path}")

    make_plots(version_subjects)
    print(f"[SAVED] module_*__multi_sensitivity.png")

    show_cols = [
        "module",

        "beta_DXxSEX_m1_full",
        "p_DXxSEX_m1_full",
        "p_DXxSEX_FDR_m1_full",

        "beta_DXxSEX_m1_drop50127",
        "p_DXxSEX_m1_drop50127",
        "p_DXxSEX_FDR_m1_drop50127",

        "beta_DXxSEX_m1_drop_male",
        "p_DXxSEX_m1_drop_male",
        "p_DXxSEX_FDR_m1_drop_male",

        "beta_DXxSEX_m1_drop_both",
        "p_DXxSEX_m1_drop_both",
        "p_DXxSEX_FDR_m1_drop_both",

        "mean_CTL_male_full",
        "mean_ASD_male_full",
        "mean_CTL_female_full",
        "mean_ASD_female_full",

        "mean_CTL_male_drop_male",
        "mean_ASD_male_drop_male",
        "mean_CTL_female_drop_male",
        "mean_ASD_female_drop_male",

        "mean_CTL_male_drop_both",
        "mean_ASD_male_drop_both",
        "mean_CTL_female_drop_both",
        "mean_ASD_female_drop_both",
    ]
    show_cols = [c for c in show_cols if c in comp.columns]

    print("\n[QUICK COMPARISON m1]")
    print(comp[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()