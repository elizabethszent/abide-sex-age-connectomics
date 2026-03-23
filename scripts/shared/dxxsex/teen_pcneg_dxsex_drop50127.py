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
    / "teen_pcneg_drop50127"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO = "OVERALL_ageSexMatched_fd-0.3"
AGE_GROUP = "teen_13_17"
METRIC_NAME = "PC_neg"
DROP_SUBJECT = 50127

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


def first_nonnull(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


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

    if "FIQ" in sub.columns:
        group_cols.append("FIQ")
    if "RIGHT_HANDED" in sub.columns:
        group_cols.append("RIGHT_HANDED")
    if "func_mean_fd" in sub.columns:
        group_cols.append("func_mean_fd")

    subj = (
        sub.groupby(group_cols, dropna=False)[metric_col]
        .median()
        .reset_index(name="value")
    )

    return subj


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

    if include_iq and tmp_model["FIQ"].nunique(dropna=True) <= 1:
        out["note"] = "FIQ_not_variable_after_dropna"
        return out

    if include_rh and tmp_model["RIGHT_HANDED"].nunique(dropna=True) <= 1:
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


def comparison_table(full_df: pd.DataFrame, drop_df: pd.DataFrame) -> pd.DataFrame:
    full_df = full_df.copy().add_suffix("_full").rename(columns={"module_full": "module"})
    drop_df = drop_df.copy().add_suffix("_drop50127").rename(columns={"module_drop50127": "module"})

    comp = full_df.merge(drop_df, on="module", how="outer")

    for model_key in ["m1", "m2", "m3"]:
        b1 = f"beta_DXxSEX_{model_key}_full"
        b2 = f"beta_DXxSEX_{model_key}_drop50127"
        p1 = f"p_DXxSEX_{model_key}_full"
        p2 = f"p_DXxSEX_{model_key}_drop50127"

        if b1 in comp.columns and b2 in comp.columns:
            comp[f"delta_beta_DXxSEX_{model_key}"] = comp[b2] - comp[b1]
        if p1 in comp.columns and p2 in comp.columns:
            comp[f"delta_p_DXxSEX_{model_key}"] = comp[p2] - comp[p1]

    return comp.sort_values("module").reset_index(drop=True)


def make_plots(subj_full: pd.DataFrame, subj_drop: pd.DataFrame):
    rng = np.random.default_rng(42)

    for module in MODULES:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

        for ax, data, title in [
            (axes[0], subj_full[subj_full["module"] == module].copy(), "full sample"),
            (axes[1], subj_drop[subj_drop["module"] == module].copy(), "excluding 50127"),
        ]:
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
            ax.set_title(f"module {module} | {title}")

        fig.suptitle(f"DX×SEX robustness | {SCENARIO} | {AGE_GROUP} | {METRIC_NAME}", y=1.03)
        fig.tight_layout()
        out_path = OUT_DIR / f"dxsex_module_{module}__full_vs_drop50127.png"
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

    subj_full = subj.copy()
    subj_drop = subj[subj["SUB_ID"] != DROP_SUBJECT].copy()

    print("\n[CHECK]")
    print("full unique subjects:", subj_full["SUB_ID"].nunique())
    print("drop unique subjects:", subj_drop["SUB_ID"].nunique())
    print("contains 50127 full:", (subj_full["SUB_ID"] == DROP_SUBJECT).any())
    print("contains 50127 drop:", (subj_drop["SUB_ID"] == DROP_SUBJECT).any())

    subj_full_path = OUT_DIR / "teen_pcneg_dxsex_subject_level_full.csv"
    subj_drop_path = OUT_DIR / "teen_pcneg_dxsex_subject_level_excluding_50127.csv"
    subj_full.to_csv(subj_full_path, index=False)
    subj_drop.to_csv(subj_drop_path, index=False)

    stats_full = run_version(subj_full, "full")
    stats_drop = run_version(subj_drop, "drop50127")

    stats_full_path = OUT_DIR / "teen_pcneg_dxsex_stats_full.csv"
    stats_drop_path = OUT_DIR / "teen_pcneg_dxsex_stats_excluding_50127.csv"
    stats_full.to_csv(stats_full_path, index=False)
    stats_drop.to_csv(stats_drop_path, index=False)

    comp = comparison_table(stats_full, stats_drop)
    comp_path = OUT_DIR / "teen_pcneg_dxsex_stats_comparison_full_vs_drop50127.csv"
    comp.to_csv(comp_path, index=False)

    make_plots(subj_full, subj_drop)

    print(f"\n[SAVED] {subj_full_path}")
    print(f"[SAVED] {subj_drop_path}")
    print(f"[SAVED] {stats_full_path}")
    print(f"[SAVED] {stats_drop_path}")
    print(f"[SAVED] {comp_path}")
    print(f"[SAVED] dxsex_module_*__full_vs_drop50127.png")

    show_cols = [
        "module",
        "beta_DXxSEX_m1_full",
        "p_DXxSEX_m1_full",
        "p_DXxSEX_FDR_m1_full",
        "beta_DXxSEX_m1_drop50127",
        "p_DXxSEX_m1_drop50127",
        "p_DXxSEX_FDR_m1_drop50127",
        "delta_beta_DXxSEX_m1",
        "mean_CTL_male_full",
        "mean_ASD_male_full",
        "mean_CTL_female_full",
        "mean_ASD_female_full",
        "mean_CTL_male_drop50127",
        "mean_ASD_male_drop50127",
        "mean_CTL_female_drop50127",
        "mean_ASD_female_drop50127",
    ]
    show_cols = [c for c in show_cols if c in comp.columns]

    print("\n[QUICK COMPARISON]")
    print(comp[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()