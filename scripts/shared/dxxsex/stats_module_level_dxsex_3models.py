import numpy as np
import pandas as pd
from pathlib import Path
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

IN_DIR = ROOT / "results" / "hubs" / "pc_z_strength_sitecov"
OUT_DIR = ROOT / "results" / "hubs" / "module_stats_dxsex"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGE_GROUPS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]

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

MODELS = {
    "m1": {"include_iq": False, "include_rh": False, "label": "site"},
    "m2": {"include_iq": True,  "include_rh": False, "label": "site_iq"},
    "m3": {"include_iq": True,  "include_rh": True,  "label": "site_iq_rh"},
}

DX_TERM = "C(DX_GROUP)[T.2]"
SEX_TERM = "C(SEX)[T.2]"
DXxSEX_TERM = "C(DX_GROUP)[T.2]:C(SEX)[T.2]"


def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def first_nonnull(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def resolve_metric_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def get_modules_present(df: pd.DataFrame) -> list[int]:
    mods = pd.to_numeric(df["module"], errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(mods)


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


def build_subject_level_table(
    subdf: pd.DataFrame,
    metric_name: str,
    metric_col: str,
    modules_present: list[int],
) -> pd.DataFrame:
    group_cols = ["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]

    wide = (
        subdf.groupby(group_cols + ["module"])[metric_col]
        .median()
        .unstack("module")
    )

    for m in modules_present:
        if m not in wide.columns:
            wide[m] = np.nan

    wide = wide[modules_present]
    wide.columns = [f"{metric_name}_M{m}" for m in wide.columns]
    wide = wide.reset_index()

    cov_cols = ["SUB_ID"]
    if "FIQ" in subdf.columns:
        cov_cols.append("FIQ")
    if "RIGHT_HANDED" in subdf.columns:
        cov_cols.append("RIGHT_HANDED")

    if len(cov_cols) > 1:
        cov_df = (
            subdf[cov_cols]
            .groupby("SUB_ID", as_index=False)
            .agg({c: first_nonnull for c in cov_cols if c != "SUB_ID"})
        )
        wide = wide.merge(cov_df, on="SUB_ID", how="left")

    return wide


def compute_group_means_and_counts(tmp_basic: pd.DataFrame, col: str) -> dict:
    masks = {
        "ASD_male": (tmp_basic["DX_GROUP"] == 1) & (tmp_basic["SEX"] == 1),
        "ASD_female": (tmp_basic["DX_GROUP"] == 1) & (tmp_basic["SEX"] == 2),
        "CTL_male": (tmp_basic["DX_GROUP"] == 2) & (tmp_basic["SEX"] == 1),
        "CTL_female": (tmp_basic["DX_GROUP"] == 2) & (tmp_basic["SEX"] == 2),
    }

    out = {}
    for key, mask in masks.items():
        vals = tmp_basic.loc[mask, col]
        out[f"mean_{key}"] = safe_float(vals.mean()) if len(vals) else np.nan
        out[f"n_{key}"] = int(len(vals))

    out["n_ASD"] = int((tmp_basic["DX_GROUP"] == 1).sum())
    out["n_CTL"] = int((tmp_basic["DX_GROUP"] == 2).sum())
    out["n_male"] = int((tmp_basic["SEX"] == 1).sum())
    out["n_female"] = int((tmp_basic["SEX"] == 2).sum())
    out["n_total"] = int(len(tmp_basic))
    return out


def fit_one_model(tmp: pd.DataFrame, col: str, include_iq: bool, include_rh: bool):
    needed = [col, "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]
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

    cell_counts = (
        tmp_model.groupby(["DX_GROUP", "SEX"]).size().to_dict()
        if len(tmp_model) else {}
    )
    n_asd_male = int(cell_counts.get((1, 1), 0))
    n_asd_female = int(cell_counts.get((1, 2), 0))
    n_ctl_male = int(cell_counts.get((2, 1), 0))
    n_ctl_female = int(cell_counts.get((2, 2), 0))

    base = {
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
        base["note"] = "no_rows_after_dropna"
        return base

    if tmp_model["DX_GROUP"].nunique() != 2:
        base["note"] = "missing_dx_group_after_dropna"
        return base

    if tmp_model["SEX"].nunique() != 2:
        base["note"] = "missing_sex_group_after_dropna"
        return base

    if min(n_asd_male, n_asd_female, n_ctl_male, n_ctl_female) == 0:
        base["note"] = "missing_dx_sex_cell_after_dropna"
        return base

    if tmp_model[col].nunique(dropna=True) <= 1:
        base["beta_DX"] = 0.0
        base["beta_SEX"] = 0.0
        base["beta_DXxSEX"] = 0.0
        base["note"] = "constant_outcome"
        return base

    if include_iq and tmp_model["FIQ"].nunique(dropna=True) <= 1:
        base["note"] = "FIQ_not_variable_after_dropna"
        return base

    if include_rh and tmp_model["RIGHT_HANDED"].nunique(dropna=True) <= 1:
        base["note"] = "RIGHT_HANDED_not_variable_after_dropna"
        return base

    terms = ["C(DX_GROUP) * C(SEX)", "AGE_AT_SCAN"]

    use_site = tmp_model["SITE_ID"].nunique() >= 2
    if use_site:
        terms.append("C(SITE_ID)")
    else:
        base["note"] = "single_site_after_dropna"

    if include_iq:
        terms.append("FIQ")
    if include_rh:
        terms.append("C(RIGHT_HANDED)")

    formula = f"{col} ~ " + " + ".join(terms)

    try:
        model = smf.ols(formula, data=tmp_model).fit()

        base["beta_DX"] = safe_float(model.params.get(DX_TERM, np.nan))
        base["p_DX"] = safe_float(model.pvalues.get(DX_TERM, np.nan))

        base["beta_SEX"] = safe_float(model.params.get(SEX_TERM, np.nan))
        base["p_SEX"] = safe_float(model.pvalues.get(SEX_TERM, np.nan))

        base["beta_DXxSEX"] = safe_float(model.params.get(DXxSEX_TERM, np.nan))
        base["p_DXxSEX"] = safe_float(model.pvalues.get(DXxSEX_TERM, np.nan))

        if use_site:
            base["p_SITE"] = joint_term_pvalue(model, "C(SITE_ID)[T.")

        if include_iq:
            base["p_IQ"] = safe_float(model.pvalues.get("FIQ", np.nan))

        if include_rh:
            base["p_RIGHT_HANDED"] = joint_term_pvalue(model, "C(RIGHT_HANDED)[T.")

    except Exception as e:
        if base["note"]:
            base["note"] = f"{base['note']};model_fail:{type(e).__name__}"
        else:
            base["note"] = f"model_fail:{type(e).__name__}"

    return base


def run_one_age_group(df: pd.DataFrame, scenario: str, age_group: str, modules_present: list[int]):
    subdf = df[df["AGE_GROUP"] == age_group].copy()
    if subdf.empty:
        print(f"[SKIP] {scenario} | {age_group}: no rows")
        return

    required = {"SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID", "module", "node"}
    missing = required - set(subdf.columns)
    if missing:
        raise ValueError(f"{scenario} | {age_group}: missing columns: {sorted(missing)}")

    print(f"\n{scenario} | {age_group}")
    subj_counts = (
        subdf[["SUB_ID", "DX_GROUP", "SEX"]]
        .drop_duplicates()
        .groupby(["DX_GROUP", "SEX"])
        .size()
        .reset_index(name="n_subjects")
        .sort_values(["DX_GROUP", "SEX"])
    )
    print(subj_counts.to_string(index=False))

    wide_tables = {}
    used_metrics = []

    for metric_name, aliases in METRIC_ALIASES.items():
        metric_col = resolve_metric_column(subdf, aliases)
        if metric_col is None:
            print(f"[WARN] {scenario} | {age_group}: missing metric column for {metric_name}")
            continue

        wide_tables[metric_name] = build_subject_level_table(
            subdf=subdf,
            metric_name=metric_name,
            metric_col=metric_col,
            modules_present=modules_present,
        )
        used_metrics.append(metric_name)

    if not wide_tables:
        print(f"[SKIP] {scenario} | {age_group}: no usable metrics")
        return

    rows = []

    for metric_name in used_metrics:
        subj_df = wide_tables[metric_name]

        for module in modules_present:
            col = f"{metric_name}_M{module}"
            if col not in subj_df.columns:
                continue

            base_cols = [col, "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]
            if "FIQ" in subj_df.columns:
                base_cols.append("FIQ")
            if "RIGHT_HANDED" in subj_df.columns:
                base_cols.append("RIGHT_HANDED")

            tmp = subj_df[base_cols].copy()
            tmp_basic = tmp.dropna(subset=[col, "DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]).copy()

            row = {
                "scenario": scenario,
                "age_group": age_group,
                "metric": metric_name,
                "module": int(module),
            }
            row.update(compute_group_means_and_counts(tmp_basic, col))

            for model_key, cfg in MODELS.items():
                fitted = fit_one_model(
                    tmp=tmp,
                    col=col,
                    include_iq=cfg["include_iq"],
                    include_rh=cfg["include_rh"],
                )

                for k, v in fitted.items():
                    row[f"{k}_{model_key}"] = v

            rows.append(row)

    out_df = pd.DataFrame(rows)

    for model_key in MODELS:
        out_df[f"p_DXxSEX_FDR_{model_key}"] = np.nan
        out_df[f"DXxSEX_FDR_significant_{model_key}"] = False

        for metric_name in out_df["metric"].dropna().unique():
            mask = (out_df["metric"] == metric_name) & np.isfinite(out_df[f"p_DXxSEX_{model_key}"])
            pvals = out_df.loc[mask, f"p_DXxSEX_{model_key}"].to_numpy()

            if pvals.size:
                reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
                out_df.loc[mask, f"p_DXxSEX_FDR_{model_key}"] = p_fdr
                out_df.loc[mask, f"DXxSEX_FDR_significant_{model_key}"] = reject

    scenario_dir = OUT_DIR / scenario / age_group
    scenario_dir.mkdir(parents=True, exist_ok=True)
    out_path = scenario_dir / "module_stats_dxsex.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")


def main():
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {IN_DIR}")

    files = sorted(IN_DIR.glob("*_node_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No *_node_metrics.csv found in {IN_DIR}")

    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] input dir: {IN_DIR}")
    print(f"[INFO] output dir: {OUT_DIR}")
    print(f"[INFO] found {len(files)} node-metrics file(s)")

    for fp in files:
        scenario = fp.stem.replace("_node_metrics", "")
        df = pd.read_csv(fp)
        df.columns = [str(c).strip() for c in df.columns]

        numeric_cols = ["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "module", "node", "FIQ", "RIGHT_HANDED"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "SITE_ID" not in df.columns:
            raise ValueError(f"{fp.name} is missing SITE_ID")
        if "AGE_GROUP" not in df.columns:
            raise ValueError(f"{fp.name} is missing AGE_GROUP")

        df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()
        df["AGE_GROUP"] = df["AGE_GROUP"].astype(str).str.strip()

        df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "AGE_AT_SCAN", "module", "node"]).copy()
        df["SUB_ID"] = df["SUB_ID"].astype(int)
        df["DX_GROUP"] = df["DX_GROUP"].astype(int)
        df["SEX"] = df["SEX"].astype(int)
        df["module"] = df["module"].astype(int)
        df["node"] = df["node"].astype(int)

        modules_present = get_modules_present(df)
        print(f"\n[INFO] {scenario}: modules present = {modules_present}")

        for age_group in AGE_GROUPS:
            run_one_age_group(
                df=df,
                scenario=scenario,
                age_group=age_group,
                modules_present=modules_present,
            )

    print("\n[DONE] DXxSEX module stats written.")


if __name__ == "__main__":
    main()