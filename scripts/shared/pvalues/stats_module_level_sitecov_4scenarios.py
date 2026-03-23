import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "results").exists() and (p / "data").exists():
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

IN_DIR = ROOT / "results" / "hubs" / "pc_z_strength_sitecov"
OUT_DIR = ROOT / "results" / "hubs" / "module_stats_sitecov"
OUT_DIR.mkdir(exist_ok=True, parents=True)

SEXES = ["female", "male"]
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


def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def first_nonnull(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def add_sex_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sex_label"] = np.where(
        df["SEX"] == 2,
        "female",
        np.where(df["SEX"] == 1, "male", "unknown"),
    )
    return df


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
    colname: str,
    modules_present: list[int],
) -> pd.DataFrame:
    group_cols = ["SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SITE_ID"]

    wide = (
        subdf.groupby(group_cols + ["module"])[colname]
        .median()
        .unstack("module")
    )

    for m in modules_present:
        if m not in wide.columns:
            wide[m] = np.nan

    wide = wide[modules_present]
    wide.columns = [f"{metric_name}_M{int(m)}" for m in wide.columns]
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


def fit_one_model(tmp: pd.DataFrame, col: str, include_iq: bool, include_rh: bool):
    beta = np.nan
    p_dx = np.nan
    p_site = np.nan
    p_iq = np.nan
    p_right_handed = np.nan
    note = ""

    needed = [col, "DX_GROUP", "AGE_AT_SCAN", "SITE_ID"]
    if include_iq:
        needed.append("FIQ")
    if include_rh:
        needed.append("RIGHT_HANDED")

    tmp_model = tmp[needed].dropna().copy()

    n_model = int(len(tmp_model))
    n_model_asd = int((tmp_model["DX_GROUP"] == 1).sum())
    n_model_ctl = int((tmp_model["DX_GROUP"] == 2).sum())

    if n_model == 0:
        return beta, p_dx, p_site, p_iq, p_right_handed, "no_rows_after_dropna", n_model, n_model_asd, n_model_ctl

    if tmp_model["DX_GROUP"].nunique() != 2:
        return beta, p_dx, p_site, p_iq, p_right_handed, "missing_group_after_dropna", n_model, n_model_asd, n_model_ctl

    if tmp_model[col].nunique(dropna=True) <= 1:
        return 0.0, np.nan, np.nan, np.nan, np.nan, "constant_outcome", n_model, n_model_asd, n_model_ctl

    if include_iq and tmp_model["FIQ"].nunique(dropna=True) <= 1:
        return beta, p_dx, p_site, p_iq, p_right_handed, "FIQ_not_variable", n_model, n_model_asd, n_model_ctl

    if include_rh and tmp_model["RIGHT_HANDED"].nunique(dropna=True) <= 1:
        return beta, p_dx, p_site, p_iq, p_right_handed, "RIGHT_HANDED_not_variable", n_model, n_model_asd, n_model_ctl

    terms = ["C(DX_GROUP)", "AGE_AT_SCAN"]

    use_site = tmp_model["SITE_ID"].nunique() >= 2
    if use_site:
        terms.append("C(SITE_ID)")
    else:
        note = "single_site_after_dropna"

    if include_iq:
        terms.append("FIQ")

    if include_rh:
        terms.append("C(RIGHT_HANDED)")

    formula = f"{col} ~ " + " + ".join(terms)

    try:
        model = smf.ols(formula, data=tmp_model).fit()

        beta = safe_float(model.params.get("C(DX_GROUP)[T.2]", np.nan))
        p_dx = safe_float(model.pvalues.get("C(DX_GROUP)[T.2]", np.nan))

        if use_site:
            p_site = joint_term_pvalue(model, "C(SITE_ID)[T.")

        if include_iq:
            p_iq = safe_float(model.pvalues.get("FIQ", np.nan))

        if include_rh:
            p_right_handed = joint_term_pvalue(model, "C(RIGHT_HANDED)[T.")

    except Exception as e:
        note = f"model_fail:{type(e).__name__}" if note == "" else f"{note};model_fail:{type(e).__name__}"

    return beta, p_dx, p_site, p_iq, p_right_handed, note, n_model, n_model_asd, n_model_ctl


def run_one_group(df: pd.DataFrame, scenario: str, sex: str, age_group: str, modules_present: list[int]):
    subdf = df[(df["sex_label"] == sex) & (df["AGE_GROUP"] == age_group)].copy()
    if subdf.empty:
        print(f"[SKIP] {scenario} | {sex} | {age_group}: no rows")
        return

    required = {"SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SITE_ID", "module", "node"}
    if not required.issubset(subdf.columns):
        missing = required - set(subdf.columns)
        raise ValueError(f"{scenario}: missing required columns: {missing}")

    subj_counts = (
        subdf[["SUB_ID", "DX_GROUP"]]
        .drop_duplicates()["DX_GROUP"]
        .value_counts()
        .rename({1: "ASD", 2: "CTL"})
        .to_dict()
    )

    print(f"\n{scenario} | {sex.upper()} | {age_group.upper()}")
    print("  Subjects by DX:", subj_counts)

    wide_tables = {}
    used_metrics = []

    for metric_name, aliases in METRIC_ALIASES.items():
        colname = resolve_metric_column(subdf, aliases)
        if colname is None:
            print(f"  [WARN] missing column for {metric_name}, skipping")
            continue

        subj_df = build_subject_level_table(subdf, metric_name, colname, modules_present)
        wide_tables[metric_name] = subj_df
        used_metrics.append(metric_name)

    if not wide_tables:
        print("  No usable metrics found.")
        return

    results = []

    for metric_name in used_metrics:
        subj_df = wide_tables[metric_name]

        for m in modules_present:
            col = f"{metric_name}_M{m}"
            if col not in subj_df.columns:
                continue

            base_cols = [col, "DX_GROUP", "AGE_AT_SCAN", "SITE_ID"]
            if "FIQ" in subj_df.columns:
                base_cols.append("FIQ")
            if "RIGHT_HANDED" in subj_df.columns:
                base_cols.append("RIGHT_HANDED")

            tmp = subj_df[base_cols].copy()
            tmp_basic = tmp.dropna(subset=[col, "DX_GROUP", "AGE_AT_SCAN", "SITE_ID"]).copy()

            n_asd = int((tmp_basic["DX_GROUP"] == 1).sum())
            n_ctl = int((tmp_basic["DX_GROUP"] == 2).sum())

            mean_asd = safe_float(tmp_basic.loc[tmp_basic["DX_GROUP"] == 1, col].mean()) if n_asd else np.nan
            mean_ctl = safe_float(tmp_basic.loc[tmp_basic["DX_GROUP"] == 2, col].mean()) if n_ctl else np.nan

            m1 = fit_one_model(tmp, col, include_iq=False, include_rh=False)
            m2 = fit_one_model(tmp, col, include_iq=True, include_rh=False)
            m3 = fit_one_model(tmp, col, include_iq=True, include_rh=True)

            results.append(
                {
                    "scenario": scenario,
                    "sex": sex,
                    "age_group": age_group,
                    "metric": metric_name,
                    "module": m,
                    "mean_ASD": mean_asd,
                    "mean_CTL": mean_ctl,
                    "n_ASD": n_asd,
                    "n_CTL": n_ctl,

                    "beta_CTL_minus_ASD_m1": m1[0],
                    "p_DX_m1": m1[1],
                    "p_SITE_m1": m1[2],
                    "p_IQ_m1": np.nan,
                    "p_RIGHT_HANDED_m1": np.nan,
                    "n_model_m1": m1[6],
                    "n_model_ASD_m1": m1[7],
                    "n_model_CTL_m1": m1[8],
                    "note_m1": m1[5],

                    "beta_CTL_minus_ASD_m2": m2[0],
                    "p_DX_m2": m2[1],
                    "p_SITE_m2": m2[2],
                    "p_IQ_m2": m2[3],
                    "p_RIGHT_HANDED_m2": np.nan,
                    "n_model_m2": m2[6],
                    "n_model_ASD_m2": m2[7],
                    "n_model_CTL_m2": m2[8],
                    "note_m2": m2[5],

                    "beta_CTL_minus_ASD_m3": m3[0],
                    "p_DX_m3": m3[1],
                    "p_SITE_m3": m3[2],
                    "p_IQ_m3": m3[3],
                    "p_RIGHT_HANDED_m3": m3[4],
                    "n_model_m3": m3[6],
                    "n_model_ASD_m3": m3[7],
                    "n_model_CTL_m3": m3[8],
                    "note_m3": m3[5],
                }
            )

    stats_df = pd.DataFrame(results)

    for suffix in ["m1", "m2", "m3"]:
        stats_df[f"p_DX_FDR_{suffix}"] = np.nan
        stats_df[f"DX_FDR_significant_{suffix}"] = False

        for metric_name in stats_df["metric"].dropna().unique():
            mask = (stats_df["metric"] == metric_name) & np.isfinite(stats_df[f"p_DX_{suffix}"])
            pvals = stats_df.loc[mask, f"p_DX_{suffix}"].to_numpy()

            if pvals.size > 0:
                reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
                stats_df.loc[mask, f"p_DX_FDR_{suffix}"] = p_fdr
                stats_df.loc[mask, f"DX_FDR_significant_{suffix}"] = reject

    out_path = OUT_DIR / f"{scenario}__{sex}__{age_group}__module_stats_sitecov.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")


def main():
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {IN_DIR}")

    files = sorted(IN_DIR.glob("*_node_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No *_node_metrics.csv found in {IN_DIR}")

    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] input dir: {IN_DIR}")
    print(f"[INFO] output dir: {OUT_DIR}")
    print(f"[INFO] found {len(files)} node-metric file(s)")

    for fp in files:
        scenario = fp.stem.replace("_node_metrics", "")
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip()

        df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
        df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
        df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
        df["module"] = pd.to_numeric(df["module"], errors="coerce")
        df["node"] = pd.to_numeric(df["node"], errors="coerce")
        df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

        if "FIQ" in df.columns:
            df["FIQ"] = pd.to_numeric(df["FIQ"], errors="coerce")
        else:
            df["FIQ"] = np.nan

        if "RIGHT_HANDED" in df.columns:
            df["RIGHT_HANDED"] = pd.to_numeric(df["RIGHT_HANDED"], errors="coerce")
        else:
            df["RIGHT_HANDED"] = np.nan

        df = df.dropna(subset=["SEX", "DX_GROUP", "AGE_AT_SCAN", "module", "node"]).copy()
        df["SEX"] = df["SEX"].astype(int)
        df["DX_GROUP"] = df["DX_GROUP"].astype(int)
        df["module"] = df["module"].astype(int)
        df["node"] = df["node"].astype(int)

        df = add_sex_label(df)

        if "AGE_GROUP" not in df.columns:
            raise ValueError(
                f"{fp.name} is missing AGE_GROUP. Re-run the metric export so it includes AGE_GROUP."
            )

        df["AGE_GROUP"] = df["AGE_GROUP"].astype(str).str.strip()

        modules_present = get_modules_present(df)
        print(f"\n[INFO] {scenario}: modules present = {modules_present}")

        for sex in SEXES:
            for age_group in AGE_GROUPS:
                run_one_group(df, scenario, sex, age_group, modules_present)

    print("\n[DONE] nested module stats rebuilt.")


if __name__ == "__main__":
    main()