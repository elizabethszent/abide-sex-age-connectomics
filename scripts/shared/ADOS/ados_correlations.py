import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "results").exists() and ((p / "data").exists() or (p / "phenotypes").exists()):
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)


MIN_N = 6
ASD_ONLY = True
REQUIRE_ADOS_RELIABLE = False

ADOS_COL = "ADOS_TOTAL"
# ADOS_COL = "ADOS_GOTHAM_TOTAL"
# ADOS_COL = "ADOS_GOTHAM_SEVERITY"

AGE_ORDER = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
SEX_LABELS = {1: "male", 2: "female"}

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
METRIC_ORDER = list(METRIC_ALIASES.keys())

NODE_DIR = ROOT / "results" / "hubs" / "pc_z_strength_sitecov"
META_FD02 = ROOT / "data" / "metadata" / "ABIDE12_phenotypes_combined_fd_0p2.csv"
META_FD03 = ROOT / "data" / "metadata" / "ABIDE12_phenotypes_combined_fd_0p3.csv"

OUT_DIR = ROOT / "results" / "qc" / "ados_correlations_all_modules"
OUT_DIR.mkdir(parents=True, exist_ok=True)



#helper
def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(s).strip())


def resolve_metric_column(df: pd.DataFrame, metric_name: str) -> str | None:
    aliases = METRIC_ALIASES.get(metric_name, [metric_name])
    for col in aliases:
        if col in df.columns:
            return col
    return None


def load_metadata_for_scenario(scenario: str) -> pd.DataFrame:
    if "fd-0.2" in scenario:
        fp = META_FD02
    elif "fd-0.3" in scenario:
        fp = META_FD03
    else:
        raise ValueError(f"Could not infer metadata file from scenario: {scenario}")

    if not fp.exists():
        raise FileNotFoundError(f"Metadata file not found: {fp}")

    df = pd.read_csv(fp)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"SUB_ID", ADOS_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{fp.name} missing required columns: {sorted(missing)}")

    df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce")
    df = df.dropna(subset=["SUB_ID"]).copy()
    df["SUB_ID"] = df["SUB_ID"].astype(int)

    for c in [
        "DX_GROUP", "SEX", "FIQ", "RIGHT_HANDED",
        "ADOS_TOTAL", "ADOS_COMM", "ADOS_SOCIAL", "ADOS_STEREO_BEHAV",
        "ADOS_GOTHAM_TOTAL", "ADOS_GOTHAM_SEVERITY", "ADOS_RSRCH_RELIABLE",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "AGE_GROUP" in df.columns:
        df["AGE_GROUP"] = df["AGE_GROUP"].astype(str).str.strip()

    keep_cols = ["SUB_ID", ADOS_COL]
    for c in [
        "DX_GROUP", "SEX", "AGE_GROUP", "SITE_ID", "FIQ", "RIGHT_HANDED",
        "ADOS_TOTAL", "ADOS_COMM", "ADOS_SOCIAL", "ADOS_STEREO_BEHAV",
        "ADOS_GOTHAM_TOTAL", "ADOS_GOTHAM_SEVERITY", "ADOS_RSRCH_RELIABLE",
    ]:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    return df[keep_cols].copy()


def load_node_metrics(scenario: str) -> pd.DataFrame:
    fp = NODE_DIR / f"{scenario}_node_metrics.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Node metrics file not found: {fp}")

    df = pd.read_csv(fp)
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["SUB_ID", "DX_GROUP", "SEX", "module", "node"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "AGE_GROUP" in df.columns:
        df["AGE_GROUP"] = df["AGE_GROUP"].astype(str).str.strip()

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "module", "node"]).copy()
    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)
    df["module"] = df["module"].astype(int)
    df["node"] = df["node"].astype(int)

    return df


def get_scenarios() -> list[str]:
    files = sorted(NODE_DIR.glob("*_node_metrics.csv"))
    return [fp.stem.replace("_node_metrics", "") for fp in files]


def get_metrics_present(node_df: pd.DataFrame) -> list[str]:
    metrics = []
    for metric in METRIC_ORDER:
        if resolve_metric_column(node_df, metric) is not None:
            metrics.append(metric)
    return metrics


def get_modules_present(node_df: pd.DataFrame) -> list[int]:
    return sorted(node_df["module"].dropna().astype(int).unique().tolist())


def print_ados_coverage(meta_df: pd.DataFrame, scenario: str):
    df = meta_df.copy()

    if ASD_ONLY:
        df = df[df["DX_GROUP"] == 1].copy()

    if REQUIRE_ADOS_RELIABLE and "ADOS_RSRCH_RELIABLE" in df.columns:
        df = df[df["ADOS_RSRCH_RELIABLE"] == 1].copy()

    df = df[df[ADOS_COL].notna()].copy()

    if df.empty:
        print(f"\n[ADOS COVERAGE] {scenario}: no rows after filters")
        return

    df["sex_label"] = df["SEX"].map(SEX_LABELS).fillna("unknown")

    counts = (
        df.groupby(["sex_label", "AGE_GROUP"])
        .size()
        .reset_index(name="n")
        .sort_values(["sex_label", "AGE_GROUP"])
    )

    print(f"\n[ADOS COVERAGE] {scenario} | {ADOS_COL}")
    print(counts.to_string(index=False))


def build_subject_level_metric(
    node_df: pd.DataFrame,
    metric: str,
    module: int,
    age_group: str,
    sex: str,
) -> pd.DataFrame:
    metric_col = resolve_metric_column(node_df, metric)
    if metric_col is None:
        return pd.DataFrame()

    sex_code = 1 if sex == "male" else 2

    sub = node_df[
        (node_df["module"] == module)
        & (node_df["AGE_GROUP"] == age_group)
        & (node_df["SEX"] == sex_code)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    group_cols = ["SUB_ID", "DX_GROUP", "SEX", "AGE_GROUP"]
    for c in ["SITE_ID", "AGE_AT_SCAN", "FIQ", "RIGHT_HANDED"]:
        if c in sub.columns:
            group_cols.append(c)

    subj = (
        sub.groupby(group_cols, dropna=False)[metric_col]
        .median()
        .reset_index(name="module_value")
    )

    return subj


def filter_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if ASD_ONLY:
        out = out[out["DX_GROUP"] == 1].copy()

    if REQUIRE_ADOS_RELIABLE and "ADOS_RSRCH_RELIABLE" in out.columns:
        out = out[out["ADOS_RSRCH_RELIABLE"] == 1].copy()

    out = out.dropna(subset=[ADOS_COL]).copy()
    return out


def run_corr(x: pd.Series, y: pd.Series):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    keep = x.notna() & y.notna()
    x = x[keep]
    y = y[keep]

    if len(x) < MIN_N:
        return {
            "n": int(len(x)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    try:
        pr, pp = pearsonr(x, y)
    except Exception:
        pr, pp = np.nan, np.nan

    try:
        sr, sp = spearmanr(x, y)
    except Exception:
        sr, sp = np.nan, np.nan

    return {
        "n": int(len(x)),
        "pearson_r": safe_float(pr),
        "pearson_p": safe_float(pp),
        "spearman_rho": safe_float(sr),
        "spearman_p": safe_float(sp),
    }


def make_scatter(df: pd.DataFrame, title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_numeric(df[ADOS_COL], errors="coerce")
    y = pd.to_numeric(df["module_value"], errors="coerce")
    keep = x.notna() & y.notna()
    x = x[keep]
    y = y[keep]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(x, y, s=28, alpha=0.85)

    if len(x) >= 2:
        try:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(float(x.min()), float(x.max()), 100)
            yy = coef[0] * xx + coef[1]
            ax.plot(xx, yy, linewidth=1.8)
        except Exception:
            pass

    ax.set_xlabel(ADOS_COL)
    ax.set_ylabel("Subject-level module median")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_fdr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # stricter: within scenario x sex across all metrics/modules/ages
    df["spearman_p_fdr_by_scenario_sex"] = np.nan
    df["spearman_fdr_sig_by_scenario_sex"] = False

    for (scenario, sex), idx in df.groupby(["scenario", "sex"]).groups.items():
        mask = df.index.isin(idx) & df["spearman_p"].notna()
        if mask.sum() == 0:
            continue
        pvals = df.loc[mask, "spearman_p"].to_numpy()
        reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        df.loc[mask, "spearman_p_fdr_by_scenario_sex"] = p_adj
        df.loc[mask, "spearman_fdr_sig_by_scenario_sex"] = reject

    # preferred: within scenario x sex x metric across modules/ages
    df["spearman_p_fdr_by_scenario_sex_metric"] = np.nan
    df["spearman_fdr_sig_by_scenario_sex_metric"] = False

    for (scenario, sex, metric), idx in df.groupby(["scenario", "sex", "metric"]).groups.items():
        mask = df.index.isin(idx) & df["spearman_p"].notna()
        if mask.sum() == 0:
            continue
        pvals = df.loc[mask, "spearman_p"].to_numpy()
        reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        df.loc[mask, "spearman_p_fdr_by_scenario_sex_metric"] = p_adj
        df.loc[mask, "spearman_fdr_sig_by_scenario_sex_metric"] = reject

    return df


# main
def main():
    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] ADOS column: {ADOS_COL}")
    print(f"[INFO] ASD_ONLY: {ASD_ONLY}")
    print(f"[INFO] REQUIRE_ADOS_RELIABLE: {REQUIRE_ADOS_RELIABLE}")
    print(f"[INFO] output dir: {OUT_DIR}")

    scenarios = get_scenarios()
    if not scenarios:
        raise RuntimeError(f"No node metric files found in {NODE_DIR}")

    node_cache = {}
    meta_cache = {}
    rows = []
    registry_rows = []

    for scenario in scenarios:
        print(f"\n[SCENARIO] {scenario}")

        node_df = load_node_metrics(scenario)
        meta_df = load_metadata_for_scenario(scenario)

        node_cache[scenario] = node_df
        meta_cache[scenario] = meta_df

        print_ados_coverage(meta_df, scenario)

        metrics_present = get_metrics_present(node_df)
        modules_present = get_modules_present(node_df)

        combos = 0
        for age_group in AGE_ORDER:
            for sex in ["female", "male"]:
                for metric in metrics_present:
                    for module in modules_present:
                        combos += 1
                        registry_rows.append({
                            "scenario": scenario,
                            "age_group": age_group,
                            "sex": sex,
                            "metric": metric,
                            "module": module,
                        })

                        subj = build_subject_level_metric(
                            node_df=node_df,
                            metric=metric,
                            module=module,
                            age_group=age_group,
                            sex=sex,
                        )

                        if subj.empty:
                            rows.append({
                                "scenario": scenario,
                                "age_group": age_group,
                                "sex": sex,
                                "metric": metric,
                                "module": module,
                                "ados_col": ADOS_COL,
                                "asd_only": ASD_ONLY,
                                "require_ados_reliable": REQUIRE_ADOS_RELIABLE,
                                "n": 0,
                                "pearson_r": np.nan,
                                "pearson_p": np.nan,
                                "spearman_rho": np.nan,
                                "spearman_p": np.nan,
                                "note": "no_subject_level_rows",
                                "subject_csv": "",
                                "scatter_png": "",
                            })
                            continue

                        merged = subj.merge(meta_df, on="SUB_ID", how="left", suffixes=("", "_meta"))
                        merged = filter_for_analysis(merged)

                        title = f"{scenario} | {age_group} | {sex} | {metric} | M{module}"
                        out_dir = OUT_DIR / scenario / age_group / sex / metric / f"module_{module}"
                        out_dir.mkdir(parents=True, exist_ok=True)

                        subject_csv = out_dir / "subject_level_with_ados.csv"
                        scatter_png = out_dir / "scatter.png"

                        merged.to_csv(subject_csv, index=False)
                        make_scatter(merged, title, scatter_png)

                        corr = run_corr(merged[ADOS_COL], merged["module_value"])

                        rows.append({
                            "scenario": scenario,
                            "age_group": age_group,
                            "sex": sex,
                            "metric": metric,
                            "module": module,
                            "ados_col": ADOS_COL,
                            "asd_only": ASD_ONLY,
                            "require_ados_reliable": REQUIRE_ADOS_RELIABLE,
                            **corr,
                            "note": "",
                            "subject_csv": str(subject_csv),
                            "scatter_png": str(scatter_png),
                        })

        print(f"[INFO] all-module combinations: {combos}")

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError("No ADOS correlation results were created.")

    out_df = add_fdr(out_df)

    out_df = out_df.sort_values(
        ["scenario", "age_group", "sex", "metric", "module"]
    ).reset_index(drop=True)

    registry_df = pd.DataFrame(registry_rows)

    out_csv = OUT_DIR / f"ados_all_modules__{ADOS_COL}.csv"
    ranked_csv = OUT_DIR / f"ados_all_modules__{ADOS_COL}__ranked.csv"
    registry_csv = OUT_DIR / f"ados_all_modules__{ADOS_COL}__registry.csv"
    fdr_hits_csv = OUT_DIR / f"ados_all_modules__{ADOS_COL}__fdr_hits.csv"

    out_df.to_csv(out_csv, index=False)
    registry_df.to_csv(registry_csv, index=False)

    out_df.assign(abs_spearman=out_df["spearman_rho"].abs()).sort_values(
        "abs_spearman", ascending=False
    ).to_csv(ranked_csv, index=False)

    out_df[
        (out_df["spearman_fdr_sig_by_scenario_sex_metric"] == True) |
        (out_df["spearman_fdr_sig_by_scenario_sex"] == True)
    ].to_csv(fdr_hits_csv, index=False)

    print(f"\n[SAVED] {out_csv}")
    print(f"[SAVED] {ranked_csv}")
    print(f"[SAVED] {registry_csv}")
    print(f"[SAVED] {fdr_hits_csv}")

    show_cols = [
        "scenario",
        "age_group",
        "sex",
        "metric",
        "module",
        "n",
        "spearman_rho",
        "spearman_p",
        "spearman_p_fdr_by_scenario_sex_metric",
        "spearman_fdr_sig_by_scenario_sex_metric",
        "spearman_p_fdr_by_scenario_sex",
        "spearman_fdr_sig_by_scenario_sex",
    ]
    print("\n[TOP 30 by |Spearman rho|]")
    print(
        out_df.assign(abs_spearman=out_df["spearman_rho"].abs())
        .sort_values("abs_spearman", ascending=False)[show_cols]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()