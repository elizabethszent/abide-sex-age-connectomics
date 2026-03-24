import numpy as np
import pandas as pd
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        has_results = (p / "results").exists()
        has_data = (p / "data").exists() or (p / "phenotypes").exists()
        if has_results and has_data:
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

NODE_DIR = ROOT / "results" / "hubs" / "pc_z_strength_sitecov"
OUT_DIR = ROOT / "results" / "qc" / "cohens_d_dxsex_by_age"
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

FD02_MODULE_LABELS = {
    1: "M1 Somatomotor",
    2: "M2 Visual-A",
    3: "M3 DefaultMode",
    4: "M4 DorsalAttention",
    5: "M5 Visual-B",
    6: "M6 Frontoparietal",
    7: "M7 Limbic",
    8: "M8 VentralAttention",
}

FD03_MODULE_LABELS = {
    1: "M1 Somatomotor",
    2: "M2 Visual-A",
    3: "M3 Limbic",
    4: "M4 Frontoparietal",
    5: "M5 VentralAttention",
    6: "M6 Visual-B",
    7: "M7 DefaultMode",
    8: "M8 DorsalAttention",
}


def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def get_module_label_map(scenario: str):
    if "fd-0.2" in scenario:
        return FD02_MODULE_LABELS, "fd-0.2"
    if "fd-0.3" in scenario:
        return FD03_MODULE_LABELS, "fd-0.3"
    return {}, "unknown"


def get_module_label(scenario: str, module: int) -> str:
    mapping, _ = get_module_label_map(scenario)
    return mapping.get(module, f"M{module}")


def resolve_metric_column(df: pd.DataFrame, metric_name: str):
    aliases = METRIC_ALIASES.get(metric_name, [metric_name])
    for col in aliases:
        if col in df.columns:
            return col
    return None


def cohens_d(mean_ctl, mean_asd, sd_ctl, sd_asd, n_ctl, n_asd):
    if any(pd.isna(x) for x in [mean_ctl, mean_asd, sd_ctl, sd_asd, n_ctl, n_asd]):
        return np.nan
    if n_ctl < 2 or n_asd < 2:
        return np.nan

    pooled_var = (((n_ctl - 1) * (sd_ctl ** 2)) + ((n_asd - 1) * (sd_asd ** 2))) / (n_ctl + n_asd - 2)
    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return np.nan

    pooled_sd = np.sqrt(pooled_var)
    return (mean_ctl - mean_asd) / pooled_sd


def load_node_metrics(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [str(c).strip() for c in df.columns]

    numeric_cols = [
        "SUB_ID",
        "DX_GROUP",
        "SEX",
        "AGE_AT_SCAN",
        "module",
        "node",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "SITE_ID" in df.columns:
        df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()
    if "AGE_GROUP" in df.columns:
        df["AGE_GROUP"] = df["AGE_GROUP"].astype(str).str.strip()

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "module", "node"]).copy()
    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)
    df["module"] = df["module"].astype(int)
    df["node"] = df["node"].astype(int)

    return df


def build_subject_level_values(df: pd.DataFrame, age_group: str, metric: str, module: int) -> pd.DataFrame:
    metric_col = resolve_metric_column(df, metric)
    if metric_col is None:
        return pd.DataFrame()

    sub = df[
        (df["AGE_GROUP"] == age_group)
        & (df["module"] == module)
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

    subj = (
        sub.groupby(group_cols, dropna=False)[metric_col]
        .median()
        .reset_index(name="value")
    )

    return subj


def summarize_group(subj: pd.DataFrame, dx_val: int, sex_val: int):
    vals = subj.loc[(subj["DX_GROUP"] == dx_val) & (subj["SEX"] == sex_val), "value"]
    return {
        "n": int(len(vals)),
        "mean": safe_float(vals.mean()) if len(vals) else np.nan,
        "sd": safe_float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
        "median": safe_float(vals.median()) if len(vals) else np.nan,
        "min": safe_float(vals.min()) if len(vals) else np.nan,
        "max": safe_float(vals.max()) if len(vals) else np.nan,
    }


def evaluate_one_module(subj: pd.DataFrame, scenario: str, age_group: str, metric: str, module: int) -> dict:
    ctl_male = summarize_group(subj, dx_val=2, sex_val=1)
    asd_male = summarize_group(subj, dx_val=1, sex_val=1)
    ctl_female = summarize_group(subj, dx_val=2, sex_val=2)
    asd_female = summarize_group(subj, dx_val=1, sex_val=2)

    d_male = cohens_d(
        mean_ctl=ctl_male["mean"],
        mean_asd=asd_male["mean"],
        sd_ctl=ctl_male["sd"],
        sd_asd=asd_male["sd"],
        n_ctl=ctl_male["n"],
        n_asd=asd_male["n"],
    )
    d_female = cohens_d(
        mean_ctl=ctl_female["mean"],
        mean_asd=asd_female["mean"],
        sd_ctl=ctl_female["sd"],
        sd_asd=asd_female["sd"],
        n_ctl=ctl_female["n"],
        n_asd=asd_female["n"],
    )

    delta_d = d_female - d_male if pd.notna(d_female) and pd.notna(d_male) else np.nan
    abs_delta_d = abs(delta_d) if pd.notna(delta_d) else np.nan

    if pd.notna(d_male) and pd.notna(d_female):
        if d_male * d_female < 0:
            interaction_pattern_d = "opposite_directions"
        elif abs(d_female) > abs(d_male):
            interaction_pattern_d = "same_direction_stronger_in_female"
        elif abs(d_male) > abs(d_female):
            interaction_pattern_d = "same_direction_stronger_in_male"
        else:
            interaction_pattern_d = "same_direction_equal_magnitude"
    else:
        interaction_pattern_d = "unknown"

    if pd.notna(d_female) and pd.notna(d_male):
        if abs(d_female) > abs(d_male):
            stronger_in_d = "female"
        elif abs(d_male) > abs(d_female):
            stronger_in_d = "male"
        else:
            stronger_in_d = "tie"
    else:
        stronger_in_d = "unknown"

    _, mapping_threshold = get_module_label_map(scenario)

    return {
        "scenario": scenario,
        "mapping_threshold": mapping_threshold,
        "age_group": age_group,
        "metric": metric,
        "module": int(module),
        "module_label": get_module_label(scenario, int(module)),

        "n_CTL_male": ctl_male["n"],
        "mean_CTL_male": ctl_male["mean"],
        "sd_CTL_male": ctl_male["sd"],
        "median_CTL_male": ctl_male["median"],
        "min_CTL_male": ctl_male["min"],
        "max_CTL_male": ctl_male["max"],

        "n_ASD_male": asd_male["n"],
        "mean_ASD_male": asd_male["mean"],
        "sd_ASD_male": asd_male["sd"],
        "median_ASD_male": asd_male["median"],
        "min_ASD_male": asd_male["min"],
        "max_ASD_male": asd_male["max"],

        "n_CTL_female": ctl_female["n"],
        "mean_CTL_female": ctl_female["mean"],
        "sd_CTL_female": ctl_female["sd"],
        "median_CTL_female": ctl_female["median"],
        "min_CTL_female": ctl_female["min"],
        "max_CTL_female": ctl_female["max"],

        "n_ASD_female": asd_female["n"],
        "mean_ASD_female": asd_female["mean"],
        "sd_ASD_female": asd_female["sd"],
        "median_ASD_female": asd_female["median"],
        "min_ASD_female": asd_female["min"],
        "max_ASD_female": asd_female["max"],

        "cohens_d_male_CTL_minus_ASD": d_male,
        "cohens_d_female_CTL_minus_ASD": d_female,
        "delta_d_female_minus_male": delta_d,
        "abs_delta_d": abs_delta_d,
        "stronger_in_d": stronger_in_d,
        "interaction_pattern_d": interaction_pattern_d,
    }


def run():
    if not NODE_DIR.exists():
        raise FileNotFoundError(f"Node dir not found: {NODE_DIR}")

    node_files = sorted(NODE_DIR.glob("*_node_metrics.csv"))
    if not node_files:
        raise FileNotFoundError(f"No *_node_metrics.csv found in {NODE_DIR}")

    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] node dir: {NODE_DIR}")
    print(f"[INFO] output dir: {OUT_DIR}")
    print(f"[INFO] found {len(node_files)} node-metrics file(s)")

    all_rows = []

    for fp in node_files:
        scenario = fp.stem.replace("_node_metrics", "")
        df = load_node_metrics(fp)
        modules_present = sorted(df["module"].dropna().astype(int).unique().tolist())
        _, mapping_threshold = get_module_label_map(scenario)

        print(f"\n[SCENARIO] {scenario} | modules={modules_present}")

        scenario_rows = []

        for age_group in AGE_GROUPS:
            sub_age = df[df["AGE_GROUP"] == age_group].copy()
            if sub_age.empty:
                continue

            for metric in METRIC_ALIASES.keys():
                metric_col = resolve_metric_column(sub_age, metric)
                if metric_col is None:
                    continue

                for module in modules_present:
                    subj = build_subject_level_values(
                        df=df,
                        age_group=age_group,
                        metric=metric,
                        module=module,
                    )
                    if subj.empty:
                        continue

                    row = evaluate_one_module(
                        subj=subj,
                        scenario=scenario,
                        age_group=age_group,
                        metric=metric,
                        module=module,
                    )
                    scenario_rows.append(row)
                    all_rows.append(row)

        scenario_df = pd.DataFrame(scenario_rows)
        if not scenario_df.empty:
            scenario_dir = OUT_DIR / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)

            scenario_path = scenario_dir / "dxsex_cohens_d_by_age.csv"
            scenario_df.to_csv(scenario_path, index=False)

            ranked_path = scenario_dir / "dxsex_cohens_d_by_age__ranked_by_abs_delta_d.csv"
            scenario_df.sort_values("abs_delta_d", ascending=False).to_csv(ranked_path, index=False)

            print(f"[SAVED] {scenario_path}")
            print(f"[SAVED] {ranked_path}")

    out_df = pd.DataFrame(all_rows)
    if out_df.empty:
        raise RuntimeError("No rows were created.")

    overall_path = OUT_DIR / "dxsex_cohens_d_by_age__all_scenarios.csv"
    out_df.to_csv(overall_path, index=False)

    overall_ranked = OUT_DIR / "dxsex_cohens_d_by_age__all_scenarios__ranked_by_abs_delta_d.csv"
    out_df.sort_values("abs_delta_d", ascending=False).to_csv(overall_ranked, index=False)

    female_stronger = OUT_DIR / "dxsex_cohens_d_by_age__female_stronger.csv"
    out_df[out_df["stronger_in_d"] == "female"].sort_values(
        "abs_delta_d", ascending=False
    ).to_csv(female_stronger, index=False)

    male_stronger = OUT_DIR / "dxsex_cohens_d_by_age__male_stronger.csv"
    out_df[out_df["stronger_in_d"] == "male"].sort_values(
        "abs_delta_d", ascending=False
    ).to_csv(male_stronger, index=False)

    print(f"\n[SAVED] {overall_path}")
    print(f"[SAVED] {overall_ranked}")
    print(f"[SAVED] {female_stronger}")
    print(f"[SAVED] {male_stronger}")

    show_cols = [
        "scenario",
        "age_group",
        "metric",
        "module",
        "module_label",
        "cohens_d_male_CTL_minus_ASD",
        "cohens_d_female_CTL_minus_ASD",
        "delta_d_female_minus_male",
        "abs_delta_d",
        "stronger_in_d",
        "interaction_pattern_d",
    ]
    print("\n[TOP 30 by |delta d|]")
    print(
        out_df.sort_values("abs_delta_d", ascending=False)[show_cols]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run()