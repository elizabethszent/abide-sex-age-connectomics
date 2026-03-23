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

IN_DIR = ROOT / "results" / "hubs" / "module_stats_dxsex"
OUT_DIR = ROOT / "results" / "qc" / "dxsex_sig_hits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "m1": "site",
    "m2": "site_iq",
    "m3": "site_iq_rh",
}

NOMINAL_ALPHA = 0.05


def safe_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "1", "yes"}


def safe_float(x):
    try:
        x = float(x)
        return x
    except Exception:
        return float("nan")


def build_expected_table_path(scenario: str, age_group: str, metric: str, model_label: str, sex: str) -> str:
    p = (
        ROOT
        / "results"
        / "hubs"
        / "module_stats_dxsex_tables"
        / scenario
        / age_group
        / metric
        / model_label
        / sex
        / "table.csv"
    )
    return str(p)


def classify_interaction_pattern(dx_male: float, dx_female: float) -> str:
    if pd.isna(dx_male) or pd.isna(dx_female):
        return "unknown"

    if dx_male == 0 and dx_female == 0:
        return "both_zero"

    if dx_male * dx_female < 0:
        return "opposite_directions"

    if abs(dx_male) > abs(dx_female):
        return "same_direction_stronger_in_male"

    if abs(dx_female) > abs(dx_male):
        return "same_direction_stronger_in_female"

    return "same_direction_equal_magnitude"


def stronger_in_sex(dx_male: float, dx_female: float) -> str:
    if pd.isna(dx_male) or pd.isna(dx_female):
        return "unknown"
    if abs(dx_male) > abs(dx_female):
        return "male"
    if abs(dx_female) > abs(dx_male):
        return "female"
    return "tie"


def extract_hits_from_file(fp: Path):
    df = pd.read_csv(fp)
    df.columns = [str(c).strip() for c in df.columns]

    scenario = fp.parent.parent.name
    age_group = fp.parent.name

    rows_fdr = []
    rows_nominal = []

    for _, row in df.iterrows():
        metric = str(row.get("metric", "")).strip()
        module = row.get("module", pd.NA)

        mean_asd_male = safe_float(row.get("mean_ASD_male"))
        mean_ctl_male = safe_float(row.get("mean_CTL_male"))
        mean_asd_female = safe_float(row.get("mean_ASD_female"))
        mean_ctl_female = safe_float(row.get("mean_CTL_female"))

        dx_effect_male = mean_ctl_male - mean_asd_male if pd.notna(mean_ctl_male) and pd.notna(mean_asd_male) else float("nan")
        dx_effect_female = mean_ctl_female - mean_asd_female if pd.notna(mean_ctl_female) and pd.notna(mean_asd_female) else float("nan")

        driving_sex = stronger_in_sex(dx_effect_male, dx_effect_female)
        pattern = classify_interaction_pattern(dx_effect_male, dx_effect_female)

        for model_key, model_label in MODELS.items():
            beta_col = f"beta_DXxSEX_{model_key}"
            p_col = f"p_DXxSEX_{model_key}"
            p_fdr_col = f"p_DXxSEX_FDR_{model_key}"
            sig_col = f"DXxSEX_FDR_significant_{model_key}"
            note_col = f"note_{model_key}"

            beta = pd.to_numeric(pd.Series([row.get(beta_col)]), errors="coerce").iloc[0]
            p_val = pd.to_numeric(pd.Series([row.get(p_col)]), errors="coerce").iloc[0]
            p_fdr = pd.to_numeric(pd.Series([row.get(p_fdr_col)]), errors="coerce").iloc[0]
            sig_fdr = safe_bool(row.get(sig_col))
            note = row.get(note_col, "")

            common = {
                "scenario": scenario,
                "age_group": age_group,
                "metric": metric,
                "module": module,
                "model_key": model_key,
                "model_label": model_label,
                "beta_DXxSEX": beta,
                "p_DXxSEX": p_val,
                "p_DXxSEX_FDR": p_fdr,
                "DXxSEX_FDR_significant": sig_fdr,

                "mean_ASD_male": mean_asd_male,
                "mean_CTL_male": mean_ctl_male,
                "mean_ASD_female": mean_asd_female,
                "mean_CTL_female": mean_ctl_female,
                "dx_effect_male": dx_effect_male,
                "dx_effect_female": dx_effect_female,
                "stronger_in_sex": driving_sex,
                "interaction_pattern": pattern,

                "note": note,
                "source_stats_path": str(fp),
                "expected_table_path_male": build_expected_table_path(scenario, age_group, metric, model_label, "male"),
                "expected_table_path_female": build_expected_table_path(scenario, age_group, metric, model_label, "female"),
            }

            if pd.notna(p_val) and p_val < NOMINAL_ALPHA:
                rows_nominal.append(common)

            if sig_fdr:
                rows_fdr.append(common)

    return rows_fdr, rows_nominal


def main():
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {IN_DIR}")

    files = sorted(IN_DIR.glob("*/*/module_stats_dxsex.csv"))
    if not files:
        raise FileNotFoundError(f"No module_stats_dxsex.csv files found under {IN_DIR}")

    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] input dir: {IN_DIR}")
    print(f"[INFO] output dir: {OUT_DIR}")
    print(f"[INFO] found {len(files)} stats file(s)")

    all_fdr = []
    all_nominal = []

    for fp in files:
        fdr_rows, nominal_rows = extract_hits_from_file(fp)
        all_fdr.extend(fdr_rows)
        all_nominal.extend(nominal_rows)

    fdr_df = pd.DataFrame(all_fdr)
    nominal_df = pd.DataFrame(all_nominal)

    sort_cols = ["scenario", "age_group", "metric", "model_key", "module"]

    if not fdr_df.empty:
        fdr_df = fdr_df.sort_values(sort_cols).reset_index(drop=True)
    if not nominal_df.empty:
        nominal_df = nominal_df.sort_values(sort_cols).reset_index(drop=True)

    fdr_out = OUT_DIR / "dxsex_fdr_significant_hits.csv"
    nominal_out = OUT_DIR / "dxsex_nominal_hits_p_lt_0p05.csv"

    fdr_df.to_csv(fdr_out, index=False)
    nominal_df.to_csv(nominal_out, index=False)

    print(f"[SAVED] {fdr_out}")
    print(f"[SAVED] {nominal_out}")

    for model_key, model_label in MODELS.items():
        if not fdr_df.empty:
            sub_fdr = fdr_df[fdr_df["model_key"] == model_key].copy()
            sub_fdr.to_csv(OUT_DIR / f"dxsex_fdr_hits__{model_label}.csv", index=False)

        if not nominal_df.empty:
            sub_nom = nominal_df[nominal_df["model_key"] == model_key].copy()
            sub_nom.to_csv(OUT_DIR / f"dxsex_nominal_hits__{model_label}.csv", index=False)

    print("\n[SUMMARY]")
    print(f"FDR-significant DXxSEX hits: {len(fdr_df)}")
    print(f"Nominal DXxSEX hits (p < {NOMINAL_ALPHA}): {len(nominal_df)}")

    if not fdr_df.empty:
        print("\n[FDR hits by model]")
        print(fdr_df["model_label"].value_counts().to_string())

        print("\n[FDR hits by stronger_in_sex]")
        print(fdr_df["stronger_in_sex"].value_counts(dropna=False).to_string())

    if not nominal_df.empty:
        print("\n[Nominal hits by model]")
        print(nominal_df["model_label"].value_counts().to_string())

        print("\n[Nominal hits by stronger_in_sex]")
        print(nominal_df["stronger_in_sex"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()