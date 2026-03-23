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
OUT_DIR = ROOT / "results" / "hubs" / "module_stats_dxsex_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "PC",
    "PC_pos",
    "PC_neg",
    "Z",
    "Z_pos",
    "Z_neg",
    "Strength_pos",
    "Strength_neg",
]

MODELS = {
    "m1": "site",
    "m2": "site_iq",
    "m3": "site_iq_rh",
}

SEXES = ["male", "female"]

ROUND_COLS = {
    "mean_ASD": 3,
    "mean_CTL": 3,
    "beta_DX": 3,
    "p_DX": 4,
    "beta_SEX": 3,
    "p_SEX": 4,
    "beta_DXxSEX": 3,
    "p_DXxSEX": 4,
    "p_DXxSEX_FDR": 4,
    "p_SITE": 4,
    "p_IQ": 4,
    "p_RIGHT_HANDED": 4,
}


def load_stats(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [str(c).strip() for c in df.columns]
    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).str.strip()
    return df


def build_model_table(sub: pd.DataFrame, model: str, sex: str) -> pd.DataFrame | None:
    if sex not in {"male", "female"}:
        raise ValueError(f"Unexpected sex: {sex}")

    mean_asd_col = f"mean_ASD_{sex}"
    mean_ctl_col = f"mean_CTL_{sex}"
    n_model_sex_col = f"n_model_{sex}"
    n_model_asd_sex_col = f"n_model_ASD_{sex}"
    n_model_ctl_sex_col = f"n_model_CTL_{sex}"

    needed = [
        "module",
        mean_asd_col,
        mean_ctl_col,
        "n_ASD",
        "n_CTL",
        "n_male",
        "n_female",
        "n_total",
        f"beta_DX_{model}",
        f"p_DX_{model}",
        f"beta_SEX_{model}",
        f"p_SEX_{model}",
        f"beta_DXxSEX_{model}",
        f"p_DXxSEX_{model}",
        f"p_DXxSEX_FDR_{model}",
        f"DXxSEX_FDR_significant_{model}",
        f"p_SITE_{model}",
        f"p_IQ_{model}",
        f"p_RIGHT_HANDED_{model}",
        f"n_model_{model}",
        f"n_model_ASD_{model}",
        f"n_model_CTL_{model}",
        f"n_model_male_{model}",
        f"n_model_female_{model}",
        f"n_model_ASD_male_{model}",
        f"n_model_ASD_female_{model}",
        f"n_model_CTL_male_{model}",
        f"n_model_CTL_female_{model}",
        f"note_{model}",
    ]

    present = [c for c in needed if c in sub.columns]
    if f"beta_DXxSEX_{model}" not in present and f"p_DXxSEX_{model}" not in present:
        return None

    out = sub[present].copy()

    rename_map = {
        mean_asd_col: "mean_ASD",
        mean_ctl_col: "mean_CTL",
        f"beta_DX_{model}": "beta_DX",
        f"p_DX_{model}": "p_DX",
        f"beta_SEX_{model}": "beta_SEX",
        f"p_SEX_{model}": "p_SEX",
        f"beta_DXxSEX_{model}": "beta_DXxSEX",
        f"p_DXxSEX_{model}": "p_DXxSEX",
        f"p_DXxSEX_FDR_{model}": "p_DXxSEX_FDR",
        f"DXxSEX_FDR_significant_{model}": "DXxSEX_FDR_significant",
        f"p_SITE_{model}": "p_SITE",
        f"p_IQ_{model}": "p_IQ",
        f"p_RIGHT_HANDED_{model}": "p_RIGHT_HANDED",
        f"n_model_{model}": "n_model",
        f"n_model_ASD_{model}": "n_model_ASD",
        f"n_model_CTL_{model}": "n_model_CTL",
        n_model_sex_col: "n_model_sex",
        n_model_asd_sex_col: "n_model_ASD_sex",
        n_model_ctl_sex_col: "n_model_CTL_sex",
        f"note_{model}": "note",
    }
    out = out.rename(columns=rename_map)

    for c, nd in ROUND_COLS.items():
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(nd)

    if "DXxSEX_FDR_significant" in out.columns:
        out["sig_DXxSEX_FDR"] = out["DXxSEX_FDR_significant"].map({True: "*", False: ""}).fillna("")
    else:
        out["sig_DXxSEX_FDR"] = ""

    out["sex_table"] = sex

    col_order = [
        "module",
        "sex_table",
        "mean_ASD",
        "mean_CTL",
        "beta_DX",
        "p_DX",
        "beta_SEX",
        "p_SEX",
        "beta_DXxSEX",
        "p_DXxSEX",
        "p_DXxSEX_FDR",
        "sig_DXxSEX_FDR",
        "p_SITE",
        "p_IQ",
        "p_RIGHT_HANDED",
        "n_ASD",
        "n_CTL",
        "n_male",
        "n_female",
        "n_total",
        "n_model",
        "n_model_ASD",
        "n_model_CTL",
        "n_model_sex",
        "n_model_ASD_sex",
        "n_model_CTL_sex",
        "note",
    ]
    col_order = [c for c in col_order if c in out.columns]

    return out[col_order].sort_values("module")


def export_tables():
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {IN_DIR}")

    files = sorted(IN_DIR.glob("*/*/module_stats_dxsex.csv"))
    if not files:
        raise FileNotFoundError(f"No module_stats_dxsex.csv files found in {IN_DIR}")

    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] input dir: {IN_DIR}")
    print(f"[INFO] output dir: {OUT_DIR}")
    print(f"[INFO] found {len(files)} stats file(s)")

    for fp in files:
        scenario = fp.parent.parent.name
        age_group = fp.parent.name

        df = load_stats(fp)
        if df.empty:
            print(f"[SKIP] Empty file: {fp}")
            continue

        if "metric" not in df.columns:
            print(f"[SKIP] Missing metric column: {fp}")
            continue

        for metric in METRICS:
            sub = df[df["metric"] == metric].copy()
            if sub.empty:
                continue

            for model_key, model_label in MODELS.items():
                for sex in SEXES:
                    out = build_model_table(sub, model_key, sex)
                    if out is None or out.empty:
                        continue

                    out_dir = OUT_DIR / scenario / age_group / metric / model_label / sex
                    out_dir.mkdir(parents=True, exist_ok=True)

                    out_path = out_dir / "table.csv"
                    out.to_csv(out_path, index=False)
                    print(f"[SAVED] {out_path}")

    print("\n[DONE] DXxSEX sex-split tables exported.")


if __name__ == "__main__":
    export_tables()