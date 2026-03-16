import pandas as pd
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "results").exists() and (p / "phenotypes").exists():
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

IN_DIR = ROOT / "results" / "hubs" / "module_stats_sitecov"
OUT_DIR = ROOT / "results" / "hubs" / "module_stats_sitecov_tables"
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

MODELS = ["m1", "m2", "m3"]

ROUND_COLS = {
    "mean_CTL": 3,
    "mean_ASD": 3,
    "beta_CTL_minus_ASD": 3,
    "p_DX": 4,
    "p_DX_FDR": 4,
    "p_SITE": 4,
    "p_IQ": 4,
    "p_RIGHT_HANDED": 4,
}


def load_stats(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip()

    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).str.strip()

    return df


def build_model_table(sub: pd.DataFrame, model: str) -> pd.DataFrame | None:
    needed = [
        "module",
        "mean_CTL",
        "mean_ASD",
        "n_ASD",
        "n_CTL",
        f"beta_CTL_minus_ASD_{model}",
        f"p_DX_{model}",
        f"p_DX_FDR_{model}",
        f"DX_FDR_significant_{model}",
        f"p_SITE_{model}",
        f"p_IQ_{model}",
        f"p_RIGHT_HANDED_{model}",
        f"n_model_{model}",
        f"n_model_ASD_{model}",
        f"n_model_CTL_{model}",
        f"note_{model}",
    ]

    present = [c for c in needed if c in sub.columns]
    if f"beta_CTL_minus_ASD_{model}" not in present and f"p_DX_{model}" not in present:
        return None

    out = sub[present].copy()

    rename_map = {
        f"beta_CTL_minus_ASD_{model}": "beta_CTL_minus_ASD",
        f"p_DX_{model}": "p_DX",
        f"p_DX_FDR_{model}": "p_DX_FDR",
        f"DX_FDR_significant_{model}": "DX_FDR_significant",
        f"p_SITE_{model}": "p_SITE",
        f"p_IQ_{model}": "p_IQ",
        f"p_RIGHT_HANDED_{model}": "p_RIGHT_HANDED",
        f"n_model_{model}": "n_model",
        f"n_model_ASD_{model}": "n_model_ASD",
        f"n_model_CTL_{model}": "n_model_CTL",
        f"note_{model}": "note",
    }
    out = out.rename(columns=rename_map)

    for c, nd in ROUND_COLS.items():
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(nd)

    if "DX_FDR_significant" in out.columns:
        out["sig_DX_FDR"] = out["DX_FDR_significant"].map({True: "*", False: ""}).fillna("")
    else:
        out["sig_DX_FDR"] = ""

    col_order = [
        "module",
        "mean_CTL",
        "mean_ASD",
        "beta_CTL_minus_ASD",
        "p_DX",
        "p_DX_FDR",
        "sig_DX_FDR",
        "p_SITE",
        "p_IQ",
        "p_RIGHT_HANDED",
        "n_ASD",
        "n_CTL",
        "n_model",
        "n_model_ASD",
        "n_model_CTL",
        "note",
    ]
    col_order = [c for c in col_order if c in out.columns]

    return out[col_order].sort_values("module")


def export_tables():
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {IN_DIR}")

    files = sorted(IN_DIR.glob("*__module_stats_sitecov.csv"))
    if not files:
        raise FileNotFoundError(f"No module stats CSVs found in {IN_DIR}")

    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] input dir: {IN_DIR}")
    print(f"[INFO] output dir: {OUT_DIR}")
    print(f"[INFO] found {len(files)} module-stats file(s)")

    for fp in files:
        df = load_stats(fp)
        if df.empty:
            print(f"[SKIP] Empty file: {fp.name}")
            continue

        prefix = fp.name.replace("__module_stats_sitecov.csv", "")

        if "metric" not in df.columns:
            print(f"[SKIP] Missing 'metric' column in {fp.name}")
            continue

        for metric in METRICS:
            sub = df[df["metric"] == metric].copy()
            if sub.empty:
                continue

            for model in MODELS:
                out = build_model_table(sub, model)
                if out is None or out.empty:
                    continue

                out_path = OUT_DIR / f"{prefix}__{metric}__{model}__table.csv"
                out.to_csv(out_path, index=False)
                print(f"[SAVED] {out_path}")

    print("\n[DONE] tables exported.")


if __name__ == "__main__":
    export_tables()