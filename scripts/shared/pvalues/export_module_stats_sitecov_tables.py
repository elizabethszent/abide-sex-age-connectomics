import re
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

SEARCH_ROOT = ROOT / "results" / "hubs"   # search recursively under here
OUT_DIR = ROOT / "results" / "hubs" / "module_stats_sitecov_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional filters (leave None to export everything found)
ONLY_SCENARIOS = None  # e.g. {"OVERALL_ageSexMatched_fd-0.2"}
ONLY_SEXES = None      # e.g. {"female", "male"}
ONLY_AGES = None       # e.g. {"child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"}
ONLY_METRICS = None    # e.g. {"PC", "Z", "Strength_pos", "Strength_neg"}

ROUND_COLS = {
    "mean_CTL": 3,
    "mean_ASD": 3,
    "beta_CTL_minus_ASD": 3,
    "p_DX": 4,
    "p_DX_FDR": 4,
    "p_SITE": 4,
    "r2": 3,
}

CANON_COLS = [
    "module",
    "mean_CTL",
    "mean_ASD",
    "beta_CTL_minus_ASD",
    "p_DX",
    "p_DX_FDR",
    "sig_DX_FDR",
    "p_SITE",
    "r2",
    "n_ASD",
    "n_CTL",
    "note",
]

ALIASES = {
    # module
    "louvain_module": "module",
    "module_id": "module",
    "mod": "module",

    # metric
    "metric_name": "metric",
    "meas": "metric",

    # means / betas
    "mean_ctl": "mean_CTL",
    "mean_control": "mean_CTL",
    "mean_controls": "mean_CTL",
    "mean_asd": "mean_ASD",
    "beta": "beta_CTL_minus_ASD",
    "beta_ctl_minus_asd": "beta_CTL_minus_ASD",

    # p-values
    "p_dx": "p_DX",
    "pvalue_dx": "p_DX",
    "p_dx_fdr": "p_DX_FDR",
    "p_dx_adj": "p_DX_FDR",
    "p_site": "p_SITE",
    "pvalue_site": "p_SITE",

    # fit
    "r_squared": "r2",

    # ns
    "n_asd": "n_ASD",
    "n_ctl": "n_CTL",
    "n_control": "n_CTL",

    # notes
    "notes": "note",

    # significance flag
    "dx_fdr_significant": "DX_FDR_significant",
    "sig_dx_fdr": "sig_DX_FDR",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    ren = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in ALIASES:
            ren[c] = ALIASES[key]
    df = df.rename(columns=ren)

    # ensure "metric" exists if it was just casing-different
    if "metric" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "metric":
                df = df.rename(columns={c: "metric"})
                break

    # ensure "module" exists if it was just casing-different
    if "module" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "module":
                df = df.rename(columns={c: "module"})
                break

    return df


def safe_tag(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    return s


def discover_files(search_root: Path) -> list[Path]:
    if not search_root.exists():
        raise FileNotFoundError(f"Search root not found: {search_root}")

    # catch both naming styles:
    #   scenario__sex__age__module_stats_sitecov.csv
    #   ...module_stats_sitecov.csv
    pats = [
        "*__module_stats_sitecov.csv",
        "*module_stats_sitecov.csv",
    ]

    files = []
    for pat in pats:
        files.extend(search_root.rglob(pat))

    # unique + sorted
    files = sorted(set(files))
    return files


def parse_scenario_sex_age(fp: Path):
    """
    Preferred filename format:
      <scenario>__<sex>__<age>__module_stats_sitecov.csv

    If it doesn't match, return (stem, "unknown", "unknown").
    """
    name = fp.name
    suffix = "__module_stats_sitecov.csv"
    if name.endswith(suffix):
        base = name[: -len(suffix)]
        parts = base.split("__")
        if len(parts) >= 3:
            scenario = "__".join(parts[:-2])
            sex = parts[-2]
            age = parts[-1]
            return scenario, sex, age

    # fallback: don't crash on other naming
    return fp.stem, "unknown", "unknown"


def load_stats(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df = normalize_columns(df)

    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).str.strip()
    if "module" in df.columns:
        df["module"] = pd.to_numeric(df["module"], errors="coerce")

    return df


def export_tables():
    files = discover_files(SEARCH_ROOT)
    if not files:
        print(f"[WARN] No module_stats_sitecov CSVs found under: {SEARCH_ROOT}")
        print("[HINT] Make sure the step that writes '*module_stats_sitecov.csv' has run.")
        return

    print(f"[INFO] Found {len(files)} module_stats_sitecov CSV(s).")
    n_saved = 0

    for fp in files:
        scenario, sex, age = parse_scenario_sex_age(fp)

        if ONLY_SCENARIOS and scenario not in ONLY_SCENARIOS:
            continue
        if ONLY_SEXES and sex not in ONLY_SEXES:
            continue
        if ONLY_AGES and age not in ONLY_AGES:
            continue

        df = load_stats(fp)
        if df.empty:
            continue
        if "metric" not in df.columns or "module" not in df.columns:
            print(f"[WARN] Skipping (missing metric/module columns): {fp}")
            continue

        metrics = sorted(df["metric"].dropna().unique().tolist())
        if ONLY_METRICS:
            metrics = [m for m in metrics if m in ONLY_METRICS]

        for metric in metrics:
            sub = df[df["metric"] == metric].copy()
            if sub.empty:
                continue

            # round numeric columns
            for c, nd in ROUND_COLS.items():
                if c in sub.columns:
                    sub[c] = pd.to_numeric(sub[c], errors="coerce").round(nd)

            # add sig column
            if "sig_DX_FDR" not in sub.columns:
                if "DX_FDR_significant" in sub.columns:
                    flag = sub["DX_FDR_significant"].map(
                        lambda x: str(x).strip().lower() in {"true", "1", "yes", "y"}
                    )
                    sub["sig_DX_FDR"] = flag.map({True: "*", False: ""})
                else:
                    sub["sig_DX_FDR"] = ""

            cols = [c for c in CANON_COLS if c in sub.columns]
            out = sub[cols].copy()

            # sort by module
            if "module" in out.columns:
                out = out.sort_values("module", kind="mergesort")

            out_name = f"{safe_tag(scenario)}__{safe_tag(sex)}__{safe_tag(age)}__{safe_tag(metric)}__table.csv"
            out_path = OUT_DIR / out_name
            out.to_csv(out_path, index=False)
            print(f"[SAVED] {out_path}")
            n_saved += 1

    print(f"\n[DONE] Exported {n_saved} table(s) to:\n  {OUT_DIR}")


if __name__ == "__main__":
    export_tables()