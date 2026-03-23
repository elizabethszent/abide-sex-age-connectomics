import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        has_results = (p / "results").exists()
        has_data = (p / "data").exists() or (p / "phenotypes").exists()
        if has_results and has_data:
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

# ----------------------------
# CONFIG
# ----------------------------
HIT_MODE = "fdr"   # "fdr" or "nominal"
WRITE_SITE_COLORED = False
RNG_SEED = 42

SIG_DIR = ROOT / "results" / "qc" / "dxsex_sig_hits"
NODE_DIR = ROOT / "results" / "hubs" / "pc_z_strength_sitecov"
OUT_ROOT = ROOT / "results" / "qc" / "plots_dxsex_sig"

if HIT_MODE == "fdr":
    HITS_FILE = SIG_DIR / "dxsex_fdr_significant_hits.csv"
elif HIT_MODE == "nominal":
    HITS_FILE = SIG_DIR / "dxsex_nominal_hits_p_lt_0p05.csv"
else:
    raise ValueError("HIT_MODE must be 'fdr' or 'nominal'.")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

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

GROUP_ORDER = [
    ("CTL male", 2, 1),
    ("ASD male", 1, 1),
    ("CTL female", 2, 2),
    ("ASD female", 1, 2),
]

# ----------------------------
# THRESHOLD-SPECIFIC MODULE LABELS
# ----------------------------
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


def sanitize_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s


def resolve_metric_column(df: pd.DataFrame, metric_name: str) -> str:
    aliases = METRIC_ALIASES.get(metric_name, [metric_name])
    for col in aliases:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find metric column for '{metric_name}'. Tried {aliases}")


def get_module_label_map(scenario: str) -> tuple[dict[int, str], str]:
    if "fd-0.2" in scenario:
        return FD02_MODULE_LABELS, "fd-0.2"
    if "fd-0.3" in scenario:
        return FD03_MODULE_LABELS, "fd-0.3"
    return {}, "unknown"


def get_module_label(scenario: str, module: int) -> str:
    label_map, _ = get_module_label_map(scenario)
    return label_map.get(module, f"M{module}")


def load_hits() -> pd.DataFrame:
    if not HITS_FILE.exists():
        raise FileNotFoundError(f"Missing hits file: {HITS_FILE}")

    df = pd.read_csv(HITS_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"scenario", "age_group", "metric", "module", "model_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{HITS_FILE.name} missing required columns: {sorted(missing)}")

    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["age_group"] = df["age_group"].astype(str).str.strip()
    df["metric"] = df["metric"].astype(str).str.strip()
    df["model_label"] = df["model_label"].astype(str).str.strip()
    df["module"] = pd.to_numeric(df["module"], errors="coerce")
    df = df.dropna(subset=["module"]).copy()
    df["module"] = df["module"].astype(int)

    df["module_label"] = df.apply(lambda r: get_module_label(r["scenario"], int(r["module"])), axis=1)

    return df


def collapse_hits(hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["scenario", "age_group", "metric", "module"]

    for keys, sub in hits.groupby(group_cols, dropna=False):
        scenario, age_group, metric, module = keys
        models = sorted(sub["model_label"].dropna().astype(str).unique().tolist())
        model_keys = sorted(sub["model_key"].dropna().astype(str).unique().tolist()) if "model_key" in sub.columns else []
        _, mapping_threshold = get_module_label_map(scenario)

        row = {
            "scenario": scenario,
            "age_group": age_group,
            "metric": metric,
            "module": int(module),
            "module_label": get_module_label(scenario, int(module)),
            "mapping_threshold": mapping_threshold,
            "models_sig": ",".join(models),
            "model_keys_sig": ",".join(model_keys),
            "n_models_sig": len(models),
        }

        for col in [
            "beta_DXxSEX",
            "p_DXxSEX",
            "p_DXxSEX_FDR",
            "stronger_in_sex",
            "interaction_pattern",
        ]:
            if col in sub.columns:
                vals = sub[col].dropna().tolist()
                row[f"{col}_values"] = "; ".join(str(v) for v in vals)

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["scenario", "age_group", "metric", "module"]).reset_index(drop=True)


def load_node_metrics_for_scenario(scenario: str) -> pd.DataFrame:
    fp = NODE_DIR / f"{scenario}_node_metrics.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing node metrics file: {fp}")

    df = pd.read_csv(fp)
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

    df = df.dropna(subset=["SUB_ID", "DX_GROUP", "SEX", "module", "node"]).copy()
    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["SEX"] = df["SEX"].astype(int)
    df["module"] = df["module"].astype(int)
    df["node"] = df["node"].astype(int)

    return df


def build_subject_level_module_values(
    df: pd.DataFrame,
    age_group: str,
    metric_name: str,
    module: int,
) -> pd.DataFrame:
    metric_col = resolve_metric_column(df, metric_name)

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


def make_boxplot(subj: pd.DataFrame, title: str, ylabel: str, out_path: Path):
    rng = np.random.default_rng(RNG_SEED)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    series_list = []
    labels = []

    for label, dx_val, sex_val in GROUP_ORDER:
        vals = subj.loc[
            (subj["DX_GROUP"] == dx_val) & (subj["SEX"] == sex_val),
            "value",
        ].to_numpy()
        series_list.append(vals)
        labels.append(label.replace(" ", "\n"))

    ax.boxplot(series_list, positions=[1, 2, 3, 4], widths=0.55, showfliers=False)

    for i, vals in enumerate(series_list, start=1):
        if len(vals):
            xs = i + rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(xs, vals, alpha=0.8, s=22)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_boxplot_colored_by_site(subj: pd.DataFrame, title: str, ylabel: str, out_path: Path):
    rng = np.random.default_rng(RNG_SEED)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    series_list = []
    labels = []

    sites = sorted(subj["SITE_ID"].dropna().astype(str).unique().tolist())
    site_to_idx = {site: i for i, site in enumerate(sites)}

    for label, dx_val, sex_val in GROUP_ORDER:
        vals = subj.loc[
            (subj["DX_GROUP"] == dx_val) & (subj["SEX"] == sex_val),
            "value",
        ].to_numpy()
        series_list.append(vals)
        labels.append(label.replace(" ", "\n"))

    ax.boxplot(series_list, positions=[1, 2, 3, 4], widths=0.55, showfliers=False)

    for i, (_, dx_val, sex_val) in enumerate(GROUP_ORDER, start=1):
        grp = subj[(subj["DX_GROUP"] == dx_val) & (subj["SEX"] == sex_val)].copy()
        if grp.empty:
            continue
        xs = i + rng.uniform(-0.08, 0.08, size=len(grp))
        colors = grp["SITE_ID"].astype(str).map(site_to_idx).to_numpy()
        ax.scatter(xs, grp["value"].to_numpy(), c=colors, alpha=0.8, s=24, cmap="tab20")

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_groups(subj: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label, dx_val, sex_val in GROUP_ORDER:
        vals = subj.loc[
            (subj["DX_GROUP"] == dx_val) & (subj["SEX"] == sex_val),
            "value",
        ]
        rows.append(
            {
                "group": label,
                "DX_GROUP": dx_val,
                "SEX": sex_val,
                "n": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "median": float(vals.median()) if len(vals) else np.nan,
                "min": float(vals.min()) if len(vals) else np.nan,
                "max": float(vals.max()) if len(vals) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def run():
    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] hit mode: {HIT_MODE}")
    print(f"[INFO] hits file: {HITS_FILE}")
    print(f"[INFO] node dir: {NODE_DIR}")
    print(f"[INFO] output root: {OUT_ROOT}")

    hits = load_hits()
    collapsed = collapse_hits(hits)

    if collapsed.empty:
        raise RuntimeError("No hits found to plot.")

    manifest_rows = []
    scenario_cache = {}

    for _, hit in collapsed.iterrows():
        scenario = hit["scenario"]
        age_group = hit["age_group"]
        metric = hit["metric"]
        module = int(hit["module"])
        module_label = hit["module_label"]
        models_sig = hit["models_sig"]
        mapping_threshold = hit["mapping_threshold"]

        if scenario not in scenario_cache:
            scenario_cache[scenario] = load_node_metrics_for_scenario(scenario)

        df = scenario_cache[scenario]
        subj = build_subject_level_module_values(
            df=df,
            age_group=age_group,
            metric_name=metric,
            module=module,
        )

        if subj.empty:
            print(f"[SKIP] No data for {scenario} | {age_group} | {metric} | {module_label}")
            continue

        out_dir = (
            OUT_ROOT
            / scenario
            / age_group
            / metric
            / sanitize_filename(module_label)
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        subj_path = out_dir / "subject_level_values.csv"
        summary_path = out_dir / "group_summary.csv"
        plot_path = out_dir / "boxplot_dxsex.png"

        subj.to_csv(subj_path, index=False)
        summarize_groups(subj).to_csv(summary_path, index=False)

        title = (
            f"{scenario} | {age_group} | {metric} | {module_label}\n"
            f"mapping: {mapping_threshold} | significant models: {models_sig}"
        )
        make_boxplot(subj, title=title, ylabel=metric, out_path=plot_path)

        site_plot_path = None
        if WRITE_SITE_COLORED:
            site_plot_path = out_dir / "boxplot_dxsex_colored_by_site.png"
            make_boxplot_colored_by_site(
                subj,
                title=title + " | colored by site",
                ylabel=metric,
                out_path=site_plot_path,
            )

        manifest_rows.append(
            {
                "scenario": scenario,
                "age_group": age_group,
                "metric": metric,
                "module": module,
                "module_label": module_label,
                "mapping_threshold": mapping_threshold,
                "models_sig": models_sig,
                "n_models_sig": int(hit["n_models_sig"]),
                "subject_level_csv": str(subj_path),
                "group_summary_csv": str(summary_path),
                "plot_path": str(plot_path),
                "site_plot_path": str(site_plot_path) if site_plot_path else "",
            }
        )

        print(f"[SAVED] {plot_path}")

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["scenario", "age_group", "metric", "module"]
    ).reset_index(drop=True)

    manifest_path = OUT_ROOT / f"manifest_{HIT_MODE}_dxsex_plots.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"\n[SAVED] {manifest_path}")
    print("[DONE] DX×SEX significant-hit boxplots generated.")


if __name__ == "__main__":
    run()