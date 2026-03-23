import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        has_results = (p / "results").exists()
        has_meta = (p / "phenotypes").exists() or (p / "data").exists()
        if has_results and has_meta:
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

IN_DIR = ROOT / "results" / "hubs" / "pc_z_strength_sitecov"
OUT_ROOT = ROOT / "results" / "qc" / "plots_review"

WRITE_SITE_COLORED = False
RNG_SEED = 42

SEX_TO_CODE = {
    "female": 2,
    "male": 1,
}

DX_TO_LABEL = {
    1: "ASD",
    2: "CTL",
}

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

FD03_MODULE_TO_YEO = {
    1: "Somatomotor",
    2: "Visual",
    3: "Limbic",
    4: "Frontoparietal",
    5: "VentralAttention",
    6: "Visual",
    7: "DefaultMode",
    8: "DorsalAttention",
}

TIER_TO_SUBDIR = {
    "core": OUT_ROOT / "main_figures" / "core",
    "secondary": OUT_ROOT / "secondary",
    "sensitivity": OUT_ROOT / "appendix" / "sensitivity",
}

FAMILIES = [
    {
        "family": "family_01_adult_male_strength_pos",
        "tier": "core",
        "scenario": "OVERALL_ageSexMatched_fd-0.3",
        "sex": "male",
        "age_group": "adult_18_plus",
        "metric": "Strength_pos",
        "modules": [2, 4, 6],
        "main_modules": [2, 6],
        "secondary_modules": [4],
        "module_yeo": {
            2: "Visual",
            4: "Frontoparietal",
            6: "Visual",
        },
        "note": "core family; modules 2 and 6 are strongest and repeat across models",
    },
    {
        "family": "family_02_adult_male_pc_pos",
        "tier": "core",
        "scenario": "OVERALL_ageSexMatched_fd-0.3",
        "sex": "male",
        "age_group": "adult_18_plus",
        "metric": "PC_pos",
        "modules": [4, 7],
        "main_modules": [4],
        "secondary_modules": [7],
        "module_yeo": {
            4: "Frontoparietal",
            7: "DefaultMode",
        },
        "note": "core family; module 4 is the main hit, module 7 is weaker",
    },
    {
        "family": "family_03_adult_female_z_all",
        "tier": "core",
        "scenario": "OVERALL_ageSexMatched_fd-0.3",
        "sex": "female",
        "age_group": "adult_18_plus",
        "metric": "Z",
        "modules": [1],
        "main_modules": [1],
        "secondary_modules": [],
        "module_yeo": {
            1: "Somatomotor",
        },
        "note": "core family",
    },
    {
        "family": "family_04_child_female_strength_neg",
        "tier": "core",
        "scenario": "OVERALL_ageSexMatched_fd-0.3",
        "sex": "female",
        "age_group": "child_0_9",
        "metric": "Strength_neg",
        "modules": [2, 3],
        "main_modules": [2],
        "secondary_modules": [3],
        "module_yeo": {
            2: "Visual",
            3: "Limbic",
        },
        "note": "core family; module 2 is strongest, module 3 is secondary",
    },
    {
        "family": "family_05_teen_male_pc_all",
        "tier": "core",
        "scenario": "OVERALL_ageSexMatched_fd-0.3",
        "sex": "male",
        "age_group": "teen_13_17",
        "metric": "PC",
        "modules": [7],
        "main_modules": [7],
        "secondary_modules": [],
        "module_yeo": {
            7: "DefaultMode",
        },
        "note": "core family",
    },
    {
        "family": "family_06_child_male_pc_all",
        "tier": "secondary",
        "scenario": "OVERALL_ageSexMatched_fd-0.3",
        "sex": "male",
        "age_group": "child_0_9",
        "metric": "PC",
        "modules": [6, 7, 8],
        "main_modules": [6, 7, 8],
        "secondary_modules": [],
        "module_yeo": {
            6: "Visual",
            7: "DefaultMode",
            8: "DorsalAttention",
        },
        "note": "secondary family; interesting but less stable across models",
    },
    {
        "family": "family_07_child_female_pc_all",
        "tier": "secondary",
        "scenario": "OVERALL_ageSexMatched_fd-0.3",
        "sex": "female",
        "age_group": "child_0_9",
        "metric": "PC",
        "modules": [4, 7],
        "main_modules": [4, 7],
        "secondary_modules": [],
        "module_yeo": {
            4: "Frontoparietal",
            7: "DefaultMode",
        },
        "note": "secondary family; mainly site-model level",
    },
    {
        "family": "family_08_child_male_strength_pos",
        "tier": "secondary",
        "scenario": "OVERALL_ageSexMatched_fd-0.3",
        "sex": "male",
        "age_group": "child_0_9",
        "metric": "Strength_pos",
        "modules": [2],
        "main_modules": [2],
        "secondary_modules": [],
        "module_yeo": {
            2: "Visual",
        },
        "note": "secondary family; only one model variant currently",
    },
    {
        "family": "family_09_teen_female_pc_neg",
        "tier": "sensitivity",
        "scenario": "OVERALL_ageSexMatched_fd-0.3",
        "sex": "female",
        "age_group": "teen_13_17",
        "metric": "PC_neg",
        "modules": [1, 2, 3, 4, 7, 8],
        "main_modules": [1, 2, 3, 4, 7, 8],
        "secondary_modules": [],
        "module_yeo": {
            1: "Somatomotor",
            2: "Visual",
            3: "Limbic",
            4: "Frontoparietal",
            7: "DefaultMode",
            8: "DorsalAttention",
        },
        "note": "sensitivity family; outlier-dependent due to subject 50127 robustness failure",
    },
]


def resolve_metric_column(df: pd.DataFrame, metric_name: str) -> str:
    aliases = METRIC_ALIASES[metric_name]
    for col in aliases:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find metric column for {metric_name}. Tried: {aliases}")


def load_node_metrics(scenario: str) -> pd.DataFrame:
    fp = IN_DIR / f"{scenario}_node_metrics.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing node metrics file: {fp}")

    df = pd.read_csv(fp)
    df.columns = [str(c).strip() for c in df.columns]

    numeric_cols = [
        "SUB_ID",
        "SEX",
        "DX_GROUP",
        "AGE_AT_SCAN",
        "node",
        "module",
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

    df = df.dropna(subset=["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN", "module", "node"]).copy()
    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["SEX"] = df["SEX"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["module"] = df["module"].astype(int)
    df["node"] = df["node"].astype(int)

    return df


def build_subject_level_family_df(df: pd.DataFrame, family_cfg: dict) -> pd.DataFrame:
    sex_code = SEX_TO_CODE[family_cfg["sex"]]
    metric_col = resolve_metric_column(df, family_cfg["metric"])
    modules = family_cfg["modules"]

    sub = df[
        (df["SEX"] == sex_code)
        & (df["AGE_GROUP"] == family_cfg["age_group"])
        & (df["module"].isin(modules))
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

    subj["dx_label"] = subj["DX_GROUP"].map(DX_TO_LABEL)
    subj["family"] = family_cfg["family"]
    subj["tier"] = family_cfg["tier"]
    subj["scenario"] = family_cfg["scenario"]
    subj["metric"] = family_cfg["metric"]
    subj["dominant_yeo"] = subj["module"].map(family_cfg["module_yeo"]).fillna("unmapped")

    return subj


def summarize_family(subj: pd.DataFrame, family_cfg: dict) -> pd.DataFrame:
    rows = []

    for module in family_cfg["modules"]:
        tmp = subj[subj["module"] == module].copy()
        if tmp.empty:
            continue

        asd = tmp[tmp["DX_GROUP"] == 1]["value"]
        ctl = tmp[tmp["DX_GROUP"] == 2]["value"]

        rows.append(
            {
                "family": family_cfg["family"],
                "tier": family_cfg["tier"],
                "scenario": family_cfg["scenario"],
                "sex": family_cfg["sex"],
                "age_group": family_cfg["age_group"],
                "metric": family_cfg["metric"],
                "module": module,
                "dominant_yeo": family_cfg["module_yeo"].get(module, "unmapped"),
                "is_main_module": module in family_cfg["main_modules"],
                "is_secondary_module": module in family_cfg["secondary_modules"],
                "n_ASD": int(len(asd)),
                "n_CTL": int(len(ctl)),
                "mean_ASD": float(asd.mean()) if len(asd) else np.nan,
                "mean_CTL": float(ctl.mean()) if len(ctl) else np.nan,
                "median_ASD": float(asd.median()) if len(asd) else np.nan,
                "median_CTL": float(ctl.median()) if len(ctl) else np.nan,
                "note": family_cfg["note"],
            }
        )

    return pd.DataFrame(rows)


def get_family_outdir(family_cfg: dict) -> Path:
    base = TIER_TO_SUBDIR[family_cfg["tier"]]
    return base / family_cfg["family"]


def make_plain_family_plot(subj: pd.DataFrame, family_cfg: dict, out_path: Path):
    modules = family_cfg["modules"]
    n = len(modules)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    rng = np.random.default_rng(RNG_SEED)

    for ax, module in zip(axes_flat, modules):
        tmp = subj[subj["module"] == module].copy()

        ctl = tmp.loc[tmp["DX_GROUP"] == 2, "value"].to_numpy()
        asd = tmp.loc[tmp["DX_GROUP"] == 1, "value"].to_numpy()

        ax.boxplot([ctl, asd], positions=[1, 2], widths=0.5, showfliers=False)

        if len(ctl):
            x_ctl = 1 + rng.uniform(-0.08, 0.08, size=len(ctl))
            ax.scatter(x_ctl, ctl, alpha=0.8, s=18)

        if len(asd):
            x_asd = 2 + rng.uniform(-0.08, 0.08, size=len(asd))
            ax.scatter(x_asd, asd, alpha=0.8, s=18)

        yeo = family_cfg["module_yeo"].get(module, "unmapped")
        ax.set_title(f"module {module} | {yeo}")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["CTL", "ASD"])
        ax.set_ylabel(family_cfg["metric"])

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{family_cfg['family']}\n"
        f"{family_cfg['metric']} | {family_cfg['scenario']} | "
        f"{family_cfg['sex']} | {family_cfg['age_group']} | tier={family_cfg['tier']}",
        y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_site_colored_family_plot(subj: pd.DataFrame, family_cfg: dict, out_path: Path):
    modules = family_cfg["modules"]
    n = len(modules)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    rng = np.random.default_rng(RNG_SEED)

    sites = sorted(subj["SITE_ID"].dropna().astype(str).unique().tolist())
    site_to_idx = {site: i for i, site in enumerate(sites)}

    for ax, module in zip(axes_flat, modules):
        tmp = subj[subj["module"] == module].copy()

        ctl = tmp.loc[tmp["DX_GROUP"] == 2, "value"].to_numpy()
        asd = tmp.loc[tmp["DX_GROUP"] == 1, "value"].to_numpy()

        ax.boxplot([ctl, asd], positions=[1, 2], widths=0.5, showfliers=False)

        for dx_value, xpos in [(2, 1), (1, 2)]:
            grp = tmp[tmp["DX_GROUP"] == dx_value].copy()
            if grp.empty:
                continue

            xs = xpos + rng.uniform(-0.08, 0.08, size=len(grp))
            colors = grp["SITE_ID"].astype(str).map(site_to_idx).to_numpy()
            ax.scatter(xs, grp["value"].to_numpy(), c=colors, alpha=0.8, s=22, cmap="tab20")

        yeo = family_cfg["module_yeo"].get(module, "unmapped")
        ax.set_title(f"module {module} | {yeo}")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["CTL", "ASD"])
        ax.set_ylabel(family_cfg["metric"])

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{family_cfg['family']}\n"
        f"{family_cfg['metric']} | colored by site | {family_cfg['scenario']} | "
        f"{family_cfg['sex']} | {family_cfg['age_group']} | tier={family_cfg['tier']}",
        y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_family_manifest():
    rows = []
    for fam in FAMILIES:
        rows.append(
            {
                "family": fam["family"],
                "tier": fam["tier"],
                "scenario": fam["scenario"],
                "sex": fam["sex"],
                "age_group": fam["age_group"],
                "metric": fam["metric"],
                "modules": ",".join(str(x) for x in fam["modules"]),
                "main_modules": ",".join(str(x) for x in fam["main_modules"]),
                "secondary_modules": ",".join(str(x) for x in fam["secondary_modules"]),
                "module_yeo": "; ".join(f"{k}:{v}" for k, v in fam["module_yeo"].items()),
                "note": fam["note"],
            }
        )

    out_dir = OUT_ROOT / "family_definitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fd0p3_family_manifest.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")


def run_one_family(family_cfg: dict, scenario_cache: dict[str, pd.DataFrame]):
    out_dir = get_family_outdir(family_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if family_cfg["scenario"] not in scenario_cache:
        scenario_cache[family_cfg["scenario"]] = load_node_metrics(family_cfg["scenario"])

    df = scenario_cache[family_cfg["scenario"]]
    subj = build_subject_level_family_df(df, family_cfg)

    if subj.empty:
        print(f"[SKIP] {family_cfg['family']} has no subject-level rows")
        return

    subj_path = out_dir / "subject_level_values.csv"
    subj.to_csv(subj_path, index=False)

    summary = summarize_family(subj, family_cfg)
    summary_path = out_dir / "family_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_path = out_dir / "boxplot_jitter.png"
    make_plain_family_plot(subj, family_cfg, plot_path)

    if WRITE_SITE_COLORED:
        site_plot_path = out_dir / "boxplot_jitter_colored_by_site.png"
        make_site_colored_family_plot(subj, family_cfg, site_plot_path)

    print(f"[SAVED] {subj_path}")
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {plot_path}")
    if WRITE_SITE_COLORED:
        print(f"[SAVED] {site_plot_path}")


def main():
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {IN_DIR}")

    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] input dir: {IN_DIR}")
    print(f"[INFO] output root: {OUT_ROOT}")
    print(f"[INFO] site-colored plots: {WRITE_SITE_COLORED}")

    write_family_manifest()

    scenario_cache = {}
    for family_cfg in FAMILIES:
        run_one_family(family_cfg, scenario_cache)

    print("\n[DONE] updated family boxplots written.")


if __name__ == "__main__":
    main()