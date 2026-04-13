from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "results").exists():
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

# config

# main input from all-modules ADOS script
RESULTS_CSV = ROOT / "results" / "qc" / "ados_correlations_all_modules" / "ados_all_modules__ADOS_TOTAL.csv"

# Where plots will go
OUT_DIR = ROOT / "results" / "qc" / "ados_correlations_all_modules" / "selected_scatterplots"

# how to choose rows to plot
# options:
#   "fdr_metric" -> rows with spearman_fdr_sig_by_scenario_sex_metric == True
#   "fdr_sex"    -> rows with spearman_fdr_sig_by_scenario_sex == True
#   "top_n"      -> top N rows by |spearman_rho|
#   "raw_p"      -> rows with raw spearman_p <= RAW_P_MAX
SELECTION_MODE = "top_n"

TOP_N = 12
RAW_P_MAX = 0.05

# plot settings
POINT_SIZE = 36
ALPHA = 0.85
DPI = 300

# whether to also save a manifest of plotted rows
SAVE_MANIFEST = True


# helpers
def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(s).strip())


def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def choose_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if SELECTION_MODE == "fdr_metric":
        if "spearman_fdr_sig_by_scenario_sex_metric" not in df.columns:
            raise ValueError("Missing column: spearman_fdr_sig_by_scenario_sex_metric")
        out = df[df["spearman_fdr_sig_by_scenario_sex_metric"] == True].copy()

    elif SELECTION_MODE == "fdr_sex":
        if "spearman_fdr_sig_by_scenario_sex" not in df.columns:
            raise ValueError("Missing column: spearman_fdr_sig_by_scenario_sex")
        out = df[df["spearman_fdr_sig_by_scenario_sex"] == True].copy()

    elif SELECTION_MODE == "top_n":
        if "spearman_rho" not in df.columns:
            raise ValueError("Missing column: spearman_rho")
        out = df.copy()
        out["abs_rho"] = out["spearman_rho"].abs()
        out = out.sort_values("abs_rho", ascending=False).head(TOP_N).copy()
        out = out.drop(columns=["abs_rho"], errors="ignore")

    elif SELECTION_MODE == "raw_p":
        if "spearman_p" not in df.columns:
            raise ValueError("Missing column: spearman_p")
        out = df[df["spearman_p"] <= RAW_P_MAX].copy()

    else:
        raise ValueError(f"Unknown SELECTION_MODE: {SELECTION_MODE}")

    return out.reset_index(drop=True)


def make_single_plot(row: pd.Series):
    subject_csv = Path(str(row["subject_csv"]))
    if not subject_csv.exists():
        print(f"[WARN] Missing subject CSV: {subject_csv}")
        return None

    df = pd.read_csv(subject_csv)
    df.columns = [str(c).strip() for c in df.columns]

    ados_col = str(row["ados_col"])
    if ados_col not in df.columns:
        print(f"[WARN] {subject_csv.name} missing ADOS column: {ados_col}")
        return None
    if "module_value" not in df.columns:
        print(f"[WARN] {subject_csv.name} missing module_value")
        return None

    x = pd.to_numeric(df[ados_col], errors="coerce")
    y = pd.to_numeric(df["module_value"], errors="coerce")
    keep = x.notna() & y.notna()
    x = x[keep]
    y = y[keep]

    if len(x) == 0:
        print(f"[WARN] No plottable rows in {subject_csv}")
        return None

    scenario = str(row["scenario"])
    age_group = str(row["age_group"])
    sex = str(row["sex"])
    metric = str(row["metric"])
    module = int(row["module"])

    rho = safe_float(row.get("spearman_rho", np.nan))
    p_raw = safe_float(row.get("spearman_p", np.nan))
    p_fdr_metric = safe_float(row.get("spearman_p_fdr_by_scenario_sex_metric", np.nan))
    p_fdr_sex = safe_float(row.get("spearman_p_fdr_by_scenario_sex", np.nan))
    n = int(row.get("n", len(x))) if pd.notna(row.get("n", np.nan)) else len(x)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(x, y, s=POINT_SIZE, alpha=ALPHA)

    # regression line
    if len(x) >= 2:
        try:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(float(x.min()), float(x.max()), 100)
            yy = coef[0] * xx + coef[1]
            ax.plot(xx, yy, linewidth=1.8)
        except Exception:
            pass

    title = f"{scenario}\n{age_group} | {sex} | {metric} | M{module}"
    ax.set_title(title)
    ax.set_xlabel(ados_col)
    ax.set_ylabel("Subject-level module median")

    stats_lines = [
        f"n = {n}",
        f"Spearman rho = {rho:.3f}" if np.isfinite(rho) else "Spearman rho = NaN",
        f"raw p = {p_raw:.4g}" if np.isfinite(p_raw) else "raw p = NaN",
    ]

    if np.isfinite(p_fdr_metric):
        stats_lines.append(f"FDR (scenario×sex×metric) = {p_fdr_metric:.4g}")
    if np.isfinite(p_fdr_sex):
        stats_lines.append(f"FDR (scenario×sex) = {p_fdr_sex:.4g}")

    ax.text(
        0.02, 0.98,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.7"),
    )

    out_subdir = OUT_DIR / sanitize(scenario) / sanitize(age_group) / sanitize(sex) / sanitize(metric)
    out_subdir.mkdir(parents=True, exist_ok=True)

    filename = f"M{module}__rho_{rho:.3f}.png" if np.isfinite(rho) else f"M{module}.png"
    out_path = out_subdir / sanitize(filename)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return out_path


# main
def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_CSV)
    df.columns = [str(c).strip() for c in df.columns]

    selected = choose_rows(df)

    if selected.empty:
        print("[INFO] No rows matched the current selection mode.")
        return

    print(f"[INFO] Results CSV: {RESULTS_CSV}")
    print(f"[INFO] Selection mode: {SELECTION_MODE}")
    print(f"[INFO] Rows selected: {len(selected)}")
    print(f"[INFO] Output dir: {OUT_DIR}")

    saved_paths = []
    for _, row in selected.iterrows():
        out_path = make_single_plot(row)
        if out_path is not None:
            saved_paths.append(str(out_path))

    if SAVE_MANIFEST:
        manifest = selected.copy()
        manifest["saved_plot"] = saved_paths + [""] * max(0, len(manifest) - len(saved_paths))
        manifest_path = OUT_DIR / f"scatterplot_manifest__{SELECTION_MODE}.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"[SAVED] {manifest_path}")

    print(f"[INFO] Scatter plots saved: {len(saved_paths)}")
    for p in saved_paths:
        print(f"[SAVED] {p}")


if __name__ == "__main__":
    main()