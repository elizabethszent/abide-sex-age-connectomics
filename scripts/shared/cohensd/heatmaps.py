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

DX_ONLY_D_DIR = ROOT / "results" / "qc" / "cohens_d_dx_only_age_sex"
DXSEX_D_DIR = ROOT / "results" / "qc" / "cohens_d_dxsex_by_age"

DX_ONLY_STATS_DIR = ROOT / "results" / "hubs" / "module_stats_sitecov"
DXSEX_STATS_DIR = ROOT / "results" / "hubs" / "module_stats_dxsex"

OUT_ROOT = ROOT / "results" / "qc" / "module_heatmaps"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

AGE_ORDER = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
SEX_ORDER = ["female", "male"]
METRIC_ORDER = [
    "PC", "PC_pos", "PC_neg",
    "Z", "Z_pos", "Z_neg",
    "Strength_pos", "Strength_neg",
]

MODELS = {
    "m1": "site",
    "m2": "site_iq",
    "m3": "site_iq_rh",
}

FD02_MODULE_LABELS = {
    1: "M1\nSomatomotor",
    2: "M2\nVisual-A",
    3: "M3\nDefaultMode",
    4: "M4\nDorsalAttention",
    5: "M5\nVisual-B",
    6: "M6\nFrontoparietal",
    7: "M7\nLimbic",
    8: "M8\nVentralAttention",
}

FD03_MODULE_LABELS = {
    1: "M1\nSomatomotor",
    2: "M2\nVisual-A",
    3: "M3\nLimbic",
    4: "M4\nFrontoparietal",
    5: "M5\nVentralAttention",
    6: "M6\nVisual-B",
    7: "M7\nDefaultMode",
    8: "M8\nDorsalAttention",
}


def get_module_label_map(scenario: str):
    if "fd-0.2" in scenario:
        return FD02_MODULE_LABELS, "fd-0.2"
    if "fd-0.3" in scenario:
        return FD03_MODULE_LABELS, "fd-0.3"
    return {}, "unknown"


def get_module_label(scenario: str, module: int) -> str:
    mapping, _ = get_module_label_map(scenario)
    return mapping.get(module, f"M{module}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "1", "yes"}


def load_dx_only_d_files():
    files = sorted(DX_ONLY_D_DIR.glob("*/dx_only_cohens_d_age_sex.csv"))
    if not files:
        raise FileNotFoundError(f"No dx_only_cohens_d_age_sex.csv files found in {DX_ONLY_D_DIR}")

    out = {}
    for fp in files:
        scenario = fp.parent.name
        df = pd.read_csv(fp)
        df.columns = [str(c).strip() for c in df.columns]
        df["scenario"] = df["scenario"].astype(str).str.strip()
        df["age_group"] = df["age_group"].astype(str).str.strip()
        df["sex"] = df["sex"].astype(str).str.strip()
        df["metric"] = df["metric"].astype(str).str.strip()
        df["module"] = pd.to_numeric(df["module"], errors="coerce").astype("Int64")
        out[scenario] = df
    return out


def load_dxsex_d_files():
    files = sorted(DXSEX_D_DIR.glob("*/dxsex_cohens_d_by_age.csv"))
    if not files:
        raise FileNotFoundError(f"No dxsex_cohens_d_by_age.csv files found in {DXSEX_D_DIR}")

    out = {}
    for fp in files:
        scenario = fp.parent.name
        df = pd.read_csv(fp)
        df.columns = [str(c).strip() for c in df.columns]
        df["scenario"] = df["scenario"].astype(str).str.strip()
        df["age_group"] = df["age_group"].astype(str).str.strip()
        df["metric"] = df["metric"].astype(str).str.strip()
        df["module"] = pd.to_numeric(df["module"], errors="coerce").astype("Int64")
        out[scenario] = df
    return out


def load_dx_only_significance(scenario: str, sex: str) -> pd.DataFrame:
    rows = []
    for age_group in AGE_ORDER:
        fp = DX_ONLY_STATS_DIR / f"{scenario}__{sex}__{age_group}__module_stats_sitecov.csv"
        if not fp.exists():
            continue

        df = pd.read_csv(fp)
        df.columns = [str(c).strip() for c in df.columns]
        if "metric" not in df.columns or "module" not in df.columns:
            continue

        df["metric"] = df["metric"].astype(str).str.strip()
        df["module"] = pd.to_numeric(df["module"], errors="coerce").astype("Int64")
        df["age_group"] = age_group
        df["sex"] = sex
        df["scenario"] = scenario

        keep = ["scenario", "age_group", "sex", "metric", "module"]
        for model_key in MODELS:
            col = f"DX_FDR_significant_{model_key}"
            if col in df.columns:
                keep.append(col)

        for c in keep:
            if c not in df.columns:
                df[c] = np.nan

        rows.append(df[keep].copy())

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    for model_key in MODELS:
        col = f"DX_FDR_significant_{model_key}"
        if col in out.columns:
            out[col] = out[col].map(safe_bool)
        else:
            out[col] = False
    return out


def load_dxsex_significance(scenario: str) -> pd.DataFrame:
    rows = []
    for age_group in AGE_ORDER:
        fp = DXSEX_STATS_DIR / scenario / age_group / "module_stats_dxsex.csv"
        if not fp.exists():
            continue

        df = pd.read_csv(fp)
        df.columns = [str(c).strip() for c in df.columns]
        if "metric" not in df.columns or "module" not in df.columns:
            continue

        df["metric"] = df["metric"].astype(str).str.strip()
        df["module"] = pd.to_numeric(df["module"], errors="coerce").astype("Int64")
        df["age_group"] = age_group
        df["scenario"] = scenario

        keep = ["scenario", "age_group", "metric", "module"]
        for model_key in MODELS:
            col = f"DXxSEX_FDR_significant_{model_key}"
            if col in df.columns:
                keep.append(col)

        for c in keep:
            if c not in df.columns:
                df[c] = np.nan

        rows.append(df[keep].copy())

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    for model_key in MODELS:
        col = f"DXxSEX_FDR_significant_{model_key}"
        if col in out.columns:
            out[col] = out[col].map(safe_bool)
        else:
            out[col] = False
    return out


def build_matrix(df: pd.DataFrame, row_col: str, value_col: str, scenario: str):
    pivot = df.pivot_table(index=row_col, columns="module", values=value_col, aggfunc="first")
    pivot = pivot.reindex(index=AGE_ORDER, fill_value=np.nan)

    cols = list(range(1, 9))
    pivot = pivot.reindex(columns=cols)
    pivot.columns = [get_module_label(scenario, m) for m in cols]
    return pivot


def build_sig_matrix(df: pd.DataFrame, row_col: str, sig_col: str, scenario: str):
    pivot = df.pivot_table(index=row_col, columns="module", values=sig_col, aggfunc="first")
    pivot = pivot.reindex(index=AGE_ORDER, fill_value=False)

    cols = list(range(1, 9))
    pivot = pivot.reindex(columns=cols, fill_value=False)
    pivot = pivot.fillna(False)
    pivot.columns = [get_module_label(scenario, m) for m in cols]
    return pivot.astype(bool)


def annotate_heatmap(ax, mat, sig_mask=None, fmt="{:.2f}"):
    nrows, ncols = mat.shape
    for i in range(nrows):
        for j in range(ncols):
            val = mat[i, j]
            if np.isnan(val):
                continue
            txt = fmt.format(val)
            if sig_mask is not None and sig_mask[i, j]:
                txt += "*"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)


def compute_vmax(*pivots):
    vals = []
    for pivot in pivots:
        arr = pivot.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 1.0
    return max(0.25, np.nanmax(np.abs(np.concatenate(vals))))


def plot_single_heatmap(pivot: pd.DataFrame, title: str, out_path: Path, sig_mask: pd.DataFrame | None = None):
    arr = pivot.to_numpy(dtype=float)
    vmax = compute_vmax(pivot)

    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#eeeeee")

    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, pad=12)

    sig_arr = sig_mask.to_numpy(dtype=bool) if sig_mask is not None else None
    annotate_heatmap(ax, arr, sig_mask=sig_arr)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Cohen's d")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dxsex_triptych(
    pivot_male: pd.DataFrame,
    pivot_female: pd.DataFrame,
    pivot_delta: pd.DataFrame,
    title: str,
    out_path: Path,
    sig_mask: pd.DataFrame | None = None,
):
    vmax = compute_vmax(pivot_male, pivot_female, pivot_delta)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#eeeeee")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), constrained_layout=True, sharey=True)

    sig_arr = sig_mask.to_numpy(dtype=bool) if sig_mask is not None else None

    mats = [
        ("Male: CTL - ASD", pivot_male),
        ("Female: CTL - ASD", pivot_female),
        ("Delta: female - male", pivot_delta),
    ]

    im = None
    for ax, (subtitle, piv) in zip(axes, mats):
        arr = piv.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(piv.shape[1]))
        ax.set_xticklabels(piv.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(piv.shape[0]))
        ax.set_yticklabels(piv.index)
        ax.set_title(subtitle, pad=8)
        annotate_heatmap(ax, arr, sig_mask=sig_arr)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Cohen's d")

    fig.suptitle(title, fontsize=15)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_dx_only_heatmaps(dx_only_d: dict, manifest_rows: list):
    for scenario, ddf in dx_only_d.items():
        _, mapping_label = get_module_label_map(scenario)

        for sex in SEX_ORDER:
            sig_df = load_dx_only_significance(scenario, sex)
            dsub = ddf[ddf["sex"] == sex].copy()
            if dsub.empty:
                continue

            metrics_present = [m for m in METRIC_ORDER if m in set(dsub["metric"].tolist())]

            for model_key, model_label in MODELS.items():
                for metric in metrics_present:
                    tmp = dsub[dsub["metric"] == metric].copy()
                    if tmp.empty:
                        continue

                    if not sig_df.empty:
                        sig_tmp = sig_df[sig_df["metric"] == metric].copy()
                        merged = tmp.merge(
                            sig_tmp[["age_group", "metric", "module", f"DX_FDR_significant_{model_key}"]],
                            on=["age_group", "metric", "module"],
                            how="left",
                        )
                    else:
                        merged = tmp.copy()
                        merged[f"DX_FDR_significant_{model_key}"] = False

                    sig_col = f"DX_FDR_significant_{model_key}"
                    merged[sig_col] = merged[sig_col].fillna(False).map(safe_bool)

                    pivot = build_matrix(merged, row_col="age_group", value_col="cohens_d_CTL_minus_ASD", scenario=scenario)
                    sig_pivot = build_sig_matrix(merged, row_col="age_group", sig_col=sig_col, scenario=scenario)

                    title = f"DX-only Cohen's d | {scenario} | {metric} | {sex} | {model_label}\nmodule mapping: {mapping_label}"
                    out_dir = OUT_ROOT / "dx_only" / scenario / model_label / metric
                    ensure_dir(out_dir)

                    out_path = out_dir / f"{sex}.png"
                    plot_single_heatmap(pivot, title, out_path, sig_mask=sig_pivot)

                    manifest_rows.append({
                        "stream": "dx_only",
                        "scenario": scenario,
                        "model": model_label,
                        "metric": metric,
                        "sex": sex,
                        "mapping_threshold": mapping_label,
                        "plot": str(out_path),
                    })


def make_dxsex_heatmaps(dxsex_d: dict, manifest_rows: list):
    for scenario, ddf in dxsex_d.items():
        _, mapping_label = get_module_label_map(scenario)
        sig_df = load_dxsex_significance(scenario)

        metrics_present = [m for m in METRIC_ORDER if m in set(ddf["metric"].tolist())]

        for model_key, model_label in MODELS.items():
            for metric in metrics_present:
                tmp = ddf[ddf["metric"] == metric].copy()
                if tmp.empty:
                    continue

                if not sig_df.empty:
                    sig_tmp = sig_df[sig_df["metric"] == metric].copy()
                    merged = tmp.merge(
                        sig_tmp[["age_group", "metric", "module", f"DXxSEX_FDR_significant_{model_key}"]],
                        on=["age_group", "metric", "module"],
                        how="left",
                    )
                else:
                    merged = tmp.copy()
                    merged[f"DXxSEX_FDR_significant_{model_key}"] = False

                sig_col = f"DXxSEX_FDR_significant_{model_key}"
                merged[sig_col] = merged[sig_col].fillna(False).map(safe_bool)

                piv_male = build_matrix(merged, row_col="age_group", value_col="cohens_d_male_CTL_minus_ASD", scenario=scenario)
                piv_female = build_matrix(merged, row_col="age_group", value_col="cohens_d_female_CTL_minus_ASD", scenario=scenario)
                piv_delta = build_matrix(merged, row_col="age_group", value_col="delta_d_female_minus_male", scenario=scenario)
                sig_pivot = build_sig_matrix(merged, row_col="age_group", sig_col=sig_col, scenario=scenario)

                title = f"DX×SEX Cohen's d | {scenario} | {metric} | {model_label}\nmodule mapping: {mapping_label}"
                out_dir = OUT_ROOT / "dxsex" / scenario / model_label / metric
                ensure_dir(out_dir)

                out_path = out_dir / "dxsex_triptych.png"
                plot_dxsex_triptych(piv_male, piv_female, piv_delta, title, out_path, sig_mask=sig_pivot)

                manifest_rows.append({
                    "stream": "dxsex",
                    "scenario": scenario,
                    "model": model_label,
                    "metric": metric,
                    "sex": "male+female+delta",
                    "mapping_threshold": mapping_label,
                    "plot": str(out_path),
                })


def main():
    print(f"[INFO] repo root: {ROOT}")
    print(f"[INFO] output root: {OUT_ROOT}")

    dx_only_d = load_dx_only_d_files()
    dxsex_d = load_dxsex_d_files()

    manifest_rows = []
    make_dx_only_heatmaps(dx_only_d, manifest_rows)
    make_dxsex_heatmaps(dxsex_d, manifest_rows)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT_ROOT / "heatmap_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"\n[SAVED] {manifest_path}")
    print("[DONE] module heatmaps generated.")


if __name__ == "__main__":
    main()