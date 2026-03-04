import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


ROOT = Path(r"C:\Users\eliza\Connectomics\TERMproject\abide-sex-age-connectomics")

HUB_DIR = ROOT / "results" / "hubs"
OUT_DIR = HUB_DIR / "module_stats_sitecov"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ABIDE12 SITE map should come from your combined metadata
META_COMBINED = ROOT / "data" / "metadata" / "ABIDE12_phenotypes_combined.csv"

SCENARIO = "ABIDE12_fd2"

# Hub files produced by your ABIDE12 hubs script
HUB_GLOB = "*_pc_z_strengthPosNeg_abide12_fd2.csv"

# Metrics we expect in hub CSVs (we’ll compute for whichever are present)
METRICS = [
    "PC", "PC_pos", "PC_neg",
    "z", "z_pos", "z_neg",
    "strength_pos", "strength_neg",
]

EXPECTED_K = 7  # modules 1..7 typically


# ----------------------------
# Utilities
# ----------------------------
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def file_id_to_subject_int(file_id: str) -> int | None:
    s = str(file_id).strip()
    s = re.sub(r"\.0$", "", s)  # strip trailing .0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def bh_fdr(pvals: list[float]) -> list[float]:
    """BH-FDR; treat non-finite as 1.0."""
    p = np.array([pv if np.isfinite(pv) else 1.0 for pv in pvals], dtype=float)
    _, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    return q.tolist()


def build_site_map_from_combined(meta_csv: Path) -> dict[int, str]:
    """
    Build map: SUB_ID (int) -> SITE_ID (str) from ABIDE12 combined metadata CSV.
    Requires columns like SUB_ID and SITE_ID (case-insensitive).
    """
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing combined metadata: {meta_csv}")

    df = pd.read_csv(meta_csv)
    df.columns = df.columns.str.strip()

    cols_upper = {c.upper(): c for c in df.columns}

    sub_col = cols_upper.get("SUB_ID") or cols_upper.get("SUBID") or cols_upper.get("SUBJECT_ID")
    site_col = cols_upper.get("SITE_ID") or cols_upper.get("SITE") or cols_upper.get("SITEID")

    if sub_col is None or site_col is None:
        raise ValueError(
            f"{meta_csv} must contain SUB_ID and SITE_ID (or close variants). "
            f"Found columns (first 30): {list(df.columns)[:30]}"
        )

    sub = df[[sub_col, site_col]].copy()
    sub[sub_col] = pd.to_numeric(sub[sub_col], errors="coerce")
    sub = sub.dropna(subset=[sub_col, site_col]).copy()

    sub[sub_col] = sub[sub_col].astype(int)
    sub[site_col] = sub[site_col].astype(str).str.strip()

    # one row per subject
    sub = sub.sort_values([sub_col, site_col]).drop_duplicates(subset=[sub_col], keep="first")
    return dict(zip(sub[sub_col].tolist(), sub[site_col].tolist()))


def discover_hub_files(hub_dir: Path) -> list[Path]:
    return sorted(hub_dir.glob(HUB_GLOB))


def parse_sex_age_from_filename(fp: Path) -> tuple[str, str] | None:
    """
    Expects:
      female_child_0_9_pc_z_strengthPosNeg_abide12_fd2.csv
    """
    m = re.match(
        r"^(female|male)_(child_0_9|preteen_10_12|teen_13_17|adult_18_plus)_pc_z_strengthPosNeg_abide12_fd2\.csv$",
        fp.name,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).lower(), m.group(2)


def ensure_required_columns(df: pd.DataFrame, fp: Path):
    need = {"DX_GROUP", "module", "node"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"{fp} missing required columns: {missing}. Found: {sorted(df.columns)}")

    if ("subject_int" not in df.columns) and ("FILE_ID" not in df.columns) and ("SUB_ID" not in df.columns):
        raise RuntimeError(f"{fp} must contain subject_int or SUB_ID or FILE_ID to map subjects.")


def attach_subject_int(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "subject_int" in df.columns:
        df["subject_int"] = safe_numeric(df["subject_int"])
        return df

    if "SUB_ID" in df.columns:
        df["subject_int"] = safe_numeric(df["SUB_ID"])
        return df

    # fallback: derive from FILE_ID
    df["subject_int"] = df["FILE_ID"].apply(file_id_to_subject_int)
    df["subject_int"] = safe_numeric(df["subject_int"])
    return df


def run_sitecov_model(sub_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Fit:
      value ~ dx_ctl + C(SITE_ID)

    dx_ctl = 1 for Control (DX_GROUP=2), 0 for ASD (DX_GROUP=1)

    Returns:
      beta_CTL_minus_ASD, p_DX, p_SITE(omnibus), r2
    """
    sub_df = sub_df.copy()
    full = smf.ols("value ~ dx_ctl + C(SITE_ID)", data=sub_df).fit()

    beta = float(full.params.get("dx_ctl", np.nan))
    p_dx = float(full.pvalues.get("dx_ctl", np.nan))
    r2 = float(full.rsquared) if hasattr(full, "rsquared") else np.nan

    site_terms = [p for p in full.params.index if p.startswith("C(SITE_ID)[T.")]
    if not site_terms:
        p_site = np.nan
    else:
        try:
            hyp = ", ".join([f"{t}=0" for t in site_terms])
            p_site = float(full.f_test(hyp).pvalue)
        except Exception:
            p_site = np.nan

    return beta, p_dx, p_site, r2


# ----------------------------
# Main
# ----------------------------
def main():
    site_map = build_site_map_from_combined(META_COMBINED)
    hub_files = discover_hub_files(HUB_DIR)

    if not hub_files:
        raise FileNotFoundError(f"No hub CSVs found under: {HUB_DIR} matching {HUB_GLOB}")

    print(f"[INFO] Found {len(hub_files)} hub CSVs. SITE map entries: {len(site_map)}")

    for fp in hub_files:
        parsed = parse_sex_age_from_filename(fp)
        if parsed is None:
            print(f"[WARN] Skipping (filename not recognized): {fp.name}")
            continue

        sex, age = parsed

        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip()
        ensure_required_columns(df, fp)

        # normalize key fields
        df = attach_subject_int(df)
        df["DX_GROUP"] = safe_numeric(df["DX_GROUP"])
        df["module"] = safe_numeric(df["module"])
        df["node"] = safe_numeric(df["node"])

        df = df.dropna(subset=["subject_int", "DX_GROUP", "module"]).copy()
        df["subject_int"] = df["subject_int"].astype(int)
        df["DX_GROUP"] = df["DX_GROUP"].astype(int)
        df["module"] = df["module"].astype(int)

        # attach SITE_ID
        df["SITE_ID"] = df["subject_int"].map(site_map)
        n_missing_site = int(df["SITE_ID"].isna().sum())
        df = df.dropna(subset=["SITE_ID"]).copy()
        df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

        present_metrics = [m for m in METRICS if m in df.columns]
        if not present_metrics:
            print(f"[WARN] {fp.name}: none of METRICS present, skipping.")
            continue

        k_max = int(df["module"].max())
        if k_max != EXPECTED_K:
            print(f"[WARN] {fp.name}: module max={k_max} (expected {EXPECTED_K})")

        out_rows = []

        # subject-level aggregation FIRST (per subject per module mean)
        for metric in present_metrics:
            tmp = df[["subject_int", "DX_GROUP", "SITE_ID", "module", metric]].copy()
            tmp = tmp.rename(columns={metric: "value"})
            tmp["value"] = safe_numeric(tmp["value"])
            tmp = tmp.dropna(subset=["value"])

            subj_mod = (
                tmp.groupby(["subject_int", "DX_GROUP", "SITE_ID", "module"], as_index=False)["value"]
                .mean()
            )

            # dx_ctl: 1 for control, 0 for ASD
            subj_mod["dx_ctl"] = (subj_mod["DX_GROUP"] == 2).astype(int)

            modules = sorted(subj_mod["module"].unique().tolist())
            for m in modules:
                s = subj_mod[subj_mod["module"] == m].copy()

                n_asd = int(s.loc[s["DX_GROUP"] == 1, "subject_int"].nunique())
                n_ctl = int(s.loc[s["DX_GROUP"] == 2, "subject_int"].nunique())
                mean_asd = float(s.loc[s["DX_GROUP"] == 1, "value"].mean()) if n_asd > 0 else np.nan
                mean_ctl = float(s.loc[s["DX_GROUP"] == 2, "value"].mean()) if n_ctl > 0 else np.nan

                note_parts = []
                if n_missing_site > 0:
                    note_parts.append(f"dropped_missing_site_rows={n_missing_site}")
                if s["SITE_ID"].nunique() < 2:
                    note_parts.append("only_one_site_in_this_module_metric")
                if (n_asd < 3) or (n_ctl < 3):
                    note_parts.append("too_few_subjects")

                beta = p_dx = p_site = r2 = np.nan

                can_fit = (n_asd >= 3) and (n_ctl >= 3) and (s["SITE_ID"].nunique() >= 2)
                if can_fit:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            beta, p_dx, p_site, r2 = run_sitecov_model(s)
                        except Exception as e:
                            note_parts.append(f"model_fail:{type(e).__name__}")

                out_rows.append({
                    "scenario": SCENARIO,
                    "sex": sex,
                    "age_group": age,
                    "metric": metric,
                    "module": int(m),
                    "mean_ASD": mean_asd,
                    "mean_CTL": mean_ctl,
                    "beta_CTL_minus_ASD": beta,
                    "p_DX": p_dx,
                    "p_SITE": p_site,
                    "r2": r2,
                    "n_ASD": n_asd,
                    "n_CTL": n_ctl,
                    "note": ";".join(note_parts) if note_parts else "",
                })

        out = pd.DataFrame(out_rows)

        # FDR within each metric across modules
        out["p_DX_FDR"] = np.nan
        out["DX_FDR_significant"] = False

        for metric, subm in out.groupby("metric"):
            qvals = bh_fdr(subm["p_DX"].tolist())
            out.loc[subm.index, "p_DX_FDR"] = qvals
            out.loc[subm.index, "DX_FDR_significant"] = [q <= 0.05 for q in qvals]

        out = out.sort_values(["metric", "module"]).reset_index(drop=True)

        out_path = OUT_DIR / f"{SCENARIO}__{sex}__{age}__module_stats_sitecov.csv"
        out.to_csv(out_path, index=False)

        print(f"[SAVED] {out_path} ({len(out)} rows)")

    print("\n[DONE] ABIDE12 module_stats_sitecov generated.")


if __name__ == "__main__":
    main()