# scripts/shared/pvalues/compute_module_stats_sitecov_from_hubs_abide1_fd2.py

import re
import numpy as np
import pandas as pd
from pathlib import Path

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

HUB_DIR = ROOT / "results" / "hubs"
OUT_DIR = HUB_DIR / "module_stats_sitecov"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHENO_DIR = ROOT / "phenotypes" / "ABIDE1"

# We will compute module stats for these metrics if present in hub CSVs.
METRICS = [
    "PC", "PC_pos", "PC_neg",
    "z", "z_pos", "z_neg",
    "strength_pos", "strength_neg",
]

# Expect modules 1..7 (but we’ll infer max from data)
EXPECTED_K = 7

# Hub files produced by your updated hubs script
HUB_GLOB = "*_pc_z_strengthPosNeg_abide1_fd2.csv"

# ----------------------------
# Utilities
# ----------------------------
def build_site_map(pheno_dir: Path) -> dict[int, str]:
    """
    Build map: SUB_ID (int) -> SITE_ID (str)
    using all phenotypic_*.csv files.
    """
    files = sorted(list(pheno_dir.glob("phenotypic_*.csv"))) + sorted(list(pheno_dir.glob("Phenotypic_*.csv")))
    if not files:
        raise FileNotFoundError(f"No phenotypic CSVs found in: {pheno_dir}")

    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip().str.upper()
        if not {"SUB_ID", "SITE_ID"}.issubset(df.columns):
            continue
        sub = df[["SUB_ID", "SITE_ID"]].copy()
        sub["SUB_ID"] = pd.to_numeric(sub["SUB_ID"], errors="coerce")
        sub = sub.dropna(subset=["SUB_ID", "SITE_ID"])
        sub["SUB_ID"] = sub["SUB_ID"].astype(int)
        sub["SITE_ID"] = sub["SITE_ID"].astype(str).str.strip()
        rows.append(sub)

    if not rows:
        raise RuntimeError(f"Could not build site map from {pheno_dir} (missing SUB_ID/SITE_ID)")

    allm = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["SUB_ID"], keep="first")
    return dict(zip(allm["SUB_ID"].tolist(), allm["SITE_ID"].tolist()))


def discover_hub_files(hub_dir: Path) -> list[Path]:
    files = sorted(hub_dir.glob(HUB_GLOB))
    return files


def parse_sex_age_from_filename(fp: Path) -> tuple[str, str] | None:
    # female_child_0_9_pc_z_strengthPosNeg_abide1_fd2.csv
    m = re.match(r"^(female|male)_(child_0_9|preteen_10_12|teen_13_17|adult_18_plus)_pc_z_strengthPosNeg_abide1_fd2\.csv$",
                 fp.name, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower(), m.group(2)


def ensure_required_columns(df: pd.DataFrame, fp: Path):
    need = {"subject_int", "DX_GROUP", "module", "node"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"{fp} missing required columns: {missing}. Found: {sorted(df.columns)}")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def run_sitecov_model(sub_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Fit:
      y ~ C(DX_GROUP, Treatment(reference=1)) + C(SITE_ID)

    Returns:
      beta (CTL-ASD), p_DX, p_SITE, r2
    """
    # enforce DX_GROUP baseline = 1 (ASD)
    sub_df = sub_df.copy()
    sub_df["DX_GROUP"] = pd.Categorical(sub_df["DX_GROUP"], categories=[1, 2])

    # full + reduced for p_SITE via nested model ANOVA
    full = smf.ols("value ~ C(DX_GROUP, Treatment(reference=1)) + C(SITE_ID)", data=sub_df).fit()
    red  = smf.ols("value ~ C(DX_GROUP, Treatment(reference=1))", data=sub_df).fit()

    dx_term = "C(DX_GROUP, Treatment(reference=1))[T.2]"

    beta = float(full.params.get(dx_term, np.nan))
    p_dx = float(full.pvalues.get(dx_term, np.nan))
    r2   = float(full.rsquared) if hasattr(full, "rsquared") else np.nan

    try:
        a = anova_lm(red, full)
        # second row is the comparison
        p_site = float(a["Pr(>F)"].iloc[1])
    except Exception:
        p_site = np.nan

    return beta, p_dx, p_site, r2


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.array([pv if np.isfinite(pv) else 1.0 for pv in pvals], dtype=float)
    _, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    return q.tolist()


# ----------------------------
# Main
# ----------------------------
def main():
    site_map = build_site_map(PHENO_DIR)
    hub_files = discover_hub_files(HUB_DIR)

    if not hub_files:
        raise FileNotFoundError(f"No hub CSVs found under: {HUB_DIR} matching {HUB_GLOB}")

    print(f"[INFO] Found {len(hub_files)} hub CSVs. SITE map entries: {len(site_map)}")

    for fp in hub_files:
        parsed = parse_sex_age_from_filename(fp)
        if parsed is None:
            continue
        sex, age = parsed

        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip()
        ensure_required_columns(df, fp)

        # normalize key fields
        df["subject_int"] = safe_numeric(df["subject_int"]).astype("Int64")
        df["DX_GROUP"]    = safe_numeric(df["DX_GROUP"]).astype("Int64")
        df["module"]      = safe_numeric(df["module"]).astype("Int64")

        df = df.dropna(subset=["subject_int", "DX_GROUP", "module"]).copy()
        df["subject_int"] = df["subject_int"].astype(int)
        df["DX_GROUP"]    = df["DX_GROUP"].astype(int)
        df["module"]      = df["module"].astype(int)

        # attach SITE_ID
        df["SITE_ID"] = df["subject_int"].map(site_map)
        df = df.dropna(subset=["SITE_ID"]).copy()
        df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

        # Which metrics actually exist in this hub file?
        present_metrics = [m for m in METRICS if m in df.columns]
        missing_metrics = [m for m in METRICS if m not in df.columns]
        if missing_metrics:
            print(f"[WARN] {fp.name}: missing metric columns: {missing_metrics}")
        if not present_metrics:
            print(f"[WARN] {fp.name}: no metric columns found, skipping.")
            continue

        k_max = int(df["module"].max())
        if k_max != EXPECTED_K:
            print(f"[WARN] {fp.name}: module max={k_max} (expected {EXPECTED_K})")

        out_rows = []

        # Compute per-subject per-module mean FIRST (so regression is at subject level)
        for metric in present_metrics:
            tmp = df[["subject_int", "DX_GROUP", "SITE_ID", "module", metric]].copy()
            tmp = tmp.rename(columns={metric: "value"})
            tmp["value"] = safe_numeric(tmp["value"])
            tmp = tmp.dropna(subset=["value"])

            subj_mod = (
                tmp.groupby(["subject_int", "DX_GROUP", "SITE_ID", "module"], as_index=False)["value"]
                   .mean()
            )

            # Build rows per module
            modules = sorted(subj_mod["module"].unique().tolist())
            for m in modules:
                s = subj_mod[subj_mod["module"] == m].copy()

                # group means
                mean_asd = float(s.loc[s["DX_GROUP"] == 1, "value"].mean()) if (s["DX_GROUP"] == 1).any() else np.nan
                mean_ctl = float(s.loc[s["DX_GROUP"] == 2, "value"].mean()) if (s["DX_GROUP"] == 2).any() else np.nan
                n_asd = int(s.loc[s["DX_GROUP"] == 1, "subject_int"].nunique())
                n_ctl = int(s.loc[s["DX_GROUP"] == 2, "subject_int"].nunique())

                note = ""
                beta = p_dx = p_site = r2 = np.nan

                # Need at least 2 groups and enough samples
                if n_asd < 3 or n_ctl < 3:
                    note = "too_few_subjects"
                else:
                    try:
                        beta, p_dx, p_site, r2 = run_sitecov_model(s)
                    except Exception as e:
                        note = f"model_fail:{type(e).__name__}"

                out_rows.append({
                    "scenario": "ABIDE1_fd2",
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
                    "note": note,
                })

        out = pd.DataFrame(out_rows)

        # FDR within each metric across modules
        out["p_DX_FDR"] = np.nan
        out["DX_FDR_significant"] = False

        for metric, subm in out.groupby("metric"):
            pvals = subm["p_DX"].tolist()
            qvals = bh_fdr(pvals)
            out.loc[subm.index, "p_DX_FDR"] = qvals
            out.loc[subm.index, "DX_FDR_significant"] = [q <= 0.05 for q in qvals]

        out = out.sort_values(["metric", "module"]).reset_index(drop=True)

        out_path = OUT_DIR / f"ABIDE1_fd2__{sex}__{age}__module_stats_sitecov.csv"
        out.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path} ({len(out)} rows)")

    print("\n[DONE] module_stats_sitecov generated.")


if __name__ == "__main__":
    main()