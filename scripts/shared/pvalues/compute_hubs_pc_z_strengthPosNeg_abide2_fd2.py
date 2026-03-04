

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import statsmodels.formula.api as smf

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

HUB_DIR = ROOT / "results" / "hubs"
OUT_DIR = ROOT / "results" / "hubs" / "module_stats_sitecov"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COMBINED = ROOT / "data" / "metadata" / "ABIDE2_phenotypes_combined.csv"

SCENARIO = "ABIDE2_fd2"

# metrics we expect in the hub CSVs
METRICS = [
    "PC", "PC_pos", "PC_neg",
    "z", "z_pos", "z_neg",
    "strength_pos", "strength_neg",
]


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out_ok = np.empty_like(pv)
    out_ok[order] = q
    out[ok] = out_ok
    return out

def file_id_to_subject_int(file_id: str) -> int | None:
    s = str(file_id).strip()
    s = re.sub(r"\.0$", "", s)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None

def parse_hub_filename(fp: Path):
    """
    Expect something like:
      female_child_0_9_pc_z_strengthPosNeg_abide2_fd2.csv
    Returns (sex, age) or (None, None)
    """
    name = fp.name.lower()
    m = re.match(
        r"^(female|male)_(child_0_9|preteen_10_12|teen_13_17|adult_18_plus)_.*abide2.*\.csv$",
        name
    )
    if not m:
        return None, None
    return m.group(1), m.group(2)

def build_site_map(meta_csv: Path) -> dict[int, str]:
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing ABIDE2 combined metadata: {meta_csv}")

    df = pd.read_csv(meta_csv)
    df.columns = df.columns.str.strip().str.upper()

    required = {"SUB_ID", "SITE_ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{meta_csv} missing columns: {missing}")

    df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce")
    df = df.dropna(subset=["SUB_ID", "SITE_ID"]).copy()
    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

    # one site per subject
    df = df.sort_values(["SUB_ID", "SITE_ID"]).drop_duplicates(subset=["SUB_ID"], keep="first")
    return dict(zip(df["SUB_ID"].tolist(), df["SITE_ID"].tolist()))

def load_hub_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip()

    # tolerate case variants
    rename = {}
    for c in df.columns:
        if c.upper() == "DX_GROUP":
            rename[c] = "DX_GROUP"
        if c.upper() == "FILE_ID":
            rename[c] = "FILE_ID"
        if c.lower() == "age_group":
            rename[c] = "AGE_GROUP"
        if c.upper() == "AGE_GROUP":
            rename[c] = "AGE_GROUP"
        if c.lower() == "age_at_scan":
            rename[c] = "AGE_AT_SCAN"
        if c.upper() == "AGE_AT_SCAN":
            rename[c] = "AGE_AT_SCAN"
        if c.lower() == "module":
            rename[c] = "module"
        if c.lower() == "subject_int":
            rename[c] = "subject_int"
    if rename:
        df = df.rename(columns=rename)

    need = {"DX_GROUP", "module"}
    if not need.issubset(df.columns):
        raise ValueError(f"{fp} missing required columns: {need - set(df.columns)}")

    # subject_int: prefer column if present, else derive from FILE_ID
    if "subject_int" not in df.columns:
        if "FILE_ID" not in df.columns:
            raise ValueError(f"{fp} missing subject_int and FILE_ID (need at least one).")
        df["subject_int"] = df["FILE_ID"].apply(file_id_to_subject_int)

    df["subject_int"] = pd.to_numeric(df["subject_int"], errors="coerce")
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
    df["module"] = pd.to_numeric(df["module"], errors="coerce")

    df = df.dropna(subset=["subject_int", "DX_GROUP", "module"]).copy()
    df["subject_int"] = df["subject_int"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)
    df["module"] = df["module"].astype(int)

    # keep only the metric columns we have
    have_metrics = [m for m in METRICS if m in df.columns]
    if not have_metrics:
        raise ValueError(f"{fp} has none of the expected metric columns: {METRICS}")

    # numeric-ify metrics
    for m in have_metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    return df

def omnibus_p_site(res) -> float:
    """F-test p-value for all SITE dummy params = 0. Returns nan if not applicable."""
    site_params = [p for p in res.params.index if p.startswith("C(SITE_ID)[T.")]
    if len(site_params) == 0:
        return float("nan")
    # build restriction string: "C(SITE_ID)[T.X]=0, C(SITE_ID)[T.Y]=0"
    hyp = ", ".join([f"{p}=0" for p in site_params])
    try:
        return float(res.f_test(hyp).pvalue)
    except Exception:
        return float("nan")

# ---------- main stats ----------
def compute_module_stats(df_nodes: pd.DataFrame, site_map: dict[int, str], sex: str, age: str, src_name: str):
    # attach site
    df = df_nodes.copy()
    df["SITE_ID"] = df["subject_int"].map(site_map)

    n_missing_site = df["SITE_ID"].isna().sum()
    if n_missing_site > 0:
        # drop rows without site (can’t do site covariate)
        df = df.dropna(subset=["SITE_ID"]).copy()

    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

    # per-subject per-module means (collapse 200 nodes -> 7 module means per subject)
    grp_cols = ["subject_int", "DX_GROUP", "SITE_ID", "module"]
    agg = df.groupby(grp_cols, as_index=False)[[m for m in METRICS if m in df.columns]].mean()

    # dx_ctl: 1 for Control (DX_GROUP=2), 0 for ASD (DX_GROUP=1)
    agg["dx_ctl"] = (agg["DX_GROUP"] == 2).astype(int)

    rows = []
    unique_sites = agg["SITE_ID"].nunique()

    for metric in [m for m in METRICS if m in agg.columns]:
        for module in sorted(agg["module"].unique()):
            sub = agg[(agg["module"] == module)].copy()
            sub = sub.dropna(subset=[metric, "dx_ctl", "SITE_ID"])

            # counts
            n_ctl = int(sub[sub["dx_ctl"] == 1]["subject_int"].nunique())
            n_asd = int(sub[sub["dx_ctl"] == 0]["subject_int"].nunique())

            note_parts = []
            if n_missing_site > 0:
                note_parts.append(f"dropped_missing_site_rows={int(n_missing_site)}")
            if unique_sites < 2:
                note_parts.append("only_one_site")
            if (n_ctl < 5) or (n_asd < 5):
                note_parts.append("low_n")

            mean_ctl = float(sub[sub["dx_ctl"] == 1][metric].mean()) if n_ctl > 0 else float("nan")
            mean_asd = float(sub[sub["dx_ctl"] == 0][metric].mean()) if n_asd > 0 else float("nan")

            # default stats
            beta = float("nan")
            p_dx = float("nan")
            p_site = float("nan")
            r2 = float("nan")

            # regress if we have both groups and at least 2 sites
            can_fit = (n_ctl > 1) and (n_asd > 1) and (sub["SITE_ID"].nunique() >= 2)

            if can_fit:
                # suppress noisy warnings (you saw some r2 warnings before)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        model = smf.ols(f"{metric} ~ dx_ctl + C(SITE_ID)", data=sub).fit()
                        beta = float(model.params.get("dx_ctl", np.nan))
                        p_dx = float(model.pvalues.get("dx_ctl", np.nan))
                        p_site = omnibus_p_site(model)
                        r2 = float(model.rsquared)
                    except Exception as e:
                        note_parts.append(f"fit_fail:{type(e).__name__}")

            rows.append({
                "scenario": SCENARIO,
                "sex": sex,
                "age_group": age,
                "source_file": src_name,
                "module": int(module),
                "metric": metric,
                "mean_CTL": mean_ctl,
                "mean_ASD": mean_asd,
                "beta_CTL_minus_ASD": beta,
                "p_DX": p_dx,
                "p_SITE": p_site,
                "r2": r2,
                "n_ASD": n_asd,
                "n_CTL": n_ctl,
                "note": ";".join(note_parts) if note_parts else "",
            })

    out = pd.DataFrame(rows)

    # FDR correction per (sex, age, metric) across modules
    out["p_DX_FDR"] = np.nan
    out["DX_FDR_significant"] = False

    for metric in out["metric"].unique():
        mask = (out["metric"] == metric)
        q = bh_fdr(out.loc[mask, "p_DX"].to_numpy(dtype=float))
        out.loc[mask, "p_DX_FDR"] = q
        out.loc[mask, "DX_FDR_significant"] = (q < 0.05)

    # stable column order
    col_order = [
        "scenario", "sex", "age_group", "source_file",
        "module", "metric",
        "mean_CTL", "mean_ASD", "beta_CTL_minus_ASD",
        "p_DX", "p_DX_FDR", "DX_FDR_significant",
        "p_SITE", "r2",
        "n_ASD", "n_CTL",
        "note",
    ]
    out = out[col_order]
    return out

def main():
    # site map from your ABIDE2 combined metadata
    site_map = build_site_map(META_COMBINED)
    print(f"[INFO] SITE map entries: {len(site_map)} (from {META_COMBINED.name})")

    hub_files = sorted(HUB_DIR.glob("*abide2*fd2*.csv"))
    # be stricter: must look like your hub outputs
    hub_files = [fp for fp in hub_files if "pc_z_strengthposneg" in fp.name.lower()]

    if not hub_files:
        raise FileNotFoundError(
            f"No ABIDE2 hub CSVs found in {HUB_DIR}\n"
            f"Expected files like: female_child_0_9_pc_z_strengthPosNeg_abide2_fd2.csv"
        )

    print(f"[INFO] Found {len(hub_files)} hub CSV(s).")

    for fp in hub_files:
        sex, age = parse_hub_filename(fp)
        if sex is None:
            print(f"[WARN] Skipping unrecognized hub filename: {fp.name}")
            continue

        df_nodes = load_hub_csv(fp)
        stats = compute_module_stats(df_nodes, site_map, sex, age, fp.name)

        out_path = OUT_DIR / f"{SCENARIO}__{sex}__{age}__module_stats_sitecov.csv"
        stats.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path} ({len(stats)} rows)")

    print("\n[DONE] ABIDE2 module_stats_sitecov generated.")

if __name__ == "__main__":
    main()