import pandas as pd
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        has_results = (p / "results").exists()
        has_meta = (p / "phenotypes").exists() or (p / "data").exists()
        if has_results and has_meta:
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)
SCAN_DIR = ROOT / "results" / "hubs_organized"
OUT_CSV = ROOT / "results" / "qc" / "significant_module_summary.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

INCLUDE_TABLES = False
FDR_ALPHA = 0.05

MODULE_TO_YEO = {
    "OVERALL_ageSexMatched_fd-0.3": {
        1: "Somatomotor",
        2: "Visual",
        3: "Limbic",
        4: "Frontoparietal",
        5: "VentralAttention",
        6: "Visual",
        7: "DefaultMode",
        8: "DorsalAttention",
    },
    "OVERALL_ageSexMatched_fd-0.2": {
        1: "unmapped_fd0p2",
        2: "unmapped_fd0p2",
        3: "unmapped_fd0p2",
        4: "unmapped_fd0p2",
        5: "unmapped_fd0p2",
        6: "unmapped_fd0p2",
        7: "unmapped_fd0p2",
        8: "unmapped_fd0p2",
    },
}


def metric_from_parts(metric_folder: str, sign_folder: str) -> str:
    if metric_folder == "PC":
        if sign_folder == "all":
            return "PC"
        if sign_folder == "pos":
            return "PC_pos"
        if sign_folder == "neg":
            return "PC_neg"
    if metric_folder == "Z":
        if sign_folder == "all":
            return "Z"
        if sign_folder == "pos":
            return "Z_pos"
        if sign_folder == "neg":
            return "Z_neg"
    if metric_folder == "Strength":
        if sign_folder == "pos":
            return "Strength_pos"
        if sign_folder == "neg":
            return "Strength_neg"
    return f"{metric_folder}_{sign_folder}"


def parse_file_context(fp: Path) -> dict | None:
    rel = fp.relative_to(SCAN_DIR)
    parts = rel.parts

    if len(parts) < 7:
        return None

    scenario = parts[0]
    metric_folder = parts[1]
    sign_folder = parts[2]
    age_group = parts[3]
    sex = parts[4]
    model = parts[5]
    filename = parts[6]

    metric = metric_from_parts(metric_folder, sign_folder)

    return {
        "scenario": scenario,
        "metric_folder": metric_folder,
        "sign_folder": sign_folder,
        "metric": metric,
        "age_group": age_group,
        "sex": sex,
        "model": model,
        "filename": filename,
    }


def scan_one_file(fp: Path) -> list[dict]:
    ctx = parse_file_context(fp)
    if ctx is None:
        return []

    df = pd.read_csv(fp)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"module", "p_DX", "p_DX_FDR", "beta_CTL_minus_ASD"}
    if not required.issubset(df.columns):
        return []

    df["module"] = pd.to_numeric(df["module"], errors="coerce")
    df["p_DX"] = pd.to_numeric(df["p_DX"], errors="coerce")
    df["p_DX_FDR"] = pd.to_numeric(df["p_DX_FDR"], errors="coerce")
    df["beta_CTL_minus_ASD"] = pd.to_numeric(df["beta_CTL_minus_ASD"], errors="coerce")

    hits = df[df["p_DX_FDR"] <= FDR_ALPHA].copy()
    if hits.empty:
        return []

    module_map = MODULE_TO_YEO.get(ctx["scenario"], {})

    rows = []
    for _, row in hits.iterrows():
        module = int(row["module"])
        rows.append(
            {
                "full_path": str(fp.resolve()),
                "file": fp.name,
                "scenario": ctx["scenario"],
                "metric_folder": ctx["metric_folder"],
                "sign_folder": ctx["sign_folder"],
                "metric": ctx["metric"],
                "age_group": ctx["age_group"],
                "sex": ctx["sex"],
                "model": ctx["model"],
                "module": module,
                "module_yeo_label": module_map.get(module, "unmapped"),
                "beta_CTL_minus_ASD": row["beta_CTL_minus_ASD"],
                "p_DX": row["p_DX"],
                "p_DX_FDR": row["p_DX_FDR"],
            }
        )
    return rows


def main():
    print(f"[INFO] scanning: {SCAN_DIR}")

    if not SCAN_DIR.exists():
        raise FileNotFoundError(f"Missing scan dir: {SCAN_DIR}")

    patterns = ["module_stats_sitecov.csv"]
    if INCLUDE_TABLES:
        patterns.append("table.csv")

    files = []
    for pattern in patterns:
        files.extend(sorted(SCAN_DIR.rglob(pattern)))

    all_rows = []
    for fp in files:
        all_rows.extend(scan_one_file(fp))

    if not all_rows:
        print("[INFO] no FDR-significant hits found")
        pd.DataFrame().to_csv(OUT_CSV, index=False)
        return

    out_df = pd.DataFrame(all_rows)
    out_df = out_df.sort_values(
        ["scenario", "metric", "age_group", "sex", "model", "module", "p_DX_FDR", "p_DX"]
    ).reset_index(drop=True)

    for _, row in out_df.iterrows():
        print()
        print(f"FULL PATH : {row['full_path']}")
        print(f"FILE      : {row['file']}")
        print(f"SCENARIO  : {row['scenario']}")
        print(f"METRIC    : {row['metric']}")
        print(f"AGE       : {row['age_group']}")
        print(f"SEX       : {row['sex']}")
        print(f"MODEL     : {row['model']}")
        print(f"MODULE    : {row['module']} ({row['module_yeo_label']})")
        print(f"BETA      : {row['beta_CTL_minus_ASD']}")
        print(f"p_DX      : {row['p_DX']}")
        print(f"p_DX_FDR  : {row['p_DX_FDR']}")
        print("-" * 80)

    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[SAVED] {OUT_CSV}")


if __name__ == "__main__":
    main()