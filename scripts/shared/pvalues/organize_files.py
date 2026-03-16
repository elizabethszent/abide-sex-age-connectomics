import shutil
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

SRC_STATS = ROOT / "results" / "hubs" / "module_stats_sitecov"
SRC_TABLES = ROOT / "results" / "hubs" / "module_stats_sitecov_tables"
DEST = ROOT / "results" / "hubs_organized"

DRY_RUN = False
MOVE_TABLES = True
OVERWRITE = True

METRIC_SPECS = {
    "PC": ["PC", "pc"],
    "PC_pos": ["PC_pos", "pc_pos"],
    "PC_neg": ["PC_neg", "pc_neg"],
    "Z": ["Z", "z"],
    "Z_pos": ["Z_pos", "z_pos"],
    "Z_neg": ["Z_neg", "z_neg"],
    "Strength_pos": ["Strength_pos", "strength_pos"],
    "Strength_neg": ["Strength_neg", "strength_neg"],
}

METRIC_ORDER = [
    "PC",
    "PC_pos",
    "PC_neg",
    "Z",
    "Z_pos",
    "Z_neg",
    "Strength_pos",
    "Strength_neg",
]

MODEL_FOLDERS = {
    "m1": "site",
    "m2": "site_iq",
    "m3": "site_iq_rh",
}

ALIAS_TO_CANON = {}
for canon, aliases in METRIC_SPECS.items():
    for a in aliases:
        ALIAS_TO_CANON[a.lower()] = canon


def normalize_metric(m: str):
    if m is None:
        return None
    return ALIAS_TO_CANON.get(str(m).strip().lower(), None)


def normalize_model(m: str):
    if m is None:
        return None
    m = str(m).strip()
    return m if m in MODEL_FOLDERS else None


def metric_path_parts(metric_canon: str):
    if metric_canon == "Strength_pos":
        return ("Strength", "pos")
    if metric_canon == "Strength_neg":
        return ("Strength", "neg")
    if metric_canon == "PC":
        return ("PC", "all")
    if metric_canon == "PC_pos":
        return ("PC", "pos")
    if metric_canon == "PC_neg":
        return ("PC", "neg")
    if metric_canon == "Z":
        return ("Z", "all")
    if metric_canon == "Z_pos":
        return ("Z", "pos")
    if metric_canon == "Z_neg":
        return ("Z", "neg")
    raise ValueError(f"Unknown metric: {metric_canon}")


def ensure_dir(p: Path):
    if not DRY_RUN:
        p.mkdir(parents=True, exist_ok=True)


def safe_remove_if_exists(p: Path):
    if not p.exists():
        return
    if not OVERWRITE:
        raise FileExistsError(f"Destination already exists and OVERWRITE=False: {p}")
    if DRY_RUN:
        print(f"[RM] {p}")
    else:
        p.unlink()


def do_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if dst.exists():
        safe_remove_if_exists(dst)
    if DRY_RUN:
        print(f"[COPY] {src} -> {dst}")
    else:
        shutil.copy2(src, dst)


def do_move(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if dst.exists():
        safe_remove_if_exists(dst)
    if DRY_RUN:
        print(f"[MOVE] {src} -> {dst}")
    else:
        shutil.move(str(src), str(dst))


def parse_stats_filename(fp: Path):
    stem = fp.stem
    suffix = "__module_stats_sitecov"
    if not stem.endswith(suffix):
        return None
    prefix = stem[:-len(suffix)]
    parts = prefix.split("__")
    if len(parts) != 3:
        return None
    scenario, sex, age = parts
    return scenario, sex, age


def parse_table_filename(fp: Path):
    stem = fp.stem
    suffix = "__table"
    if not stem.endswith(suffix):
        return None

    prefix = stem[:-len(suffix)]
    parts = prefix.split("__")

    if len(parts) == 5:
        scenario, sex, age, metric_raw, model_raw = parts
        metric_canon = normalize_metric(metric_raw)
        model_key = normalize_model(model_raw)
        if metric_canon is None or model_key is None:
            return None
        return scenario, sex, age, metric_canon, model_key

    return None


def metric_dest_dir(scenario: str, sex: str, age: str, metric_canon: str, model_key: str) -> Path:
    metric_folder, sign_folder = metric_path_parts(metric_canon)
    model_folder = MODEL_FOLDERS[model_key]
    return DEST / scenario / metric_folder / sign_folder / age / sex / model_folder


def build_model_specific_stats(sub: pd.DataFrame, model_key: str) -> pd.DataFrame | None:
    base_cols = [
        "scenario",
        "sex",
        "age_group",
        "metric",
        "module",
        "mean_ASD",
        "mean_CTL",
        "n_ASD",
        "n_CTL",
    ]

    model_cols = [
        f"beta_CTL_minus_ASD_{model_key}",
        f"p_DX_{model_key}",
        f"p_DX_FDR_{model_key}",
        f"DX_FDR_significant_{model_key}",
        f"p_SITE_{model_key}",
        f"p_IQ_{model_key}",
        f"p_RIGHT_HANDED_{model_key}",
        f"n_model_{model_key}",
        f"n_model_ASD_{model_key}",
        f"n_model_CTL_{model_key}",
        f"note_{model_key}",
    ]

    if not any(c in sub.columns for c in model_cols):
        return None

    present = [c for c in base_cols + model_cols if c in sub.columns]
    out = sub[present].copy()

    rename_map = {
        f"beta_CTL_minus_ASD_{model_key}": "beta_CTL_minus_ASD",
        f"p_DX_{model_key}": "p_DX",
        f"p_DX_FDR_{model_key}": "p_DX_FDR",
        f"DX_FDR_significant_{model_key}": "DX_FDR_significant",
        f"p_SITE_{model_key}": "p_SITE",
        f"p_IQ_{model_key}": "p_IQ",
        f"p_RIGHT_HANDED_{model_key}": "p_RIGHT_HANDED",
        f"n_model_{model_key}": "n_model",
        f"n_model_ASD_{model_key}": "n_model_ASD",
        f"n_model_CTL_{model_key}": "n_model_CTL",
        f"note_{model_key}": "note",
    }
    out = out.rename(columns=rename_map)

    out["model"] = model_key
    out["model_label"] = MODEL_FOLDERS[model_key]

    if "module" in out.columns:
        out["module"] = pd.to_numeric(out["module"], errors="coerce")
        out = out.sort_values("module")

    return out


def organize_stats():
    if not SRC_STATS.exists():
        print(f"[SKIP stats] missing {SRC_STATS}")
        return

    stat_files = sorted(SRC_STATS.glob("*__module_stats_sitecov.csv"))
    if not stat_files:
        print(f"[SKIP stats] no stats files in {SRC_STATS}")
        return

    for src in stat_files:
        parsed = parse_stats_filename(src)
        if parsed is None:
            print(f"[WARN] could not parse stats filename: {src.name}")
            continue

        scenario, sex, age = parsed

        df = pd.read_csv(src)
        df.columns = df.columns.str.strip()

        if "metric" not in df.columns:
            print(f"[WARN] stats file missing metric column: {src.name}")
            continue

        df["_metric_canon"] = df["metric"].apply(normalize_metric)

        for metric_canon in METRIC_ORDER:
            sub_metric = df[df["_metric_canon"] == metric_canon].drop(columns=["_metric_canon"], errors="ignore")
            if sub_metric.empty:
                continue

            for model_key in MODEL_FOLDERS:
                sub_model = build_model_specific_stats(sub_metric, model_key)
                if sub_model is None or sub_model.empty:
                    continue

                dst_dir = metric_dest_dir(scenario, sex, age, metric_canon, model_key)
                dst = dst_dir / "module_stats_sitecov.csv"

                ensure_dir(dst_dir)
                if dst.exists():
                    safe_remove_if_exists(dst)

                if DRY_RUN:
                    print(f"[WRITE] {src.name} [{metric_canon} {model_key}] -> {dst}")
                else:
                    sub_model.to_csv(dst, index=False)


def organize_tables():
    if not SRC_TABLES.exists():
        print(f"[SKIP tables] missing {SRC_TABLES}")
        return

    table_files = sorted(SRC_TABLES.glob("*__table.csv"))
    if not table_files:
        print(f"[SKIP tables] no table files in {SRC_TABLES}")
        return

    for src in table_files:
        parsed = parse_table_filename(src)
        if parsed is None:
            print(f"[WARN] could not parse table filename: {src.name}")
            continue

        scenario, sex, age, metric_canon, model_key = parsed
        dst_dir = metric_dest_dir(scenario, sex, age, metric_canon, model_key)
        dst = dst_dir / "table.csv"

        if MOVE_TABLES:
            do_move(src, dst)
        else:
            do_copy(src, dst)


def main():
    print(f"DRY_RUN={DRY_RUN} MOVE_TABLES={MOVE_TABLES} OVERWRITE={OVERWRITE}")
    print(f"ROOT={ROOT}")
    print(f"SRC_STATS={SRC_STATS}")
    print(f"SRC_TABLES={SRC_TABLES}")
    print(f"DEST={DEST}")

    organize_stats()
    organize_tables()

    print("\nDone.")


if __name__ == "__main__":
    main()