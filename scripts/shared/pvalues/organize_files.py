# scripts/shared/pvalues/organize_files.py

import shutil
from pathlib import Path
import re
import pandas as pd

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

SRC_STATS  = ROOT / "results" / "hubs" / "module_stats_sitecov"
SRC_TABLES = ROOT / "results" / "hubs" / "module_stats_sitecov_tables"
DEST       = ROOT / "results" / "hubs_organized"

# -------------------------
# Safety switches
# -------------------------
DRY_RUN = False        # True = preview only
MOVE_TABLES = True     # True = move table CSVs, False = copy (recommended until you're sure)
OVERWRITE = True       # True = overwrite destination files if they exist

SEXES = {"female", "male"}
AGE_GROUPS = {"child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"}

# Canonical metrics we organize into folders (and their aliases we accept)
METRIC_SPECS = {
    "PC": ["PC", "pc", "PC_abs", "pc_abs"],
    "PC_pos": ["PC_pos", "pc_pos", "PCpos", "pcpos"],
    "PC_neg": ["PC_neg", "pc_neg", "PCneg", "pcneg"],

    "Z": ["Z", "z", "Z_abs", "z_abs"],
    "Z_pos": ["Z_pos", "z_pos", "Zpos", "zpos"],
    "Z_neg": ["Z_neg", "z_neg", "Zneg", "zneg"],

    "Strength_pos": ["Strength_pos", "strength_pos", "STR_pos", "str_pos"],
    "Strength_neg": ["Strength_neg", "strength_neg", "STR_neg", "str_neg"],
}
METRIC_ORDER = ["PC", "PC_pos", "PC_neg", "Z", "Z_pos", "Z_neg", "Strength_pos", "Strength_neg"]

# alias -> canonical
ALIAS_TO_CANON = {}
for canon, aliases in METRIC_SPECS.items():
    for a in aliases:
        ALIAS_TO_CANON[str(a).strip().lower()] = canon

def normalize_metric(m: str):
    if m is None:
        return None
    return ALIAS_TO_CANON.get(str(m).strip().lower(), None)

def metric_path_parts(metric_canon: str):
    """
    Folder structure:
      PC/all, PC/pos, PC/neg
      Z/all,  Z/pos,  Z/neg
      Strength/pos, Strength/neg
    """
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

    return (metric_canon, "all")

def ensure_dir(p: Path):
    if DRY_RUN:
        return
    p.mkdir(parents=True, exist_ok=True)

def safe_remove_if_exists(p: Path):
    if not p.exists():
        return
    if not OVERWRITE:
        raise FileExistsError(f"Destination exists and OVERWRITE=False: {p}")
    if DRY_RUN:
        print(f"[RM] {p}")
        return
    p.unlink()

def do_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if dst.exists():
        safe_remove_if_exists(dst)
    if DRY_RUN:
        print(f"[COPY] {src} -> {dst}")
        return
    shutil.copy2(src, dst)

def do_move(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if dst.exists():
        safe_remove_if_exists(dst)
    if DRY_RUN:
        print(f"[MOVE] {src} -> {dst}")
        return
    shutil.move(str(src), str(dst))

def parse_stats_filename(name: str):
    """
    Expect:
      scenario__sex__age__module_stats_sitecov.csv
    Example:
      ABIDE1_fd2__male__adult_18_plus__module_stats_sitecov.csv
    Returns (scenario, sex, age) or None
    """
    stem = Path(name).stem
    parts = stem.split("__")
    if len(parts) != 4:
        return None
    scenario, sex, age, tail = parts
    if tail != "module_stats_sitecov":
        return None
    if sex not in SEXES:
        return None
    if age not in AGE_GROUPS:
        return None
    if not scenario:
        return None
    return scenario, sex, age

def parse_table_filename(name: str):
    """
    Expect:
      scenario__sex__age__metric__table.csv
    Example:
      ABIDE1_fd2__male__adult_18_plus__z_pos__table.csv
    Returns (scenario, sex, age, metric_canon) or None
    """
    stem = Path(name).stem
    parts = stem.split("__")
    if len(parts) != 5:
        return None
    scenario, sex, age, metric_raw, tail = parts
    if tail.lower() != "table":
        return None
    if sex not in SEXES or age not in AGE_GROUPS or not scenario:
        return None

    metric_canon = normalize_metric(metric_raw)
    if metric_canon is None:
        return None
    return scenario, sex, age, metric_canon

def organize_stats():
    if not SRC_STATS.exists():
        print(f"[SKIP stats] missing {SRC_STATS}")
        return 0

    stats_files = sorted(SRC_STATS.glob("*__*__*__module_stats_sitecov.csv"))
    if not stats_files:
        print(f"[WARN] No stats CSVs found under: {SRC_STATS}")
        return 0

    wrote = 0
    for src in stats_files:
        parsed = parse_stats_filename(src.name)
        if parsed is None:
            print(f"[WARN] Unrecognized stats filename (skipping): {src.name}")
            continue
        scenario, sex, age = parsed

        df = pd.read_csv(src)
        df.columns = df.columns.str.strip()

        # Full copy
        full_dst = DEST / scenario / "_full" / age / sex / "module_stats_sitecov_FULL.csv"
        do_copy(src, full_dst)
        wrote += 1

        if "metric" not in df.columns:
            print(f"[WARN] stats file missing 'metric' column (cannot split): {src}")
            continue

        df["metric"] = df["metric"].astype(str).str.strip()
        df["_metric_canon"] = df["metric"].apply(normalize_metric)

        for metric_canon in METRIC_ORDER:
            sub = df[df["_metric_canon"] == metric_canon].copy()
            if sub.empty:
                continue

            sub = sub.drop(columns=["_metric_canon"], errors="ignore")
            metric_folder, posneg = metric_path_parts(metric_canon)
            dst = DEST / scenario / metric_folder / posneg / age / sex / "module_stats_sitecov.csv"

            ensure_dir(dst.parent)
            if dst.exists():
                safe_remove_if_exists(dst)

            if DRY_RUN:
                print(f"[WRITE] {scenario} {sex} {age} {metric_canon} -> {dst}")
            else:
                sub.to_csv(dst, index=False)
            wrote += 1

    return wrote

def organize_tables():
    if not SRC_TABLES.exists():
        print(f"[SKIP tables] missing {SRC_TABLES}")
        return 0

    table_files = sorted(SRC_TABLES.glob("*.csv"))
    if not table_files:
        print(f"[WARN] No table CSVs found under: {SRC_TABLES} "
              f"(if you already ran once with MOVE_TABLES=True, they were moved)")
        return 0

    moved = 0
    for src in table_files:
        parsed = parse_table_filename(src.name)
        if parsed is None:
            continue

        scenario, sex, age, metric_canon = parsed
        metric_folder, posneg = metric_path_parts(metric_canon)

        dst = DEST / scenario / metric_folder / posneg / age / sex / "table.csv"
        if MOVE_TABLES:
            do_move(src, dst)
        else:
            do_copy(src, dst)
        moved += 1

    return moved

def main():
    print(f"DRY_RUN={DRY_RUN} MOVE_TABLES={MOVE_TABLES} OVERWRITE={OVERWRITE}")
    print(f"SRC_STATS={SRC_STATS}")
    print(f"SRC_TABLES={SRC_TABLES}")
    print(f"DEST={DEST}")

    n_stats = organize_stats()
    n_tables = organize_tables()

    print(f"\n[SUMMARY] wrote/copies from stats: {n_stats}")
    print(f"[SUMMARY] moved/copied tables: {n_tables}")
    print("Done.")

if __name__ == "__main__":
    main()