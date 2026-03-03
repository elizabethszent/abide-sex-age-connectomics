import shutil
import re
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

SRC_STATS  = ROOT / "results" / "hubs" / "module_stats_sitecov"
SRC_TABLES = ROOT / "results" / "hubs" / "module_stats_sitecov_tables"
DEST       = ROOT / "results" / "hubs_organized"

# -------------------------
# Safety switches
# -------------------------
DRY_RUN = False        # True = preview only
MOVE_TABLES = True     # True = move table CSVs, False = copy
OVERWRITE = True       # True = overwrite existing destination files

SCENARIOS = [
    "OVERALL_sexbalanced_fd-0.2",
    "OVERALL_sexbalanced_fd-0.3",
    "OVERALL_ageSexMatched_fd-0.2",
    "OVERALL_ageSexMatched_fd-0.3",
]

SEXES = ["female", "male"]
AGE_GROUPS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]

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

# Build alias->canonical map (case-insensitive)
ALIAS_TO_CANON = {}
for canon, aliases in METRIC_SPECS.items():
    for a in aliases:
        ALIAS_TO_CANON[a.lower()] = canon

def normalize_metric(m: str):
    if m is None:
        return None
    return ALIAS_TO_CANON.get(str(m).strip().lower(), None)

def metric_path_parts(metric_canon: str):
    """
    Folder structure:
      PC/all
      PC/pos
      PC/neg
      Z/all
      Z/pos
      Z/neg
      Strength/pos
      Strength/neg
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
        raise FileExistsError(f"Destination already exists and OVERWRITE=False: {p}")
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

def parse_table_filename(name: str):
    """
    Supports BOTH:
      1) scenario__sex__age__metric__table.csv
      2) scenario_sex_age_metric_table.csv

    Returns (scenario, sex, age, metric_canon) or None.
    """
    stem = Path(name).stem

    # Style 1: double-underscore
    if "__" in stem:
        parts = stem.split("__")
        # expected: scenario, sex, age, metric, table
        if len(parts) >= 5:
            scenario, sex, age, metric_raw = parts[0], parts[1], parts[2], parts[3]
            metric_canon = normalize_metric(metric_raw)
            if scenario in SCENARIOS and sex in SEXES and age in AGE_GROUPS and metric_canon in METRIC_SPECS:
                return scenario, sex, age, metric_canon

    # Style 2: underscore style
    # Example: OVERALL_ageSexMatched_fd-0.2_female_adult_18_plus_PC_pos_table
    metric_pat = r"(PC|PC_pos|PC_neg|Z|Z_pos|Z_neg|Strength_pos|Strength_neg)"
    m = re.match(
        rf"^(OVERALL_(?:sexbalanced|ageSexMatched)_fd-\d\.\d)_(female|male)_(child_0_9|preteen_10_12|teen_13_17|adult_18_plus)_{metric_pat}_table$",
        stem
    )
    if m:
        scenario, sex, age, metric_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        metric_canon = normalize_metric(metric_raw)
        if scenario in SCENARIOS and metric_canon in METRIC_SPECS:
            return scenario, sex, age, metric_canon

    return None

def organize_stats():
    """
    Copies each stats file:
      scenario__sex__age__module_stats_sitecov.csv

    into:
      DEST/scenario/<metric_folder>/<posneg>/<age>/<sex>/module_stats_sitecov.csv

    and also keeps a FULL copy at:
      DEST/scenario/_full/<age>/<sex>/module_stats_sitecov_FULL.csv
    """
    if not SRC_STATS.exists():
        print(f"[SKIP stats] missing {SRC_STATS}")
        return

    for scenario in SCENARIOS:
        for sex in SEXES:
            for age in AGE_GROUPS:
                src = SRC_STATS / f"{scenario}__{sex}__{age}__module_stats_sitecov.csv"
                if not src.exists():
                    continue

                df = pd.read_csv(src)
                df.columns = df.columns.str.strip()
                if "metric" not in df.columns:
                    print(f"[WARN] stats file missing 'metric': {src}")
                    continue

                # full copy
                full_dst = DEST / scenario / "_full" / age / sex / "module_stats_sitecov_FULL.csv"
                do_copy(src, full_dst)

                # normalize metrics in file so z/Z etc work
                df["_metric_canon"] = df["metric"].apply(normalize_metric)

                # split by canonical metric
                for metric_canon in METRIC_ORDER:
                    sub = df[df["_metric_canon"] == metric_canon].drop(columns=["_metric_canon"], errors="ignore")
                    if sub.empty:
                        continue

                    metric_folder, posneg = metric_path_parts(metric_canon)
                    dst = DEST / scenario / metric_folder / posneg / age / sex / "module_stats_sitecov.csv"

                    ensure_dir(dst.parent)
                    if dst.exists():
                        safe_remove_if_exists(dst)

                    if DRY_RUN:
                        print(f"[WRITE] {metric_canon} subset -> {dst}")
                    else:
                        sub.to_csv(dst, index=False)

def organize_tables():
    """
    Moves/copies table files into:
      DEST/scenario/<metric_folder>/<posneg>/<age>/<sex>/table.csv
    """
    if not SRC_TABLES.exists():
        print(f"[SKIP tables] missing {SRC_TABLES}")
        return

    for src in sorted(SRC_TABLES.glob("*.csv")):
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

def main():
    print(f"DRY_RUN={DRY_RUN} MOVE_TABLES={MOVE_TABLES} OVERWRITE={OVERWRITE}")
    print(f"SRC_STATS={SRC_STATS}")
    print(f"SRC_TABLES={SRC_TABLES}")
    print(f"DEST={DEST}")

    organize_stats()
    organize_tables()

    print("\nDone.")

if __name__ == "__main__":
    main()