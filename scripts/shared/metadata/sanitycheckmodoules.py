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

MODULE_FILES = {
    "fd-0.2": ROOT / "results" / "group_connectomes" / "ABIDE12_CC200" / "ABIDE_modules_asym_min10_fd-0.2.txt",
    "fd-0.3": ROOT / "results" / "group_connectomes" / "ABIDE12_CC200" / "ABIDE_modules_asym_min10_fd-0.3.txt",
}

NODE_FILES = {
    "OVERALL_ageSexMatched_fd-0.2": ROOT / "results" / "hubs" / "pc_z_strength_sitecov" / "OVERALL_ageSexMatched_fd-0.2_node_metrics.csv",
    "OVERALL_ageSexMatched_fd-0.3": ROOT / "results" / "hubs" / "pc_z_strength_sitecov" / "OVERALL_ageSexMatched_fd-0.3_node_metrics.csv",
}

STATS_DIR = ROOT / "results" / "hubs" / "module_stats_sitecov"


def audit_module_file(label: str, fp: Path):
    print(f"\n=== module file audit: {label} ===")
    print(f"path: {fp}")

    if not fp.exists():
        print("[MISSING]")
        return

    df = pd.read_csv(fp, sep=r"\s+")
    df.columns = [str(c).strip() for c in df.columns]

    if "Module" not in df.columns:
        print("[ERROR] no 'Module' column")
        return

    mods = pd.to_numeric(df["Module"], errors="coerce").dropna().astype(int)
    uniq = sorted(mods.unique().tolist())

    print(f"unique modules: {uniq}")
    print(f"n unique modules: {len(uniq)}")
    print("roi counts per module:")
    print(mods.value_counts().sort_index().to_string())


def audit_node_file(label: str, fp: Path):
    print(f"\n=== node metrics audit: {label} ===")
    print(f"path: {fp}")

    if not fp.exists():
        print("[MISSING]")
        return

    df = pd.read_csv(fp)
    df.columns = [str(c).strip() for c in df.columns]

    if "module" not in df.columns:
        print("[ERROR] no 'module' column")
        return

    mods = pd.to_numeric(df["module"], errors="coerce").dropna().astype(int)
    uniq = sorted(mods.unique().tolist())

    print(f"unique modules in node metrics: {uniq}")
    print(f"n unique modules: {len(uniq)}")

    subj_mod = (
        df[["SUB_ID", "module"]]
        .dropna()
        .assign(module=lambda x: pd.to_numeric(x["module"], errors="coerce"))
        .dropna()
    )
    subj_mod["module"] = subj_mod["module"].astype(int)

    print("subject-node rows per module:")
    print(subj_mod["module"].value_counts().sort_index().to_string())


def audit_stats_outputs():
    print(f"\n=== stats output audit ===")
    print(f"path: {STATS_DIR}")

    if not STATS_DIR.exists():
        print("[MISSING stats dir]")
        return

    files = sorted(STATS_DIR.glob("*__module_stats_sitecov.csv"))
    if not files:
        print("[NO stats files found]")
        return

    problems = []

    for fp in files:
        df = pd.read_csv(fp)
        df.columns = [str(c).strip() for c in df.columns]

        if "module" not in df.columns:
            problems.append((fp.name, "missing module column", None))
            continue

        mods = pd.to_numeric(df["module"], errors="coerce").dropna().astype(int)
        uniq = sorted(mods.unique().tolist())

        if 8 not in uniq:
            problems.append((fp.name, "module 8 missing", uniq))

    if not problems:
        print("all stats files include module 8")
    else:
        print("files with possible module-8 issue:")
        for name, issue, uniq in problems:
            print(f"  {name}")
            print(f"    issue: {issue}")
            print(f"    modules present: {uniq}")


def main():
    print(f"[INFO] repo root: {ROOT}")

    for label, fp in MODULE_FILES.items():
        audit_module_file(label, fp)

    for label, fp in NODE_FILES.items():
        audit_node_file(label, fp)

    audit_stats_outputs()

    print("\nexpected dominant yeo mapping from your screenshot:")
    print("  module 1 -> Somatomotor")
    print("  module 2 -> Visual")
    print("  module 3 -> Limbic")
    print("  module 4 -> Frontoparietal")
    print("  module 5 -> VentralAttention")
    print("  module 6 -> Visual")
    print("  module 7 -> DefaultMode")
    print("  module 8 -> DorsalAttention")


if __name__ == "__main__":
    main()