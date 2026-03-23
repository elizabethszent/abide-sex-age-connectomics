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

OUT_DIR = ROOT / "results" / "qc" / "module_yeo_lookup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FD02_MAP = {
    1: "Somatomotor",
    2: "Visual",
    3: "DefaultMode",
    4: "DorsalAttention",
    5: "Visual",
    6: "Frontoparietal",
    7: "Limbic",
    8: "VentralAttention",
}

FD03_MAP = {
    1: "Somatomotor",
    2: "Visual",
    3: "Limbic",
    4: "Frontoparietal",
    5: "VentralAttention",
    6: "Visual",
    7: "DefaultMode",
    8: "DorsalAttention",
}


def write_lookup(mapping: dict[int, str], threshold_label: str):
    df = pd.DataFrame(
        {
            "module": list(mapping.keys()),
            "dominant_yeo": list(mapping.values()),
            "yeo_mapping_threshold": [threshold_label] * len(mapping),
        }
    ).sort_values("module")

    out_path = OUT_DIR / f"module_to_yeo_{threshold_label}.csv"
    df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")


def main():
    write_lookup(FD02_MAP, "fd-0.2")
    write_lookup(FD03_MAP, "fd-0.3")
    print("\n[DONE] lookup CSVs written.")


if __name__ == "__main__":
    main()