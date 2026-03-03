# scripts/shared/louvain/summarize_bestk7_modules_and_yeo.py

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
BASE = ROOT / r"results\louvain_bestK7_vs_yeo"

# Your 4 BESTK7 partitions (mix of .npy and .txt is fine)
INPUTS = [
    BASE / r"OVERALL_sexbalanced_fd-0.2"  / "OVERALL_sexbalanced_fd-0.2_BESTK7_modules.npy",
    BASE / r"OVERALL_sexbalanced_fd-0.3"  / "OVERALL_sexbalanced_fd-0.3_BESTK7_modules.npy",
    BASE / r"OVERALL_ageSexMatched_fd-0.2" / "OVERALL_ageSexMatched_fd-0.2_BESTK7_modules.npy",
    BASE / r"OVERALL_ageSexMatched_fd-0.3" / "OVERALL_ageSexMatched_fd-0.3_BESTK7_modules.txt",
]

EXPECTED_N = 200
EXPECTED_K = 7

# Column order we expect in the Yeo crosstab CSV you generated earlier
YEO_COLS = [
    "Background",
    "DefaultMode",
    "DorsalAttention",
    "Frontoparietal",
    "Limbic",
    "Somatomotor",
    "VentralAttention",
    "Visual",
]


def load_modules(path: Path) -> np.ndarray:
    """Load modules vector length 200 with labels 1..K from .npy or .txt."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npy":
        mods = np.load(path).astype(int)
        mods = np.asarray(mods).reshape(-1)
        return mods

    # txt: either a 200-line list, or "ROI_index Module" table
    txt = path.read_text(encoding="utf-8").strip().splitlines()
    if not txt:
        raise ValueError(f"Empty file: {path}")

    # if it looks like a table with ROI_index + Module
    if ("ROI_index" in txt[0]) and ("Module" in txt[0]):
        df = pd.read_csv(path, sep=r"\s+")
        df = df.sort_values("ROI_index")
        return df["Module"].to_numpy().astype(int)

    # else: assume one int per line
    vals = []
    for line in txt:
        line = line.strip()
        if not line:
            continue
        vals.append(int(line))
    return np.array(vals, dtype=int)


def module_sizes(mods: np.ndarray) -> dict:
    c = Counter(mods.tolist())
    return {k: c.get(k, 0) for k in range(1, EXPECTED_K + 1)}


def read_yeo_counts(next_to_modules: Path) -> pd.DataFrame | None:
    """
    Your BESTK7 script saved:
      BESTK7_module_x_yeo_counts.csv
    in the SAME folder as the modules file (one directory up from filename).
    """
    folder = next_to_modules.parent
    fp = folder / "BESTK7_module_x_yeo_counts.csv"
    if not fp.exists():
        return None

    df = pd.read_csv(fp)
    # normalize columns
    df.columns = [c.strip() for c in df.columns]

    # Sometimes pandas writes an unnamed first column if it was an index
    if df.columns[0].lower().startswith("unnamed"):
        df = df.rename(columns={df.columns[0]: "Louvain_module"})

    # If it came out as a pivot with module as index, ensure module column exists
    if "Louvain_module" not in df.columns:
        # try treat first col as module
        df = df.rename(columns={df.columns[0]: "Louvain_module"})

    # Ensure all expected Yeo cols exist (fill missing with 0)
    for c in YEO_COLS:
        if c not in df.columns:
            df[c] = 0

    df["Louvain_module"] = df["Louvain_module"].astype(int)
    df = df.sort_values("Louvain_module").reset_index(drop=True)
    return df[["Louvain_module"] + YEO_COLS]


def row_percent(df_counts: pd.DataFrame) -> pd.DataFrame:
    df = df_counts.copy()
    yeo = df[YEO_COLS].to_numpy(dtype=float)
    row_sum = yeo.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(row_sum > 0, (yeo / row_sum) * 100.0, 0.0)
    out = pd.DataFrame(pct, columns=[c + "_pct" for c in YEO_COLS])
    out.insert(0, "Louvain_module", df["Louvain_module"].values)
    return out


def unique_dominant_labels(df_counts: pd.DataFrame, allow_background=False) -> dict:
    """
    Assign EACH Louvain module a unique dominant Yeo label.
    Greedy: sort modules by their best-available max overlap, assign best unused label.
    If allow_background=False, Background is never chosen unless it's the only nonzero.
    """
    df = df_counts.copy()

    allowed = YEO_COLS.copy()
    if not allow_background and "Background" in allowed:
        allowed.remove("Background")

    # build preference lists
    prefs = {}
    for _, row in df.iterrows():
        m = int(row["Louvain_module"])
        scores = {c: int(row[c]) for c in YEO_COLS}
        # if background not allowed, push it to the bottom
        ordered = sorted(
            scores.items(),
            key=lambda kv: (kv[1], 0 if (kv[0] in allowed) else -1),
            reverse=True
        )
        prefs[m] = ordered

    # modules with highest top-score go first
    ordering = sorted(
        prefs.keys(),
        key=lambda m: max(v for _, v in prefs[m]),
        reverse=True
    )

    used = set()
    assign = {}
    for m in ordering:
        pick = None
        for label, score in prefs[m]:
            if score <= 0:
                continue
            if (not allow_background) and (label == "Background"):
                continue
            if label not in used:
                pick = label
                break
        # fallback: if nothing available, allow repeats
        if pick is None:
            pick = prefs[m][0][0]
        assign[m] = pick
        used.add(pick)

    return assign


def main():
    summary_rows = []

    for path in INPUTS:
        label = path.parent.name
        print("\n" + "=" * 70)
        print(f"{label}")
        print(f"Modules file: {path}")

        mods = load_modules(path)

        if mods.shape != (EXPECTED_N,):
            print(f"[WARN] Expected {EXPECTED_N} entries, got {mods.shape}. Continuing anyway.")

        k = len(set(mods.tolist()))
        if k != EXPECTED_K:
            print(f"[WARN] Expected K={EXPECTED_K}, got K={k} (labels={sorted(set(mods.tolist()))}).")

        sizes = module_sizes(mods)
        print("Module sizes (node counts):")
        for m in range(1, EXPECTED_K + 1):
            print(f"  module {m}: {sizes[m]}")

        # store for final table
        row = {"label": label}
        row.update({f"module_{m}_n": sizes[m] for m in range(1, EXPECTED_K + 1)})
        summary_rows.append(row)

        # If you have Yeo crosstab saved, print % + unique mapping
        df_counts = read_yeo_counts(path)
        if df_counts is None:
            print("[INFO] No BESTK7_module_x_yeo_counts.csv found next to this modules file.")
            continue

        print("\nYeo overlap counts (per module):")
        print(df_counts.to_string(index=False))

        df_pct = row_percent(df_counts)
        print("\nYeo overlap row-% (per module):")
        # prettier formatting
        df_pct_fmt = df_pct.copy()
        for c in df_pct_fmt.columns:
            if c.endswith("_pct"):
                df_pct_fmt[c] = df_pct_fmt[c].map(lambda x: f"{x:6.1f}")
        print(df_pct_fmt.to_string(index=False))

        uniq = unique_dominant_labels(df_counts, allow_background=False)
        print("\nUnique dominant Yeo label per module (no repeats, no Background):")
        for m in range(1, EXPECTED_K + 1):
            print(f"  module {m}: {uniq.get(m, 'NA')}")

    # One combined summary table
    if summary_rows:
        out = pd.DataFrame(summary_rows)
        out_path = BASE / "BESTK7_module_size_summary.csv"
        out.to_csv(out_path, index=False)
        print("\n" + "=" * 70)
        print(f"[SAVED] Combined module-size summary -> {out_path}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()