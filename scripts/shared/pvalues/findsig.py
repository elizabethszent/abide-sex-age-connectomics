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

SEARCH_DIR = ROOT / "results" / "hubs_organized"
OUT_CSV = ROOT / "results" / "qc" / "significant_module_summary.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def load_csv(fp: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(fp)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"[WARN] Could not read {fp}: {e}")
        return None


def find_sig_column(df: pd.DataFrame) -> str | None:
    candidates = ["sig_DX_FDR", "DX_FDR_significant"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def mask_significant(df: pd.DataFrame, sig_col: str) -> pd.Series:
    if sig_col == "sig_DX_FDR":
        return df[sig_col].astype(str).str.strip().eq("*")
    if sig_col == "DX_FDR_significant":
        vals = df[sig_col]
        if vals.dtype == bool:
            return vals.fillna(False)
        return vals.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    return pd.Series([False] * len(df), index=df.index)


def summarize_file(fp: Path) -> list[dict]:
    df = load_csv(fp)
    if df is None or df.empty:
        return []

    if "module" not in df.columns:
        return []

    sig_col = find_sig_column(df)
    if sig_col is None:
        return []

    sig_mask = mask_significant(df, sig_col)
    sig_df = df[sig_mask].copy()

    if sig_df.empty:
        return []

    sig_df["module"] = pd.to_numeric(sig_df["module"], errors="coerce")
    sig_df = sig_df.dropna(subset=["module"]).copy()
    sig_df["module"] = sig_df["module"].astype(int)

    rows = []
    for _, row in sig_df.iterrows():
        rows.append(
            {
                "full_path": str(fp.resolve()),
                "file_name": fp.name,
                "module": int(row["module"]),
                "beta_CTL_minus_ASD": row["beta_CTL_minus_ASD"] if "beta_CTL_minus_ASD" in row else None,
                "p_DX": row["p_DX"] if "p_DX" in row else None,
                "p_DX_FDR": row["p_DX_FDR"] if "p_DX_FDR" in row else None,
                "sig_marker": row[sig_col],
            }
        )

    return rows


def main():
    if not SEARCH_DIR.exists():
        raise FileNotFoundError(f"Search dir not found: {SEARCH_DIR}")

    csv_files = sorted(SEARCH_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found under {SEARCH_DIR}")

    all_rows = []

    print(f"[INFO] scanning: {SEARCH_DIR}\n")

    for fp in csv_files:
        rows = summarize_file(fp)
        if not rows:
            continue

        for r in rows:
            print(f"FULL PATH : {r['full_path']}")
            print(f"FILE      : {r['file_name']}")
            print(f"MODULE    : {r['module']}")
            print(f"BETA      : {r['beta_CTL_minus_ASD']}")
            print(f"p_DX      : {r['p_DX']}")
            print(f"p_DX_FDR  : {r['p_DX_FDR']}")
            print("-" * 80)

        all_rows.extend(rows)

    if not all_rows:
        print("[INFO] No significant modules found.")
        return

    out_df = pd.DataFrame(all_rows)
    out_df = out_df.sort_values(["full_path", "module"]).reset_index(drop=True)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\n[SAVED] {OUT_CSV}")


if __name__ == "__main__":
    main()