import numpy as np
import pandas as pd
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").exists() and (p / "results").exists():
            return p
    raise FileNotFoundError("Could not find repo root.")


ROOT = find_repo_root(Path(__file__).resolve().parent)

META_DIR = ROOT / "data" / "metadata"
OUT_DIR = ROOT / "results" / "qc" / "missingness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "fd_0p2": META_DIR / "ABIDE12_phenotypes_combined_fd_0p2.csv",
    "fd_0p3": META_DIR / "ABIDE12_phenotypes_combined_fd_0p3.csv",
}


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SUB_ID"] = pd.to_numeric(df["SUB_ID"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")
    df["AGE_AT_SCAN"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")

    if "FIQ" in df.columns:
        df["FIQ"] = pd.to_numeric(df["FIQ"], errors="coerce")
    else:
        df["FIQ"] = np.nan

    if "RIGHT_HANDED" in df.columns:
        df["RIGHT_HANDED"] = pd.to_numeric(df["RIGHT_HANDED"], errors="coerce")
    else:
        df["RIGHT_HANDED"] = np.nan

    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()

    df = df.dropna(subset=["SUB_ID", "SEX", "DX_GROUP", "AGE_AT_SCAN"]).copy()
    df["SUB_ID"] = df["SUB_ID"].astype(int)
    df["SEX"] = df["SEX"].astype(int)
    df["DX_GROUP"] = df["DX_GROUP"].astype(int)

    df["sex_label"] = df["SEX"].map({1: "male", 2: "female"}).fillna("unknown")
    df["dx_label"] = df["DX_GROUP"].map({1: "ASD", 2: "CTL"}).fillna("unknown")

    if "AGE_GROUP" not in df.columns:
        bins = [0, 10, 13, 18, 200]
        labels = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
        df["AGE_GROUP"] = pd.cut(
            df["AGE_AT_SCAN"],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True,
        )
    else:
        df["AGE_GROUP"] = df["AGE_GROUP"].astype(str).str.strip()

    df["has_FIQ"] = df["FIQ"].notna()
    df["has_RIGHT_HANDED"] = df["RIGHT_HANDED"].notna()
    df["has_both"] = df["has_FIQ"] & df["has_RIGHT_HANDED"]

    return df


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_subjects=("SUB_ID", "nunique"),
            n_FIQ=("has_FIQ", "sum"),
            n_RIGHT_HANDED=("has_RIGHT_HANDED", "sum"),
            n_complete_both=("has_both", "sum"),
        )
        .reset_index()
    )

    g["pct_FIQ"] = 100 * g["n_FIQ"] / g["n_subjects"]
    g["pct_RIGHT_HANDED"] = 100 * g["n_RIGHT_HANDED"] / g["n_subjects"]
    g["pct_complete_both"] = 100 * g["n_complete_both"] / g["n_subjects"]

    return g


def main():
    for cutoff, fp in FILES.items():
        if not fp.exists():
            print(f"[SKIP] Missing file: {fp}")
            continue

        df = pd.read_csv(fp)
        df.columns = [str(c).strip() for c in df.columns]
        df = add_labels(df)

        print(f"\n=== {cutoff} ===")
        print(f"[INFO] subjects: {df['SUB_ID'].nunique()}")

        overall = pd.DataFrame([{
            "cutoff": cutoff,
            "n_subjects": df["SUB_ID"].nunique(),
            "n_FIQ": int(df["has_FIQ"].sum()),
            "pct_FIQ": 100 * df["has_FIQ"].mean(),
            "n_RIGHT_HANDED": int(df["has_RIGHT_HANDED"].sum()),
            "pct_RIGHT_HANDED": 100 * df["has_RIGHT_HANDED"].mean(),
            "n_complete_both": int(df["has_both"].sum()),
            "pct_complete_both": 100 * df["has_both"].mean(),
        }])

        by_group = summarize_group(df, ["sex_label", "dx_label", "AGE_GROUP"])
        by_site = summarize_group(df, ["SITE_ID"])

        overall_out = OUT_DIR / f"{cutoff}_overall_missingness.csv"
        group_out = OUT_DIR / f"{cutoff}_sex_dx_age_missingness.csv"
        site_out = OUT_DIR / f"{cutoff}_site_missingness.csv"

        overall.to_csv(overall_out, index=False)
        by_group.to_csv(group_out, index=False)
        by_site.to_csv(site_out, index=False)

        print(f"[SAVED] {overall_out}")
        print(f"[SAVED] {group_out}")
        print(f"[SAVED] {site_out}")

        print("\n[overall]")
        print(overall.to_string(index=False))

        print("\n[sex x dx x age]")
        print(by_group.to_string(index=False))


if __name__ == "__main__":
    main()