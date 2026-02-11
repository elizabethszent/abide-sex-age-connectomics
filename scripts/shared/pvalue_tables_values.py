import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
HUB_DIR = ROOT / "results" / "hubs"

sexes = ["female", "male"]
age_groups = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]

# NOW includes Strength (must match exactly what's in your stats_df["metric"])
metrics = ["PC", "Z", "Strength"]


def load_stats(sex, age_group):
    # updated filename to match the new stats script output
    fname = f"{sex}_{age_group}_pc_z_strength_module_stats_revised.csv"
    path = HUB_DIR / fname
    print(f"Loading {path}")

    if not path.exists():
        print(f"[SKIP] Missing stats file: {path}")
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def make_stats_table(df, metric, sex, age_group):
    """
    Make a nice table for one metric (PC, Z, Strength)
    and save it as a separate CSV you can drop into Excel/LaTeX.
    """
    sub = df[df["metric"] == metric].copy()

    if sub.empty:
        print(f"[SKIP] No rows found for metric={metric} in {sex} {age_group}")
        return

    # round columns
    for col in ["mean_CTL", "mean_ASD", "beta_CTL_minus_ASD", "p_uncorrected", "p_FDR"]:
        if col in sub.columns:
            sub[col] = sub[col].round(3)

    # significance marker
    sub["sig"] = sub["FDR_significant"].map({True: "*", False: ""})

    # keep + order columns
    sub = sub[
        [
            "module",
            "mean_CTL",
            "mean_ASD",
            "beta_CTL_minus_ASD",
            "p_uncorrected",
            "p_FDR",
            "sig",
        ]
    ].sort_values("module")

    print(f"\n=== {sex}, {age_group}, {metric} ===")
    print(sub.to_string(index=False))

    out_name = f"{sex}_{age_group}_{metric}_table.csv"
    out_path = HUB_DIR / out_name
    sub.to_csv(out_path, index=False)
    print(f"Saved table to {out_path}")


for sex in sexes:
    for age in age_groups:
        df = load_stats(sex, age)
        if df is None:
            continue
        for metric in metrics:
            make_stats_table(df, metric, sex, age)
