import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
HUB_DIR = ROOT / "results" / "hubs"

sexes = ["female", "male"]
age_groups = ["child", "teen", "young_adult"]
metrics = ["PC", "Z"]

def load_stats(sex, age_group):
    #fname = f"{sex}_{age_group}_pc_z_module_stats.csv"
    fname = f"{sex}_{age_group}_pc_z_module_stats_revised.csv"
    path = HUB_DIR / fname
    print(f"Loading {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    return df

def make_stats_table(df, metric, sex, age_group):
    """
    Make a nice table for one metric (PC or Z)
    and save it as a separate CSV you can drop into Excel/LaTeX.
    """
    sub = df[df["metric"] == metric].copy()

  
    for col in [
        "mean_CTL",
        "mean_ASD",
        "beta_CTL_minus_ASD",
        "p_uncorrected",
        "p_FDR",
    ]:
        sub[col] = sub[col].round(3)

    sub["sig"] = sub["FDR_significant"].map({True: "*", False: ""})

 
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
    sub.to_csv(HUB_DIR / out_name, index=False)
    print(f"Saved table to {HUB_DIR / out_name}")

for sex in sexes:
    for age in age_groups:
        df = load_stats(sex, age)
        for metric in metrics:
            make_stats_table(df, metric, sex, age)
