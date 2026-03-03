import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
IN_DIR  = ROOT / "results" / "hubs" / "module_stats_sitecov"
OUT_DIR = ROOT / "results" / "hubs" / "module_stats_sitecov_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "OVERALL_sexbalanced_fd-0.2",
    "OVERALL_sexbalanced_fd-0.3",
    "OVERALL_ageSexMatched_fd-0.2",
    "OVERALL_ageSexMatched_fd-0.3",
]
SEXES = ["female", "male"]
AGE_GROUPS = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]

METRICS = [
    "PC", "PC_pos", "PC_neg",
    "Z", "Z_pos", "Z_neg",
    "Strength_pos", "Strength_neg",
]

ROUND_COLS = {
    "mean_CTL": 3,
    "mean_ASD": 3,
    "beta_CTL_minus_ASD": 3,
    "p_DX": 4,
    "p_DX_FDR": 4,
    "p_SITE": 4,
    "r2": 3,
}

def load_stats(scenario, sex, age):
    fp = IN_DIR / f"{scenario}__{sex}__{age}__module_stats_sitecov.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip()
    df["metric"] = df["metric"].astype(str).str.strip()
    return df

def export_tables():
    for scenario in SCENARIOS:
        for sex in SEXES:
            for age in AGE_GROUPS:
                df = load_stats(scenario, sex, age)
                if df is None or df.empty:
                    continue

                for metric in METRICS:
                    sub = df[df["metric"] == metric].copy()
                    if sub.empty:
                        continue

                    for c, nd in ROUND_COLS.items():
                        if c in sub.columns:
                            sub[c] = pd.to_numeric(sub[c], errors="coerce").round(nd)

                    if "DX_FDR_significant" in sub.columns:
                        sub["sig_DX_FDR"] = sub["DX_FDR_significant"].map({True: "*", False: ""})
                    else:
                        sub["sig_DX_FDR"] = ""

                    cols = [c for c in [
                        "module","mean_CTL","mean_ASD","beta_CTL_minus_ASD",
                        "p_DX","p_DX_FDR","sig_DX_FDR",
                        "p_SITE","r2","n_ASD","n_CTL","note"
                    ] if c in sub.columns]

                    out = sub[cols].sort_values("module")
                    out_path = OUT_DIR / f"{scenario}__{sex}__{age}__{metric}__table.csv"
                    out.to_csv(out_path, index=False)
                    print(f"[SAVED] {out_path}")

    print("\n[DONE] tables exported.")

if __name__ == "__main__":
    export_tables()