from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")
HUBS_DIR = ROOT / "results" / "hubs"

sexes = ["female", "male"]
ages = ["child", "teen", "young_adult"]

# (column_name, pretty_label)
metrics = [("PC", "PC"), ("z", "Z")]

for sex in sexes:
    for age in ages:
        csv_path = HUBS_DIR / f"{sex}_{age}_pc_z_revised.csv"

        if not csv_path.exists():
            print(f"[WARN] Missing file: {csv_path}")
            continue

        print(f"\nLoading {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        print("  Columns:", list(df.columns))

        # --- which column has diagnosis? ---
        group_col = "DX_GROUP" if "DX_GROUP" in df.columns else "dx_group"
        if group_col not in df.columns:
            print(f"  [SKIP] No DX_GROUP/dx_group column in {csv_path}")
            continue

        print("  Unique DX_GROUP values:", df[group_col].unique(), 
              "dtype:", df[group_col].dtype)

        # Map numeric codes and any string labels to ASD / CTL
        group_map = {
            1: "CTL",
            2: "ASD",
            "1": "CTL",
            "2": "ASD",
            "Control": "CTL",
            "control": "CTL",
            "CTL": "CTL",
            "ASD": "ASD",
            "asd": "ASD",
        }

        valid_vals = list(group_map.keys())
        sub = df[df[group_col].isin(valid_vals)].copy()
        if sub.empty:
            print(f"  [SKIP] No ASD/Control rows in {csv_path}")
            continue

        sub["group"] = sub[group_col].map(group_map)

        if "module" not in sub.columns:
            print(f"  [SKIP] No 'module' column in {csv_path}")
            continue

        for col_name, label in metrics:
            if col_name not in sub.columns:
                print(f"  [SKIP] {col_name} column missing in {csv_path}")
                continue

            print(f"  Making {label} boxplot for {sex}, {age}")

            plt.figure(figsize=(8, 4))
            sns.boxplot(
                data=sub,
                x="module",
                y=col_name,
                hue="group",
                showfliers=False,
            )

            # Show classical hub threshold but don't clamp at 0.5
            plt.axhline(0.5, linestyle="--", linewidth=1, color="grey")

            plt.title(
                f"{sex.capitalize()} {age.replace('_',' ')}: {label} by module (ASD vs CTL)"
            )
            plt.xlabel("Module")
            plt.ylabel(label)
            plt.tight_layout()

            out_path = HUBS_DIR / f"{sex}_{age}_{label}_boxplot.png"
            print(f"  Saving {out_path}")
            plt.savefig(out_path, dpi=300)
            plt.close()

