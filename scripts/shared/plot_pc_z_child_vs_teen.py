import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")
HUB_DIR = ROOT / "results/hubs"
OUT_DIR = ROOT / "results/hubs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------

def cohen_d(asd, ctl):
    """Cohen's d (CTL - ASD)."""
    asd = np.asarray(asd, dtype=float)
    ctl = np.asarray(ctl, dtype=float)
    asd = asd[~np.isnan(asd)]
    ctl = ctl[~np.isnan(ctl)]
    n1, n2 = len(asd), len(ctl)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = asd.mean(), ctl.mean()
    s1, s2 = asd.std(ddof=1), ctl.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0 or np.isnan(sp):
        return np.nan
    return (m2 - m1) / sp   # CTL - ASD


def guess_column(df, kind: str) -> str:
    """
    Try to find the column name for 'pc' or 'z' in df.
    kind: 'pc' or 'z'
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # 1) obvious exact matches
    if kind == "pc":
        for key in ["pc", "pc_mean", "participation", "participation_coeff"]:
            if key in cols_lower:
                return cols_lower[key]
    else:  # kind == "z"
        for key in ["z", "z_mean", "within_z", "within_module_z"]:
            if key in cols_lower:
                return cols_lower[key]

    # 2) fallback: any column containing substring
    if kind == "pc":
        candidates = [c for c in df.columns if "pc" in c.lower()
                      or "particip" in c.lower()]
    else:
        candidates = [c for c in df.columns
                      if c.lower().startswith("z") and "score" not in c.lower()]

    if not candidates:
        raise ValueError(
            f"Could not guess {kind} column. Available columns:\n{df.columns}"
        )
    # if multiple, just take the first
    return candidates[0]


def effects_by_module(df, metric_col: str) -> dict:
    """
    df: pc_z table for one sex/age
    metric_col: name of the column with PC or z
    returns: dict {module_id -> d}
    """
    out = {}
    for m in sorted(df["module"].unique()):
        sub = df[df["module"] == m]
        asd = sub[sub["DX_GROUP"] == 1][metric_col]
        ctl = sub[sub["DX_GROUP"] == 2][metric_col]
        if len(asd) < 2 or len(ctl) < 2:
            out[m] = np.nan
        else:
            out[m] = cohen_d(asd, ctl)
    return out


def load_pc_z(sex, age):
    path = HUB_DIR / f"{sex}_{age}_pc_z.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # make sure module & DX_GROUP exist
    needed = {"module", "DX_GROUP"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def nice_age_label(age_key: str) -> str:
    """Map internal age keys to pretty labels."""
    mapping = {
        "child": "Child",
        "teen": "Teen",
        "young_adult": "Young Adult",
    }
    return mapping.get(age_key, age_key)


def plot_sex(sex):
    # Try to load each age group
    age_dfs = {}
    for age in ["child", "teen", "young_adult"]:
        try:
            df = load_pc_z(sex, age)
            age_dfs[age] = df
        except FileNotFoundError:
            print(f"[{sex}] No file for age group '{age}', skipping that age.")
        except ValueError as e:
            print(f"[{sex}] Problem with {age} file: {e}, skipping that age.")

    if len(age_dfs) < 2:
        print(f"[{sex}] Not enough age groups to plot (need ≥ 2).")
        return

    # Use CHILD (if present) or the first df we have to guess PC/Z columns
    sample_df = age_dfs.get("child", next(iter(age_dfs.values())))
    pc_col = guess_column(sample_df, "pc")
    z_col = guess_column(sample_df, "z")
    print(f"[{sex}] Using PC column '{pc_col}', Z column '{z_col}'")

    # Only modules that appear in *all* available age groups
    common_modules = None
    for df in age_dfs.values():
        mods = set(df["module"].unique())
        if common_modules is None:
            common_modules = mods
        else:
            common_modules = common_modules & mods

    common_modules = sorted(common_modules)
    if not common_modules:
        print(f"[{sex}] No common modules across age groups, skipping.")
        return

    # Restrict each df to the common modules
    for age in age_dfs:
        age_dfs[age] = age_dfs[age][age_dfs[age]["module"].isin(common_modules)]

    # Compute Cohen's d per age × metric
    age_pc = {}
    age_z = {}
    for age, df_age in age_dfs.items():
        age_pc[age] = effects_by_module(df_age, pc_col)
        age_z[age] = effects_by_module(df_age, z_col)

    x = np.arange(len(common_modules))
    labels = [f"M{m}" for m in common_modules]

    # Styling (cyclical if more than 3 ages, but we only have 3)
    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", ":", "-."]

    # Sorted so order is child, teen, young_adult if present
    age_keys = sorted(age_dfs.keys(), key=lambda a: ["child", "teen", "young_adult"].index(a))

    # -------- PC plot --------
    plt.figure(figsize=(8, 4))
    plt.axhline(0, color="black", linewidth=1)

    for i, age in enumerate(age_keys):
        plt.plot(
            x,
            [age_pc[age][m] for m in common_modules],
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            label=nice_age_label(age),
        )

    plt.xticks(x, labels)
    plt.ylabel("Cohen's d (CTL - ASD)")
    pretty_ages = ", ".join(nice_age_label(a) for a in age_keys)
    plt.title(f"{sex.capitalize()} – Participation Coefficient (PC)\n{pretty_ages}")
    plt.legend()
    plt.tight_layout()

    out_name_pc = f"{sex}_{'_'.join(age_keys)}_PC.png"
    plt.savefig(OUT_DIR / out_name_pc, dpi=300)
    plt.close()

    # -------- Z plot --------
    plt.figure(figsize=(8, 4))
    plt.axhline(0, color="black", linewidth=1)

    for i, age in enumerate(age_keys):
        plt.plot(
            x,
            [age_z[age][m] for m in common_modules],
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            label=nice_age_label(age),
        )

    plt.xticks(x, labels)
    plt.ylabel("Cohen's d (CTL - ASD)")
    plt.title(f"{sex.capitalize()} – Within-module degree z\n{pretty_ages}")
    plt.legend()
    plt.tight_layout()

    out_name_z = f"{sex}_{'_'.join(age_keys)}_Z.png"
    plt.savefig(OUT_DIR / out_name_z, dpi=300)
    plt.close()


if __name__ == "__main__":
    for sex in ["female", "male"]:
        print(f"Plotting {sex}…")
        plot_sex(sex)
    print("Done.")
