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


def plot_sex(sex):
    # load tables
    child = load_pc_z(sex, "child")
    teen = load_pc_z(sex, "teen")

    # guess metric column names from CHILD df (should be same schema)
    pc_col = guess_column(child, "pc")
    z_col = guess_column(child, "z")
    print(f"[{sex}] Using PC column '{pc_col}', Z column '{z_col}'")

    # only modules that appear in both age groups
    common_modules = sorted(
        set(child["module"].unique()) & set(teen["module"].unique())
    )
    child = child[child["module"].isin(common_modules)]
    teen = teen[teen["module"].isin(common_modules)]

    # -------- PC --------
    child_pc = effects_by_module(child, pc_col)
    teen_pc = effects_by_module(teen, pc_col)

    x = np.arange(len(common_modules))
    labels = [f"M{m}" for m in common_modules]

    plt.figure(figsize=(8, 4))
    plt.axhline(0, color="black", linewidth=1)
    plt.plot(x, [child_pc[m] for m in common_modules],
             marker="o", linestyle="-", label="Child")
    plt.plot(x, [teen_pc[m] for m in common_modules],
             marker="s", linestyle="--", label="Teen")

    plt.xticks(x, labels)
    plt.ylabel("Cohen's d (CTL - ASD)")
    plt.title(f"{sex.capitalize()} – Participation Coefficient (PC)\nChild vs Teen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{sex}_child_vs_teen_PC.png", dpi=300)
    plt.close()

    # -------- within-module z --------
    child_z = effects_by_module(child, z_col)
    teen_z = effects_by_module(teen, z_col)

    plt.figure(figsize=(8, 4))
    plt.axhline(0, color="black", linewidth=1)
    plt.plot(x, [child_z[m] for m in common_modules],
             marker="o", linestyle="-", label="Child")
    plt.plot(x, [teen_z[m] for m in common_modules],
             marker="s", linestyle="--", label="Teen")

    plt.xticks(x, labels)
    plt.ylabel("Cohen's d (CTL - ASD)")
    plt.title(f"{sex.capitalize()} – Within-module degree z\nChild vs Teen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{sex}_child_vs_teen_Z.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    for sex in ["female", "male"]:
        print(f"Plotting {sex}…")
        plot_sex(sex)
    print("Done.")
