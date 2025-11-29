import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats 


ROOT = Path(r"C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")
HUB_DIR = ROOT / "results/hubs"
OUT_DIR = ROOT / "results/hubs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

female_path = HUB_DIR / "female_global_Q_by_age.csv"
male_path   = HUB_DIR / "male_global_Q_by_age.csv"


female = pd.read_csv(female_path)
male   = pd.read_csv(male_path)

female["sex"] = "female"
male["sex"]   = "male"

df = pd.concat([female, male], ignore_index=True)

#Make AGE_GROUP ordered
age_order = ["child", "teen", "young_adult"]
df["AGE_GROUP"] = pd.Categorical(df["AGE_GROUP"], categories=age_order, ordered=True)

#Map DX_GROUP to labels
dx_map = {1: "ASD", 2: "CTL"}
df["DX_LABEL"] = df["DX_GROUP"].map(dx_map)

print("Loaded global Q data:")
print(df.groupby(["sex", "AGE_GROUP", "DX_LABEL"])["Q"].describe()[["mean", "std", "count"]])

def plot_boxplots_for_sex(df_sex: pd.DataFrame, sex_label: str, out_path: Path):

    fig, ax = plt.subplots(figsize=(7, 4))

    #positions: for each age, two boxes (ASD, CTL)
    age_groups = age_order
    n_ages = len(age_groups)
    width = 0.35

    positions_asd = []
    positions_ctl = []
    data_asd = []
    data_ctl = []
    xtick_positions = []

    for i, age in enumerate(age_groups):
        sub = df_sex[df_sex["AGE_GROUP"] == age]

        q_asd = sub[sub["DX_LABEL"] == "ASD"]["Q"].to_numpy()
        q_ctl = sub[sub["DX_LABEL"] == "CTL"]["Q"].to_numpy()

        if len(q_asd) == 0 and len(q_ctl) == 0:
            continue

        center = i * 2.0 #spacing between age groups
        p_asd = center - width / 2
        p_ctl = center + width / 2

        positions_asd.append(p_asd)
        positions_ctl.append(p_ctl)
        data_asd.append(q_asd)
        data_ctl.append(q_ctl)
        xtick_positions.append(center)

    #ASD boxplots
    bp_asd = ax.boxplot(
        data_asd,
        positions=positions_asd,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    for patch in bp_asd["boxes"]:
        patch.set_facecolor("#d95f02") #orange-ish

    #CTL boxplots
    bp_ctl = ax.boxplot(
        data_ctl,
        positions=positions_ctl,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    for patch in bp_ctl["boxes"]:
        patch.set_facecolor("#1b9e77")  # green-ish

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(age_groups)
    ax.set_ylabel("Modularity Q")
    ax.set_title(f"{sex_label.capitalize()} – Global modularity Q\nASD vs Control by age group")

    #legend (top-left corner, away from boxes)
    asd_handle = bp_asd["boxes"][0]
    ctl_handle = bp_ctl["boxes"][0]
    ax.legend(
        [asd_handle, ctl_handle],
        ["ASD", "Control"],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mean_sem_for_sex(df_sex: pd.DataFrame, sex_label: str, out_path: Path):

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(age_order))

    for dx_label, color, marker in [("ASD", "#d95f02", "o"), ("CTL", "#1b9e77", "s")]:
        means = []
        sems = []
        for age in age_order:
            sub = df_sex[(df_sex["AGE_GROUP"] == age) & (df_sex["DX_LABEL"] == dx_label)]
            q_vals = sub["Q"].to_numpy()
            if len(q_vals) == 0:
                means.append(np.nan)
                sems.append(np.nan)
            else:
                means.append(np.mean(q_vals))
                sems.append(stats.sem(q_vals))
        means = np.array(means, dtype=float)
        sems = np.array(sems, dtype=float)

        ax.errorbar(
            x,
            means,
            yerr=sems,
            label=dx_label,
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(age_order)
    ax.set_ylabel("Modularity Q")
    ax.set_title(f"{sex_label.capitalize()} – Global modularity Q\nMean ± SEM across age groups")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")



if __name__ == "__main__":
    #split by sex
    df_female = df[df["sex"] == "female"].copy()
    df_male   = df[df["sex"] == "male"].copy()

    #boxplots
    plot_boxplots_for_sex(
        df_female,
        "female",
        OUT_DIR / "female_Q_ASD_vs_CTL_by_age_boxplot.png",
    )
    plot_boxplots_for_sex(
        df_male,
        "male",
        OUT_DIR / "male_Q_ASD_vs_CTL_by_age_boxplot.png",
    )

    #mean±SEM line plots
    plot_mean_sem_for_sex(
        df_female,
        "female",
        OUT_DIR / "female_Q_ASD_vs_CTL_by_age_meanSEM.png",
    )
    plot_mean_sem_for_sex(
        df_male,
        "male",
        OUT_DIR / "male_Q_ASD_vs_CTL_by_age_meanSEM.png",
    )

    print("Done.")
