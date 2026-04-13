import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf

# paths
PROJECT_ROOT = "/work/ioannou_lab/elizabeth.szentmiklo/abide-sex-age-connectomics"
CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "ABIDE12_phenotypes_combined_fd_0p3.csv"
)

OUTDIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "q2_iq_dx_by_sex_age_fd0p3"
)
os.makedirs(OUTDIR, exist_ok=True)

# plot style
sns.set_theme(style="whitegrid")

# label maps / ordering
DX_MAP = {
    1: "ASD",
    2: "Control"
}

AGE_ORDER = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
AGE_LABELS = {
    "child_0_9": "Child (0–9)",
    "preteen_10_12": "Preteen (10–12)",
    "teen_13_17": "Teen (13–17)",
    "adult_18_plus": "Adult (18+)"
}

SEX_ORDER = ["Male", "Female"]
DX_ORDER = ["ASD", "Control"]

# load data
df = pd.read_csv(CSV_PATH)

# prefer the string sex column if present
if "sex" in df.columns:
    sex_series = df["sex"].astype(str).str.strip().str.lower()
    sex_series = sex_series.replace({
        "m": "male",
        "f": "female"
    })
else:
    # fallback if dataset only has numeric sex coding
    sex_series = df["SEX"].map({
        1: "male",
        2: "female"
    })

df["diagnosis"] = df["DX_GROUP"].map(DX_MAP)
df["sex_label"] = sex_series.str.title()
df["age_group"] = pd.Categorical(df["AGE_GROUP"], categories=AGE_ORDER, ordered=True)
df["age_label"] = df["age_group"].map(AGE_LABELS)

# keep rows needed for Q2
q2 = df.dropna(subset=["FIQ", "diagnosis", "sex_label", "age_group", "SITE_ID"]).copy()
q2["FIQ"] = pd.to_numeric(q2["FIQ"], errors="coerce")
q2 = q2.dropna(subset=["FIQ"])

print(f"Loaded phenotype file: {CSV_PATH}")
print(f"Rows with usable FIQ for Q2: {len(q2)}")

# helper
def mean_ci95(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = s.mean()
    if n == 1:
        return mean, mean, mean
    se = s.std(ddof=1) / np.sqrt(n)
    ci = 1.96 * se
    return mean, mean - ci, mean + ci

# figure 4.1
# FIQ-available sample sizes by age, sex, and diagnosis
count_df = (
    q2.groupby(["age_label", "sex_label", "diagnosis"], observed=False)
      .size()
      .reset_index(name="n")
)

count_df["group"] = count_df["sex_label"] + " | " + count_df["diagnosis"]
count_pivot = count_df.pivot(index="age_label", columns="group", values="n").fillna(0)

ordered_age_labels = [AGE_LABELS[a] for a in AGE_ORDER]
count_pivot = count_pivot.reindex(ordered_age_labels)

plt.figure(figsize=(10, 5))
sns.heatmap(count_pivot, annot=True, fmt=".0f", cmap="Blues")
plt.title("Figure 4.1: FIQ-Available Sample Size by Age, Sex, and Diagnosis")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure_4_1_fiq_counts_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()

# figure 4.2
# FIQ distributions by diagnosis within each age group, split by sex
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

for ax, age in zip(axes.ravel(), AGE_ORDER):
    sub = q2[q2["age_group"] == age].copy()
    sns.boxplot(
        data=sub,
        x="diagnosis",
        y="FIQ",
        hue="sex_label",
        order=DX_ORDER,
        hue_order=SEX_ORDER,
        ax=ax
    )
    ax.set_title(AGE_LABELS[age])
    ax.set_xlabel("")
    ax.set_ylabel("Full-Scale IQ")

for ax in axes.ravel()[1:]:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

handles, labels = axes.ravel()[0].get_legend_handles_labels()
axes.ravel()[0].legend(handles, labels, title="Sex", loc="best")

fig.suptitle("Figure 4.2: FIQ Distribution by Diagnosis Within Each Age Group", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure_4_2_fiq_boxplots_by_age.png"), dpi=300, bbox_inches="tight")
plt.close()

# figure 4.3
# mean FIQ across age groups by diagnosis, split by sex
summary_rows = []

for sex in SEX_ORDER:
    for age in AGE_ORDER:
        for dx in DX_ORDER:
            sub = q2[
                (q2["sex_label"] == sex) &
                (q2["age_group"] == age) &
                (q2["diagnosis"] == dx)
            ]["FIQ"]

            mean, low, high = mean_ci95(sub)

            summary_rows.append({
                "sex_label": sex,
                "age_group": age,
                "age_label": AGE_LABELS[age],
                "diagnosis": dx,
                "mean_fiq": mean,
                "ci_low": low,
                "ci_high": high,
                "n": sub.notna().sum()
            })

summary_df = pd.DataFrame(summary_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, sex in zip(axes, SEX_ORDER):
    sub = summary_df[summary_df["sex_label"] == sex].copy()

    for dx, marker in zip(DX_ORDER, ["o", "s"]):
        dx_sub = sub[sub["diagnosis"] == dx].copy()
        dx_sub = dx_sub.set_index("age_group").reindex(AGE_ORDER).reset_index()

        x = np.arange(len(dx_sub))
        y = dx_sub["mean_fiq"].values
        yerr = [
            y - dx_sub["ci_low"].values,
            dx_sub["ci_high"].values - y
        ]

        ax.errorbar(
            x=x,
            y=y,
            yerr=yerr,
            marker=marker,
            linewidth=2,
            capsize=4,
            label=dx
        )

    ax.set_xticks(np.arange(len(ordered_age_labels)))
    ax.set_xticklabels(ordered_age_labels, rotation=20)
    ax.set_title(sex)
    ax.set_xlabel("")
    ax.set_ylabel("Mean Full-Scale IQ")
    ax.legend(title="Diagnosis")

fig.suptitle("Figure 4.3: Mean FIQ Across Age Groups by Diagnosis, Split by Sex", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "figure_4_3_mean_fiq_by_age_sex_dx.png"), dpi=300, bbox_inches="tight")
plt.close()

# figure 4.4
# control - ASD difference within each age-sex subgroup
diff_rows = []

for age in AGE_ORDER:
    for sex in SEX_ORDER:
        sub = q2[
            (q2["age_group"] == age) &
            (q2["sex_label"] == sex)
        ].copy()

        asd = sub.loc[sub["diagnosis"] == "ASD", "FIQ"].dropna()
        ctl = sub.loc[sub["diagnosis"] == "Control", "FIQ"].dropna()

        if len(asd) == 0 or len(ctl) == 0:
            continue

        diff = ctl.mean() - asd.mean()

        if len(asd) > 1 and len(ctl) > 1:
            se = np.sqrt(asd.var(ddof=1) / len(asd) + ctl.var(ddof=1) / len(ctl))
            ci = 1.96 * se
            lower = diff - ci
            upper = diff + ci
        else:
            lower = diff
            upper = diff

        diff_rows.append({
            "age_group": age,
            "age_label": AGE_LABELS[age],
            "sex_label": sex,
            "diff_control_minus_asd": diff,
            "ci_low": lower,
            "ci_high": upper
        })

diff_df = pd.DataFrame(diff_rows)

if not diff_df.empty:
    diff_df = diff_df.sort_values(["age_group", "sex_label"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ypos = np.arange(len(diff_df))
    ax.errorbar(
        diff_df["diff_control_minus_asd"],
        ypos,
        xerr=[
            diff_df["diff_control_minus_asd"] - diff_df["ci_low"],
            diff_df["ci_high"] - diff_df["diff_control_minus_asd"]
        ],
        fmt="o",
        capsize=4
    )

    ax.axvline(0, linestyle="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(diff_df["age_label"] + " | " + diff_df["sex_label"])
    ax.set_xlabel("Mean FIQ Difference (Control - ASD)")
    ax.set_title("Figure 4.4: Control-ASD FIQ Difference Within Each Age-Sex Subgroup")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figure_4_4_control_minus_asd_diff.png"), dpi=300, bbox_inches="tight")
    plt.close()

# site-adjusted ANOVA
model = smf.ols(
    "FIQ ~ C(diagnosis) * C(sex_label) * C(age_group) + C(SITE_ID)",
    data=q2
).fit()

anova_table = sm.stats.anova_lm(model, typ=2)

print("\n" + "=" * 80)
print("SITE-ADJUSTED TYPE II ANOVA")
print("=" * 80)
print(anova_table)

# welch t-tests within each age-sex subgroup
test_rows = []

for age in AGE_ORDER:
    for sex in SEX_ORDER:
        sub = q2[
            (q2["age_group"] == age) &
            (q2["sex_label"] == sex)
        ].copy()

        asd = sub.loc[sub["diagnosis"] == "ASD", "FIQ"].dropna()
        ctl = sub.loc[sub["diagnosis"] == "Control", "FIQ"].dropna()

        if len(asd) == 0 or len(ctl) == 0:
            continue

        t_stat, p_val = ttest_ind(ctl, asd, equal_var=False, nan_policy="omit")

        test_rows.append({
            "age_group": AGE_LABELS[age],
            "sex": sex,
            "n_asd": len(asd),
            "n_control": len(ctl),
            "mean_asd": asd.mean(),
            "mean_control": ctl.mean(),
            "control_minus_asd": ctl.mean() - asd.mean(),
            "welch_t": t_stat,
            "p_value": p_val
        })

test_df = pd.DataFrame(test_rows)

print("\n" + "=" * 80)
print("SUBGROUP WELCH T-TESTS (CONTROL VS ASD WITHIN EACH AGE-SEX GROUP)")
print("=" * 80)
print(test_df.to_string(index=False))

# save tables
summary_df.to_csv(os.path.join(OUTDIR, "q2_summary_means_and_ci.csv"), index=False)
diff_df.to_csv(os.path.join(OUTDIR, "q2_control_minus_asd_differences.csv"), index=False)
test_df.to_csv(os.path.join(OUTDIR, "q2_subgroup_welch_tests.csv"), index=False)
anova_table.to_csv(os.path.join(OUTDIR, "q2_site_adjusted_anova.csv"))

print(f"\nSaved all Q2 outputs to: {OUTDIR}")