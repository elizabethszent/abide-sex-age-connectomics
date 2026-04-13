import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    "q3_missingness_phenotypic_fd0p3_corrected"
)
os.makedirs(OUTDIR, exist_ok=True)

# plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

# load data
df = pd.read_csv(CSV_PATH)

# diagnosis mapping
if "DX_GROUP" in df.columns:
    df["diagnosis"] = (
        df["DX_GROUP"]
        .replace({1: "ASD", 2: "Control", "1": "ASD", "2": "Control"})
    )
else:
    raise ValueError("DX_GROUP column not found in phenotype file.")

# sex mapping
if "sex" in df.columns:
    sex_series = df["sex"].astype(str).str.strip().str.lower()
    sex_series = sex_series.replace({"m": "male", "f": "female"})
    df["sex_label"] = sex_series.str.title()
elif "SEX" in df.columns:
    df["sex_label"] = df["SEX"].replace({
        1: "Male",
        2: "Female",
        "1": "Male",
        "2": "Female"
    })
else:
    raise ValueError("No sex column found (expected 'sex' or 'SEX').")

# age mapping
AGE_ORDER = ["child_0_9", "preteen_10_12", "teen_13_17", "adult_18_plus"]
AGE_LABELS = {
    "child_0_9": "Child (0–9)",
    "preteen_10_12": "Preteen (10–12)",
    "teen_13_17": "Teen (13–17)",
    "adult_18_plus": "Adult (18+)"
}

if "AGE_GROUP" not in df.columns:
    raise ValueError("AGE_GROUP column not found in phenotype file.")

df["age_group"] = pd.Categorical(df["AGE_GROUP"], categories=AGE_ORDER, ordered=True)
df["age_label"] = df["age_group"].map(AGE_LABELS)

SEX_ORDER = ["Male", "Female"]
DX_ORDER = ["ASD", "Control"]
AGE_LABEL_ORDER = [AGE_LABELS[a] for a in AGE_ORDER]

# key phenotypic fields
candidate_fields = [
    "FIQ",
    "RIGHT_HANDED",
    "ADOS_MODULE",
    "ADOS_TOTAL",
    "ADOS_COMM",
    "ADOS_SOCIAL",
    "ADOS_STEREO_BEHAV",
    "ADOS_RSRCH_RELIABLE",
    "ADOS_GOTHAM_SOCAFFECT",
    "ADOS_GOTHAM_RRB",
    "ADOS_GOTHAM_TOTAL",
    "ADOS_GOTHAM_SEVERITY",
]

field_labels = {
    "FIQ": "FIQ",
    "RIGHT_HANDED": "Right-handed",
    "ADOS_MODULE": "ADOS module",
    "ADOS_TOTAL": "ADOS total",
    "ADOS_COMM": "ADOS communication",
    "ADOS_SOCIAL": "ADOS social",
    "ADOS_STEREO_BEHAV": "ADOS stereotyped behaviour",
    "ADOS_RSRCH_RELIABLE": "ADOS research reliable",
    "ADOS_GOTHAM_SOCAFFECT": "Gotham social affect",
    "ADOS_GOTHAM_RRB": "Gotham RRB",
    "ADOS_GOTHAM_TOTAL": "Gotham total",
    "ADOS_GOTHAM_SEVERITY": "Gotham severity",
}

key_fields = [
    c for c in candidate_fields
    if c in df.columns and df[c].notna().any()
]

print(f"Loaded phenotype file: {CSV_PATH}")
print(f"Total rows in file: {len(df)}")
print(f"Fields used for Q3: {key_fields}")

# helpers
def complete_case_mask(dataframe, required_fields):
    if len(required_fields) == 0:
        return pd.Series(True, index=dataframe.index)
    return dataframe[required_fields].notna().all(axis=1)

def retention_summary(dataframe, group_cols, requirement_sets):
    rows = []

    grouped = dataframe.groupby(group_cols, dropna=False, observed=False)
    for group_key, subdf in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        base_n = len(subdf)

        for req_name, req_fields in requirement_sets.items():
            mask = complete_case_mask(subdf, req_fields)
            n_retained = int(mask.sum())
            pct_retained = 100 * n_retained / base_n if base_n > 0 else np.nan

            row = {col: val for col, val in zip(group_cols, group_key)}
            row.update({
                "requirement_set": req_name,
                "base_n": base_n,
                "n_retained": n_retained,
                "pct_retained": pct_retained,
            })
            rows.append(row)

    return pd.DataFrame(rows)

# requirement sets
general_requirement_sets = {
    "Total cohort": [],
    "FIQ only": ["FIQ"],
    "FIQ + handedness": ["FIQ", "RIGHT_HANDED"],
}

asd_requirement_sets = {
    "ASD cohort": [],
    "FIQ only": ["FIQ"],
    "FIQ + handedness": ["FIQ", "RIGHT_HANDED"],
    "FIQ + ADOS total": ["FIQ", "ADOS_TOTAL"],
    "FIQ + classic ADOS battery": [
        "FIQ",
        "ADOS_MODULE",
        "ADOS_TOTAL",
        "ADOS_COMM",
        "ADOS_SOCIAL",
        "ADOS_STEREO_BEHAV",
    ],
    "FIQ + Gotham battery": [
        "FIQ",
        "ADOS_GOTHAM_SOCAFFECT",
        "ADOS_GOTHAM_RRB",
        "ADOS_GOTHAM_TOTAL",
        "ADOS_GOTHAM_SEVERITY",
    ],
    "All selected phenotypic fields": key_fields,
}

general_requirement_sets = {
    name: fields
    for name, fields in general_requirement_sets.items()
    if all(f in df.columns for f in fields)
}

asd_requirement_sets = {
    name: fields
    for name, fields in asd_requirement_sets.items()
    if all(f in df.columns for f in fields)
}

asd_df = df[df["diagnosis"] == "ASD"].copy()

# figure 5.1
general_fields = [f for f in ["FIQ", "RIGHT_HANDED"] if f in key_fields]

clinical_asd_fields = [
    f for f in [
        "ADOS_MODULE",
        "ADOS_TOTAL",
        "ADOS_COMM",
        "ADOS_SOCIAL",
        "ADOS_STEREO_BEHAV",
        "ADOS_RSRCH_RELIABLE",
        "ADOS_GOTHAM_SOCAFFECT",
        "ADOS_GOTHAM_RRB",
        "ADOS_GOTHAM_TOTAL",
        "ADOS_GOTHAM_SEVERITY",
    ]
    if f in key_fields
]

general_missing_rows = []
for field in general_fields:
    for dx in DX_ORDER:
        sub = df[df["diagnosis"] == dx]
        general_missing_rows.append({
            "field": field,
            "field_label": field_labels[field],
            "diagnosis": dx,
            "pct_missing": 100 * sub[field].isna().mean(),
        })

general_missing_df = pd.DataFrame(general_missing_rows)

general_pivot = general_missing_df.pivot(
    index="field_label",
    columns="diagnosis",
    values="pct_missing"
).reindex(
    index=[field_labels[f] for f in general_fields],
    columns=DX_ORDER
)

asd_clinical_rows = []
for field in clinical_asd_fields:
    asd_clinical_rows.append({
        "field": field,
        "field_label": field_labels[field],
        "pct_missing": 100 * asd_df[field].isna().mean(),
    })

asd_clinical_df = pd.DataFrame(asd_clinical_rows).sort_values("pct_missing", ascending=False)

fig, axes = plt.subplots(
    1, 2, figsize=(13.5, 7),
    gridspec_kw={"width_ratios": [1, 1.7]}
)

sns.heatmap(
    general_pivot,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    vmin=0,
    vmax=100,
    ax=axes[0]
)
axes[0].set_title("General fields by diagnosis", fontsize=13, pad=12)
axes[0].set_xlabel("")
axes[0].set_ylabel("")

sns.barplot(
    data=asd_clinical_df,
    x="pct_missing",
    y="field_label",
    ax=axes[1]
)
axes[1].set_title("ASD-only clinical fields", fontsize=13, pad=12)
axes[1].set_xlabel("Percent Missing")
axes[1].set_ylabel("")

for i, row in enumerate(asd_clinical_df.itertuples(index=False)):
    axes[1].text(
        row.pct_missing + 1,
        i,
        f"{row.pct_missing:.1f}%",
        va="center",
        fontsize=10
    )

axes[1].set_xlim(0, 100)

fig.suptitle(
    "Figure 5.1: Missingness in General and ASD-Specific Phenotypic Fields",
    fontsize=16,
    y=0.97
)
fig.subplots_adjust(top=0.86, wspace=0.55)
plt.savefig(
    os.path.join(OUTDIR, "figure_5_1_missingness_corrected.png"),
    bbox_inches="tight"
)
plt.close()

# figure 5.2
overall_general_rows = []
total_n = len(df)

for set_name, req_fields in general_requirement_sets.items():
    mask = complete_case_mask(df, req_fields)
    n_retained = int(mask.sum())
    overall_general_rows.append({
        "requirement_set": set_name,
        "n_retained": n_retained,
        "pct_retained": 100 * n_retained / total_n,
        "n_lost": total_n - n_retained,
        "pct_lost": 100 - (100 * n_retained / total_n),
    })

overall_general_df = pd.DataFrame(overall_general_rows)

fig, ax = plt.subplots(figsize=(9.5, 6.2))
sns.barplot(
    data=overall_general_df,
    x="requirement_set",
    y="n_retained",
    ax=ax
)
ax.set_title(
    "Figure 5.2: Full-Cohort Retention Under General Phenotypic Requirements",
    fontsize=16,
    pad=18
)
ax.set_xlabel("")
ax.set_ylabel("Number of Subjects Retained")
ax.tick_params(axis="x", rotation=20)

ymax = overall_general_df["n_retained"].max()
ax.set_ylim(0, ymax * 1.16)

for patch, row in zip(ax.patches, overall_general_df.itertuples(index=False)):
    ax.text(
        patch.get_x() + patch.get_width() / 2,
        patch.get_height() + ymax * 0.02,
        f"{row.n_retained}\n({row.pct_retained:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10
    )

fig.subplots_adjust(top=0.88, bottom=0.18)
plt.savefig(
    os.path.join(OUTDIR, "figure_5_2_full_cohort_retention_general.png"),
    bbox_inches="tight"
)
plt.close()

# figure 5.3
# full-cohort retention by sex and age
sex_general_df = retention_summary(
    df.dropna(subset=["sex_label"]).copy(),
    group_cols=["sex_label"],
    requirement_sets=general_requirement_sets
)

age_general_df = retention_summary(
    df.dropna(subset=["age_label"]).copy(),
    group_cols=["age_label"],
    requirement_sets=general_requirement_sets
)

age_general_df["age_label"] = pd.Categorical(
    age_general_df["age_label"],
    categories=AGE_LABEL_ORDER,
    ordered=True
)

fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.1))

sns.barplot(
    data=sex_general_df,
    x="requirement_set",
    y="pct_retained",
    hue="sex_label",
    hue_order=SEX_ORDER,
    ax=axes[0]
)
axes[0].set_title("By Sex", fontsize=13, pad=10)
axes[0].set_xlabel("")
axes[0].set_ylabel("Percent Retained Within Group")
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend(title="Sex")

sns.barplot(
    data=age_general_df,
    x="requirement_set",
    y="pct_retained",
    hue="age_label",
    hue_order=AGE_LABEL_ORDER,
    ax=axes[1]
)
axes[1].set_title("By Age Group", fontsize=13, pad=10)
axes[1].set_xlabel("")
axes[1].set_ylabel("Percent Retained Within Group")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend(title="Age group", bbox_to_anchor=(1.02, 1), loc="upper left")

fig.suptitle(
    "Figure 5.3: Full-Cohort Retention Under General Requirements by Sex and Age",
    fontsize=16,
    y=0.98
)
fig.subplots_adjust(top=0.82, wspace=0.18, right=0.83, bottom=0.18)
plt.savefig(
    os.path.join(OUTDIR, "figure_5_3_general_retention_by_sex_and_age.png"),
    bbox_inches="tight"
)
plt.close()

# figure 5.4
# ASD-only retention under strict requirements
asd_overall_rows = []
asd_total_n = len(asd_df)

for set_name, req_fields in asd_requirement_sets.items():
    mask = complete_case_mask(asd_df, req_fields)
    n_retained = int(mask.sum())
    asd_overall_rows.append({
        "requirement_set": set_name,
        "n_retained": n_retained,
        "pct_retained": 100 * n_retained / asd_total_n,
        "n_lost": asd_total_n - n_retained,
        "pct_lost": 100 - (100 * n_retained / asd_total_n),
    })

asd_overall_df = pd.DataFrame(asd_overall_rows)

fig, ax = plt.subplots(figsize=(12.5, 6.2))
sns.barplot(
    data=asd_overall_df,
    x="requirement_set",
    y="n_retained",
    ax=ax
)
ax.set_title(
    "Figure 5.4: ASD-Only Retention Under Increasingly Strict Phenotypic Requirements",
    fontsize=16,
    pad=18
)
ax.set_xlabel("")
ax.set_ylabel("Number of ASD Subjects Retained")
ax.tick_params(axis="x", rotation=20)

ymax = asd_overall_df["n_retained"].max()
ax.set_ylim(0, ymax * 1.15)

for patch, row in zip(ax.patches, asd_overall_df.itertuples(index=False)):
    ax.text(
        patch.get_x() + patch.get_width() / 2,
        patch.get_height() + ymax * 0.02,
        f"{row.n_retained}\n({row.pct_retained:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10
    )

fig.subplots_adjust(top=0.88, bottom=0.22)
plt.savefig(
    os.path.join(OUTDIR, "figure_5_4_asd_retention_strict_requirements.png"),
    bbox_inches="tight"
)
plt.close()

# figures 5.5 and 5.6
# ASD-only retention by age and sex
heatmap_requirements = [
    req for req in [
        "FIQ only",
        "FIQ + handedness",
        "FIQ + ADOS total",
        "All selected phenotypic fields",
    ]
    if req in asd_requirement_sets
]

asd_age_sex_rows = []
for age in AGE_ORDER:
    for sex in SEX_ORDER:
        sub = asd_df[
            (asd_df["age_group"] == age) &
            (asd_df["sex_label"] == sex)
        ].copy()

        base_n = len(sub)

        for req_name in heatmap_requirements:
            req_fields = asd_requirement_sets[req_name]
            mask = complete_case_mask(sub, req_fields)
            n_retained = int(mask.sum())
            pct_retained = 100 * n_retained / base_n if base_n > 0 else np.nan

            asd_age_sex_rows.append({
                "age_group": age,
                "age_label": AGE_LABELS[age],
                "sex_label": sex,
                "requirement_set": req_name,
                "base_n": base_n,
                "n_retained": n_retained,
                "pct_retained": pct_retained,
            })

asd_age_sex_df = pd.DataFrame(asd_age_sex_rows)

# figure 5.5
fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
axes = axes.ravel()

for ax, req_name in zip(axes, heatmap_requirements):
    sub = asd_age_sex_df[asd_age_sex_df["requirement_set"] == req_name].copy()

    heatmap_pct = sub.pivot(
        index="age_label",
        columns="sex_label",
        values="pct_retained"
    ).reindex(index=AGE_LABEL_ORDER, columns=SEX_ORDER)

    sns.heatmap(
        heatmap_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        ax=ax
    )
    ax.set_title(req_name, fontsize=12, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.suptitle(
    "Figure 5.5: Percent of ASD Subjects Retained by Age and Sex",
    fontsize=16,
    y=0.97
)
fig.subplots_adjust(top=0.90, hspace=0.28, wspace=0.22)
plt.savefig(
    os.path.join(OUTDIR, "figure_5_5_asd_age_sex_retention_percent_heatmaps.png"),
    bbox_inches="tight"
)
plt.close()

# figure 5.6
fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
axes = axes.ravel()

for ax, req_name in zip(axes, heatmap_requirements):
    sub = asd_age_sex_df[asd_age_sex_df["requirement_set"] == req_name].copy()

    heatmap_n = sub.pivot(
        index="age_label",
        columns="sex_label",
        values="n_retained"
    ).reindex(index=AGE_LABEL_ORDER, columns=SEX_ORDER)

    sns.heatmap(
        heatmap_n,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=ax
    )
    ax.set_title(req_name, fontsize=12, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.suptitle(
    "Figure 5.6: Number of ASD Subjects Retained by Age and Sex",
    fontsize=16,
    y=0.97
)
fig.subplots_adjust(top=0.90, hspace=0.28, wspace=0.22)
plt.savefig(
    os.path.join(OUTDIR, "figure_5_6_asd_age_sex_retention_count_heatmaps.png"),
    bbox_inches="tight"
)
plt.close()

# save CSV summaries
general_missing_df.to_csv(
    os.path.join(OUTDIR, "q3_general_missingness_by_diagnosis.csv"),
    index=False
)

asd_clinical_df.to_csv(
    os.path.join(OUTDIR, "q3_asd_clinical_missingness.csv"),
    index=False
)

overall_general_df.to_csv(
    os.path.join(OUTDIR, "q3_full_cohort_general_retention.csv"),
    index=False
)

sex_general_df.to_csv(
    os.path.join(OUTDIR, "q3_general_retention_by_sex.csv"),
    index=False
)

age_general_df.to_csv(
    os.path.join(OUTDIR, "q3_general_retention_by_age.csv"),
    index=False
)

asd_overall_df.to_csv(
    os.path.join(OUTDIR, "q3_asd_only_retention.csv"),
    index=False
)

asd_age_sex_df.to_csv(
    os.path.join(OUTDIR, "q3_asd_retention_by_age_sex_and_requirement.csv"),
    index=False
)

print(f"\nSaved all Q3 outputs to: {OUTDIR}")