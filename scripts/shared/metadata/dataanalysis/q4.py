import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

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
    "q4_missingness_association_fd0p3_corrected"
)
os.makedirs(OUTDIR, exist_ok=True)

# plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

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

# field definitions
general_fields = [f for f in ["FIQ", "RIGHT_HANDED"] if f in df.columns]

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
    if f in df.columns
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

asd_df = df[df["diagnosis"] == "ASD"].copy()

# summary scores
df["n_missing_general_fields"] = df[general_fields].isna().sum(axis=1)
asd_df["n_missing_clinical_fields"] = asd_df[clinical_asd_fields].isna().sum(axis=1)
asd_df["pct_missing_clinical_fields"] = 100 * asd_df["n_missing_clinical_fields"] / len(clinical_asd_fields)

# helpers
def cramers_v(x, y):
    """
    Bias-corrected Cramér's V.
    """
    tab = pd.crosstab(x, y)

    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan, np.nan, tab

    chi2, p_value, _, _ = chi2_contingency(tab)
    n = tab.values.sum()

    if n == 0:
        return np.nan, np.nan, tab

    phi2 = chi2 / n
    r, k = tab.shape

    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    denom = min((k_corr - 1), (r_corr - 1))
    if denom <= 0:
        return np.nan, p_value, tab

    v = np.sqrt(phi2_corr / denom)
    return v, p_value, tab

def add_missing_indicator_table(dataframe, fields, predictors, prefix):
    rows = []
    for field in fields:
        outcome = dataframe[field].isna().map({True: "Missing", False: "Available"})
        for predictor_name, predictor_col in predictors.items():
            sub = dataframe[[predictor_col]].copy()
            sub["missingness"] = outcome
            sub = sub.dropna(subset=[predictor_col, "missingness"])

            v, p_value, _ = cramers_v(sub[predictor_col], sub["missingness"])

            rows.append({
                "analysis_block": prefix,
                "field": field,
                "field_label": field_labels[field],
                "predictor": predictor_name,
                "cramers_v": v,
                "p_value": p_value,
                "n_rows": len(sub),
                "n_levels_predictor": sub[predictor_col].nunique()
            })
    return pd.DataFrame(rows)

# predictors
full_predictors = {
    "Site": "SITE_ID",
    "Age group": "age_label",
    "Diagnosis": "diagnosis",
    "Sex": "sex_label",
}

asd_predictors = {
    "Site": "SITE_ID",
    "Age group": "age_label",
    "Sex": "sex_label",
}

# association tables
general_assoc_df = add_missing_indicator_table(
    dataframe=df,
    fields=general_fields,
    predictors=full_predictors,
    prefix="general_fields_full_cohort"
)

clinical_assoc_df = add_missing_indicator_table(
    dataframe=asd_df,
    fields=clinical_asd_fields,
    predictors=asd_predictors,
    prefix="clinical_fields_asd_only"
)

general_assoc_summary = (
    general_assoc_df.groupby("predictor", as_index=False)
    .agg(mean_cramers_v=("cramers_v", "mean"))
)

clinical_assoc_summary = (
    clinical_assoc_df.groupby("predictor", as_index=False)
    .agg(mean_cramers_v=("cramers_v", "mean"))
)

# figure 6.1
# general fields: association strength heatmap
general_heatmap = general_assoc_df.pivot(
    index="field_label",
    columns="predictor",
    values="cramers_v"
).reindex(
    index=[field_labels[f] for f in general_fields],
    columns=["Site", "Age group", "Diagnosis", "Sex"]
)

plt.figure(figsize=(8.5, 4.8))
sns.heatmap(
    general_heatmap,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=max(0.5, np.nanmax(general_heatmap.values))
)
plt.title(
    "Figure 6.1: Association Strength for Missingness in General Fields",
    fontsize=13,
    pad=10
)
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "figure_6_1_general_missingness_association_heatmap.png"),
    bbox_inches="tight"
)
plt.close()

# figure 6.2
# ASD-only clinical fields: association strength heatmap
clinical_heatmap = clinical_assoc_df.pivot(
    index="field_label",
    columns="predictor",
    values="cramers_v"
).reindex(
    index=[field_labels[f] for f in clinical_asd_fields],
    columns=["Site", "Age group", "Sex"]
)

plt.figure(figsize=(8.5, 8.5))
sns.heatmap(
    clinical_heatmap,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=max(0.5, np.nanmax(clinical_heatmap.values))
)
plt.title(
    "Figure 6.2: Association Strength for Missingness in ASD-Specific Clinical Fields",
    fontsize=13,
    pad=10
)
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "figure_6_2_clinical_missingness_association_heatmap.png"),
    bbox_inches="tight"
)
plt.close()

# figure 6.3
# general field missingness by site
top_sites = df["SITE_ID"].value_counts().head(15).index.tolist()

site_general_summary = (
    df[df["SITE_ID"].isin(top_sites)]
    .groupby("SITE_ID", as_index=False)
    .agg(
        n_subjects=("SITE_ID", "size"),
        mean_missing_general=("n_missing_general_fields", "mean"),
        pct_fiq_missing=("FIQ", lambda s: 100 * s.isna().mean()),
        pct_handed_missing=("RIGHT_HANDED", lambda s: 100 * s.isna().mean())
    )
    .sort_values("mean_missing_general", ascending=False)
)

fig, axes = plt.subplots(1, 2, figsize=(19, 6.8))

sns.barplot(
    data=site_general_summary,
    x="SITE_ID",
    y="pct_fiq_missing",
    ax=axes[0]
)
axes[0].set_title("FIQ missingness by site", fontsize=12, pad=8)
axes[0].set_xlabel("Site")
axes[0].set_ylabel("Percent Missing")
plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

sns.barplot(
    data=site_general_summary,
    x="SITE_ID",
    y="pct_handed_missing",
    ax=axes[1]
)
axes[1].set_title("Handedness missingness by site", fontsize=12, pad=8)
axes[1].set_xlabel("Site")
axes[1].set_ylabel("Percent Missing")
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

fig.suptitle(
    "Figure 6.3: General Field Missingness Across Major Sites",
    fontsize=14,
    y=0.97
)
fig.subplots_adjust(top=0.82, bottom=0.30, wspace=0.28)
plt.savefig(
    os.path.join(OUTDIR, "figure_6_3_general_missingness_by_site.png"),
    bbox_inches="tight"
)
plt.close()

# figure 6.4
# general field missingness by age/diagnosis
fiq_age_dx = (
    df.groupby(["age_label", "diagnosis"], as_index=False, observed=False)
    .agg(pct_fiq_missing=("FIQ", lambda s: 100 * s.isna().mean()))
)

fiq_age_dx_pivot = fiq_age_dx.pivot(
    index="age_label",
    columns="diagnosis",
    values="pct_fiq_missing"
).reindex(index=AGE_LABEL_ORDER, columns=DX_ORDER)

handed_age_dx = (
    df.groupby(["age_label", "diagnosis"], as_index=False, observed=False)
    .agg(pct_handed_missing=("RIGHT_HANDED", lambda s: 100 * s.isna().mean()))
)

handed_age_dx_pivot = handed_age_dx.pivot(
    index="age_label",
    columns="diagnosis",
    values="pct_handed_missing"
).reindex(index=AGE_LABEL_ORDER, columns=DX_ORDER)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(
    fiq_age_dx_pivot,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    vmin=0,
    vmax=100,
    ax=axes[0]
)
axes[0].set_title("FIQ missingness", fontsize=12, pad=8)
axes[0].set_xlabel("")
axes[0].set_ylabel("")
plt.setp(axes[0].get_yticklabels(), rotation=0)
plt.setp(axes[0].get_xticklabels(), rotation=0)

sns.heatmap(
    handed_age_dx_pivot,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    vmin=0,
    vmax=100,
    ax=axes[1]
)
axes[1].set_title("Handedness missingness", fontsize=12, pad=8)
axes[1].set_xlabel("")
axes[1].set_ylabel("")
plt.setp(axes[1].get_yticklabels(), rotation=0)
plt.setp(axes[1].get_xticklabels(), rotation=0)

fig.suptitle(
    "Figure 6.5: General Field Missingness by Age Group and Diagnosis",
    fontsize=14,
    y=0.97
)
fig.subplots_adjust(top=0.82, wspace=0.35, left=0.10)
plt.savefig(
    os.path.join(OUTDIR, "figure_6_4_general_missingness_by_age_and_diagnosis.png"),
    bbox_inches="tight"
)
plt.close()

# figure 6.5
# ASD-only clinical missingness by site
asd_top_sites = asd_df["SITE_ID"].value_counts().head(15).index.tolist()

asd_site_summary = (
    asd_df[asd_df["SITE_ID"].isin(asd_top_sites)]
    .groupby("SITE_ID", as_index=False)
    .agg(
        n_subjects=("SITE_ID", "size"),
        mean_pct_missing_clinical=("pct_missing_clinical_fields", "mean"),
        pct_ados_total_missing=("ADOS_TOTAL", lambda s: 100 * s.isna().mean() if "ADOS_TOTAL" in asd_df.columns else np.nan),
    )
    .sort_values("mean_pct_missing_clinical", ascending=False)
)

plt.figure(figsize=(15, 6.5))
ax = sns.barplot(
    data=asd_site_summary,
    x="SITE_ID",
    y="mean_pct_missing_clinical"
)
ax.set_title(
    "Figure 6.4: ASD Clinical Missingness Across Major Sites",
    fontsize=14,
    pad=12
)
ax.set_xlabel("Site")
ax.set_ylabel("Mean Percent Missing Clinical Fields")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

for patch, row in zip(ax.patches, asd_site_summary.itertuples(index=False)):
    ax.text(
        patch.get_x() + patch.get_width() / 2,
        patch.get_height() + 1,
        f"n={row.n_subjects}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.subplots_adjust(bottom=0.28, top=0.86)
plt.savefig(
    os.path.join(OUTDIR, "figure_6_5_clinical_missingness_by_site.png"),
    bbox_inches="tight"
)
plt.close()

# figure 6.6
# ASD-only clinical missingness by age and sex
asd_age_sex_summary = (
    asd_df.groupby(["age_label", "sex_label"], as_index=False, observed=False)
    .agg(
        mean_pct_missing_clinical=("pct_missing_clinical_fields", "mean"),
        n_subjects=("sex_label", "size")
    )
)

asd_age_sex_pivot = asd_age_sex_summary.pivot(
    index="age_label",
    columns="sex_label",
    values="mean_pct_missing_clinical"
).reindex(index=AGE_LABEL_ORDER, columns=SEX_ORDER)

plt.figure(figsize=(7.4, 6.0))
sns.heatmap(
    asd_age_sex_pivot,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    vmin=0,
    vmax=100
)
plt.title(
    "Figure 6.6: ASD Clinical Missingness by Age Group and Sex",
    fontsize=13,
    pad=10
)
plt.xlabel("")
plt.ylabel("")
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTDIR, "figure_6_6_clinical_missingness_by_age_and_sex.png"),
    bbox_inches="tight"
)
plt.close()

# save CSV summaries
general_assoc_df.to_csv(
    os.path.join(OUTDIR, "q4_general_missingness_associations.csv"),
    index=False
)

clinical_assoc_df.to_csv(
    os.path.join(OUTDIR, "q4_clinical_missingness_associations_asd_only.csv"),
    index=False
)

general_assoc_summary.to_csv(
    os.path.join(OUTDIR, "q4_general_missingness_summary_by_predictor.csv"),
    index=False
)

clinical_assoc_summary.to_csv(
    os.path.join(OUTDIR, "q4_clinical_missingness_summary_by_predictor_asd_only.csv"),
    index=False
)

site_general_summary.to_csv(
    os.path.join(OUTDIR, "q4_general_missingness_by_site.csv"),
    index=False
)

asd_site_summary.to_csv(
    os.path.join(OUTDIR, "q4_clinical_missingness_by_site_asd_only.csv"),
    index=False
)

asd_age_sex_summary.to_csv(
    os.path.join(OUTDIR, "q4_clinical_missingness_by_age_and_sex_asd_only.csv"),
    index=False
)

# console output
print("\n" + "=" * 80)
print("Q4 GENERAL FIELD MISSINGNESS ASSOCIATIONS")
print("=" * 80)
print(general_assoc_df.sort_values(["field_label", "cramers_v"], ascending=[True, False]).to_string(index=False))

print("\n" + "=" * 80)
print("Q4 ASD-ONLY CLINICAL FIELD MISSINGNESS ASSOCIATIONS")
print("=" * 80)
print(clinical_assoc_df.sort_values(["field_label", "cramers_v"], ascending=[True, False]).to_string(index=False))

print("\n" + "=" * 80)
print("MEAN CRAMER'S V BY PREDICTOR (GENERAL FIELDS)")
print("=" * 80)
print(general_assoc_summary.sort_values("mean_cramers_v", ascending=False).to_string(index=False))

print("\n" + "=" * 80)
print("MEAN CRAMER'S V BY PREDICTOR (ASD-ONLY CLINICAL FIELDS)")
print("=" * 80)
print(clinical_assoc_summary.sort_values("mean_cramers_v", ascending=False).to_string(index=False))

print(f"\nSaved all Q4 outputs to: {OUTDIR}")