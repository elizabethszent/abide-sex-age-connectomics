import os
import textwrap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# style settings
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.25)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlepad": 14,
    "axes.labelpad": 10,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "legend.frameon": True,
})

# helper function to wrap long site labels
def wrap_site_label(site, width=11):
    return textwrap.fill(
        site.replace("_", "-"),
        width=width,
        break_long_words=False,
        break_on_hyphens=True
    )

# load data
file_path = '/work/ioannou_lab/elizabeth.szentmiklo/abide-sex-age-connectomics/data/metadata/ABIDE12_phenotypes_combined_fd_0p2.csv'
df = pd.read_csv(file_path)

# clean and format the data
age_map = {
    'child_0_9': 'Child\n0–9',
    'preteen_10_12': 'Preteen\n10–12',
    'teen_13_17': 'Teen\n13–17',
    'adult_18_plus': 'Adult\n18+'
}
age_order = ['Child\n0–9', 'Preteen\n10–12', 'Teen\n13–17', 'Adult\n18+']
dx_order = ['ASD', 'Control']

df['FIQ_clean'] = pd.to_numeric(df['FIQ'], errors='coerce')
df['FIQ_clean'] = df['FIQ_clean'].where((df['FIQ_clean'] > 40) & (df['FIQ_clean'] < 200), np.nan)

df['Age_Display'] = df['AGE_GROUP'].map(age_map)
df['DX_Label'] = df['DX_GROUP'].map({1: 'ASD', 2: 'Control'})
df['Sex_Label'] = df['sex'].str.title()

plot_df = df.dropna(subset=['FIQ_clean', 'Age_Display', 'Sex_Label', 'DX_Label']).copy()
plot_df['Age_Display'] = pd.Categorical(plot_df['Age_Display'], categories=age_order, ordered=True)

# keep only top 8 largest sites
top_sites = plot_df['SITE_ID'].value_counts().nlargest(8).index.tolist()
site_df = plot_df[plot_df['SITE_ID'].isin(top_sites)].copy()
site_df['SITE_ID'] = pd.Categorical(site_df['SITE_ID'], categories=top_sites, ordered=True)

wrapped_site_labels = [wrap_site_label(site) for site in top_sites]

# chart 1: IQ Distribution by Sex, Age Group, and Diagnosis
g1 = sns.catplot(
    data=plot_df,
    x='Age_Display',
    y='FIQ_clean',
    hue='Sex_Label',
    col='DX_Label',
    col_order=dx_order,
    kind='box',
    palette='pastel',
    order=age_order,
    height=6.8,
    aspect=1.05,
    sharey=True,
    fliersize=4
)

g1.set_axis_labels('Age Group', 'Full Scale IQ')
g1.set_titles("{col_name} Cohort")
g1.fig.suptitle('IQ Distribution by Sex, Age Group, and Diagnosis', fontsize=24)

for ax in g1.axes.flat:
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=12)

g1.fig.subplots_adjust(top=0.84, bottom=0.18, right=0.86, wspace=0.08)

if g1._legend is not None:
    g1._legend.set_title('Sex_Label')
    g1._legend.set_bbox_to_anchor((1.01, 0.5))

g1.fig.savefig('Q1_Chart1_Master_Stratification.png', bbox_inches='tight', pad_inches=0.2)
plt.close(g1.fig)

# chart 2: Average IQ by Top Scanning Sites and Diagnosis
fig, ax = plt.subplots(figsize=(16, 8))

sns.barplot(
    data=site_df,
    x='SITE_ID',
    y='FIQ_clean',
    hue='DX_Label',
    hue_order=dx_order,
    order=top_sites,
    palette='Set2',
    errorbar='ci',
    capsize=0.08,
    ax=ax
)

ax.set_title('Average IQ by Top Scanning Sites and Diagnosis', fontsize=22)
ax.set_xlabel('Scanning Site')
ax.set_ylabel('Mean Full Scale IQ')
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(wrapped_site_labels, rotation=0, ha='center')

ax.legend(title='Diagnosis', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

fig.subplots_adjust(left=0.08, right=0.82, bottom=0.18, top=0.90)
fig.savefig('Q1_Chart2_Site_Variance.png', bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

# chart 3: Heatmap of Average IQ by Site and Age Group
heatmap_data = site_df.pivot_table(
    values='FIQ_clean',
    index='SITE_ID',
    columns='Age_Display',
    aggfunc='mean'
).reindex(index=top_sites, columns=age_order)

fig, ax = plt.subplots(figsize=(13, 9))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    mask=heatmap_data.isna(),
    linewidths=0.7,
    linecolor='white',
    cbar_kws={'label': 'Mean Full Scale IQ', 'shrink': 0.9},
    annot_kws={'size': 14},
    ax=ax
)

ax.set_title('Heatmap of Average IQ by Site and Age Group', fontsize=22)
ax.set_xlabel('Age Group')
ax.set_ylabel('Scanning Site')
ax.tick_params(axis='x', rotation=0)
ax.set_yticklabels([wrap_site_label(site) for site in heatmap_data.index], rotation=0)

fig.subplots_adjust(left=0.20, right=0.96, bottom=0.12, top=0.90)
fig.savefig('Q1_Chart3_Age_Site_Heatmap.png', bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

# chart 4: Distribution of IQ by Sex Across Top Scanning Sites
fig, ax = plt.subplots(figsize=(16, 8))

sns.boxplot(
    data=site_df,
    x='SITE_ID',
    y='FIQ_clean',
    hue='Sex_Label',
    order=top_sites,
    palette='pastel',
    ax=ax
)

ax.set_title('Distribution of IQ by Sex Across Top Scanning Sites', fontsize=22)
ax.set_xlabel('Scanning Site')
ax.set_ylabel('Full Scale IQ')
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(wrapped_site_labels, rotation=0, ha='center')

ax.legend(title='Sex', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

fig.subplots_adjust(left=0.08, right=0.82, bottom=0.18, top=0.90)
fig.savefig('Q1_Chart4_Sex_Site_Match.png', bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

print("All revised charts saved successfully.")