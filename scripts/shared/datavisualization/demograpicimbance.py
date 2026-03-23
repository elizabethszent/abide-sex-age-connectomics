import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "/work/ioannou_lab/elizabeth.szentmiklo/abide-sex-age-connectomics/data/metadata/ABIDE12_phenotypes_combined_fd_0p2.csv"
df = pd.read_csv(path)


df['Diagnosis'] = df['DX_GROUP'].map({1: 'ASD', 2: 'Control'})
df['Sex_Label'] = df['sex'].str.capitalize()
df['Group'] = df['Sex_Label'] + " " + df['Diagnosis']


plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='AGE_GROUP', hue='Group', 
              palette={'Male ASD': '#e67e22', 'Male Control': '#3498db', 
                       'Female ASD': '#f1c40f', 'Female Control': '#9b59b6'},
              order=['child_0_9', 'preteen_10_12', 'teen_13_17', 'adult_18_plus'])

plt.title('Demographic Imbalance: Characterizing Subgroup Distribution for FD 0.2mm', fontsize=15, fontweight='bold')
plt.ylabel('Number of Retained Connectomes', fontsize=12)
plt.xlabel('Developmental Age Group', fontsize=12)
plt.legend(title='Subgroup', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig('demographic_imbalance_chart.png', dpi=300, bbox_inches='tight')