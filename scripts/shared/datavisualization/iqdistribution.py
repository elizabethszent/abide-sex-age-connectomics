import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load phenotypic data
path = "/work/ioannou_lab/elizabeth.szentmiklo/abide-sex-age-connectomics/data/metadata/ABIDE12_phenotypes_combined_fd_0p3.csv"
df = pd.read_csv(path)

# cleaning: Mapping Diagnosis labels for the legend
df['Diagnosis'] = df['DX_GROUP'].map({1: 'ASD', 2: 'Control'})

# visualization: Kernel Density Estimate (KDE) plot of IQ
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='FIQ', hue='Diagnosis', fill=True, palette='viridis', alpha=0.5)

plt.title('Analytical Profiling: IQ Distribution by Diagnosis for FD 0.3mm', fontsize=14, fontweight='bold')
plt.xlabel('Full-Scale IQ (FIQ)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# analytical Note: Identifying missing data (a core cleaning requirement)
missing_iq = df['FIQ'].isna().sum()
plt.annotate(f'Missing IQ Values: {missing_iq}', xy=(0.05, 0.9), xycoords='axes fraction', 
             fontsize=10, color='red', fontweight='bold')

plt.savefig('iq_distribution_analytics.png', dpi=300, bbox_inches='tight')