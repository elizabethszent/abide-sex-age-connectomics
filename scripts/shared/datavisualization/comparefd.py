import pandas as pd
import matplotlib.pyplot as plt

# load counts from both files
path = "/work/ioannou_lab/elizabeth.szentmiklo/abide-sex-age-connectomics/data/metadata/ABIDE12_phenotypes_combined_fd_0p2.csv"
df02 = pd.read_csv(path)
path = "/work/ioannou_lab/elizabeth.szentmiklo/abide-sex-age-connectomics/data/metadata/ABIDE12_phenotypes_combined_fd_0p3.csv"
df03 = pd.read_csv(path)

counts = {
    'Strict (0.2mm)': len(df02),
    'Permissive (0.3mm)': len(df03)
}

# plot
plt.figure(figsize=(8, 5))
bars = plt.bar(counts.keys(), counts.values(), color=['#e74c3c', '#3498db'])

plt.title('Impact of Cleaning: Sample Retention by Motion Threshold', fontsize=14, fontweight='bold')
plt.ylabel('Number of Subjects Retained', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, yval, ha='center', va='bottom', fontweight='bold')

plt.savefig('retention_comparison.png', dpi=300, bbox_inches='tight')