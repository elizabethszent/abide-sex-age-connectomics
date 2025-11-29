import os
import pandas as pd
import numpy as np

# paths 
PHENO_PATH = r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\processed\Phenotypic_V1_0b_preprocessed1.csv"
MATRIX_DIR = r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\connectivity_matrices"
OUT_DIR = r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\female"

os.makedirs(OUT_DIR, exist_ok=True)

#load phenotypic file
pheno = pd.read_csv(PHENO_PATH)

#filter for females only
#in ABIDE, SEX: 1 = Male, 2 = Female
females = pheno[pheno['SEX'] == 2].copy()
print(f"Total females: {len(females)}")

#filter for those who have an existing matrix file
all_files = [f.replace('_corr.npy', '') for f in os.listdir(MATRIX_DIR) if f.endswith('_corr.npy')]
females = females[females['FILE_ID'].isin(all_files)]
print(f"Females with available matrices: {len(females)}")


bins = [0, 13, 18, 30, 45, 100]
labels = ['child', 'teen', 'young_adult', 'adult', 'older']
females['AGE_GROUP'] = pd.cut(females['AGE_AT_SCAN'], bins=bins, labels=labels)

#split by ASD vs Control
#in ABIDE, DX_GROUP: 1 = Control, 2 = ASD
asd = females[females['DX_GROUP'] == 2]
control = females[females['DX_GROUP'] == 1]

#save filtered participant lists
females.to_csv(os.path.join(OUT_DIR, "females_all.csv"), index=False)
asd.to_csv(os.path.join(OUT_DIR, "females_asd.csv"), index=False)
control.to_csv(os.path.join(OUT_DIR, "females_control.csv"), index=False)

#summarize
summary = females.groupby(['AGE_GROUP', 'DX_GROUP']).size().reset_index(name='count')
summary['DX_GROUP'] = summary['DX_GROUP'].replace({1: 'Control', 2: 'ASD'})
summary.to_csv(os.path.join(OUT_DIR, "summary_by_age.csv"), index=False)

print("\nSummary of females by age group and diagnosis:")
print(summary)
