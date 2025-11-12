import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
MATRIX_DIR = r"data\connectivity_matrices"
PHENO_PATH = r"data\filtered_groups\females_all.csv"
OUT_DIR = r"data\group_averages"

os.makedirs(OUT_DIR, exist_ok=True)

# Load filtered phenotypic data
pheno = pd.read_csv(PHENO_PATH)

# Define helper function to load and average matrices
def average_matrices(file_ids):
    matrices = []
    for fid in file_ids:
        path = os.path.join(MATRIX_DIR, f"{fid}_corr.npy")
        if os.path.exists(path):
            mat = np.load(path)
            matrices.append(mat)
    if not matrices:
        return None
    matrices = np.stack(matrices)
    return np.nanmean(matrices, axis=0)

# Get unique combinations of age group + diagnosis
groups = pheno[['AGE_GROUP', 'DX_GROUP']].dropna().drop_duplicates()

for _, row in tqdm(groups.iterrows(), total=len(groups)):
    age_group = row['AGE_GROUP']
    dx_group = int(row['DX_GROUP'])
    dx_label = 'ASD' if dx_group == 2 else 'Control'

    group = pheno[(pheno['AGE_GROUP'] == age_group) & (pheno['DX_GROUP'] == dx_group)]
    file_ids = group['FILE_ID'].tolist()

    avg_mat = average_matrices(file_ids)
    if avg_mat is not None:
        out_file = os.path.join(OUT_DIR, f"{age_group}_{dx_label}_mean.npy")
        np.save(out_file, avg_mat)
        print(f"Saved {out_file} with {len(file_ids)} participants")
