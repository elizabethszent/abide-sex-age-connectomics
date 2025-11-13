import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

# Paths
ROI_DIR = r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\roi_timeseries\cpac\nofilt_noglobal\rois_cc200"
OUTPUT_DIR = r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\connectivity_matrices"
PHENO_PATH = r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject\data\Phenotypic_V1_0b_preprocessed1.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load phenotypic file (for filtering later)
pheno = pd.read_csv(PHENO_PATH)

# Get all files
files = [f for f in os.listdir(ROI_DIR) if f.endswith('.1D')]

for f in tqdm(files):
    file_id = f.split('_rois')[0]
    path = os.path.join(ROI_DIR, f)

    # Load ROI time series
    ts = np.loadtxt(path)
    
    # Compute correlation matrix
    corr = np.corrcoef(ts.T)

    # Save as numpy binary
    np.save(os.path.join(OUTPUT_DIR, f"{file_id}_corr.npy"), corr)
