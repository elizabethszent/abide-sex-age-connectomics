import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pick any included female subject (change iloc[0] if you want a specific one)
meta = pd.read_csv('results/metrics_merged.csv')
subj = meta['FILE_ID'].iloc[0]

# path to that subject's ROI time series
ts_path = os.path.join('data','roi_timeseries','cpac','nofilt_noglobal','rois_cc200',
                       f'{subj}_rois_cc200.1D')


ts = pd.read_csv(ts_path,
                 sep=r"\s+",        
                 header=None,
                 comment="#", #ignore commented lines if present
                 engine="python")
ts = ts.apply(pd.to_numeric, errors="coerce")  #force numeric, turn junk into NaN

# drop columns with any NaN or zero variance (can’t correlate these)
non_nan  = np.isfinite(ts).all(axis=0).values
non_zero = ts.std(axis=0, ddof=0).values > 0
good = non_nan & non_zero
bad_count = int((~good).sum())
ts = ts.loc[:, good]
print(f'Loaded {ts_path}')
print(f'Kept {ts.shape[1]} ROIs (dropped {bad_count} problematic columns).')

# a small 8×8 preview table for your slide
head = ts.iloc[:8, :8].copy()
head.columns = [f'ROI{i+1}' for i in range(head.shape[1])]
os.makedirs('results', exist_ok=True)
head.to_csv('results/ts_head_for_slide.csv', index=False)

# correlation matrix (connectome) + heatmap
C = np.corrcoef(ts.values.T)
np.fill_diagonal(C, 0)  # nicer visually

plt.figure(figsize=(6,5))
im = plt.imshow(C, vmin=-1, vmax=1, cmap='coolwarm', origin='lower')
plt.colorbar(im, label='Pearson r')
plt.title(f'{subj} – CC200 correlation matrix')
plt.xlabel('ROI'); plt.ylabel('ROI')
plt.tight_layout()
plt.savefig('results/connectome_heatmap.png', dpi=200)
plt.close()

print('Wrote: results/ts_head_for_slide.csv and results/connectome_heatmap.png')
