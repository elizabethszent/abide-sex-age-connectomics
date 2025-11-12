import numpy as np

#single subject
m = np.load(r"data\connectivity_matrices\KKI_0050783_corr.npy")  # example
print(m.shape)   # -> (200, 200)

#group average
g = np.load(r"data\group_averages\child_ASD_mean.npy")
print(g.shape)   # -> (200, 200)