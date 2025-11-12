import pandas as pd

# Binary metrics you already made
binm = pd.read_csv("results/metrics_merged.csv")   # has global_clustering (binary)

# Weighted metrics from the script I gave you
w = pd.read_csv("results/weighted_clustering_subjects.csv")  # has Cw_emp

# Merge and compare
m = pd.merge(binm[["FILE_ID","DX_GROUP","global_clustering"]],
             w[["FILE_ID","Cw_emp"]], on="FILE_ID")

print("Corr(binary vs weighted clustering):",
      m["global_clustering"].corr(m["Cw_emp"]))

# Group means
m["group"] = m["DX_GROUP"].map({1:"ASD",2:"Control"})
print("\nBinary clustering means:\n", m.groupby("group")["global_clustering"].mean())
print("\nWeighted clustering means:\n", m.groupby("group")["Cw_emp"].mean())
