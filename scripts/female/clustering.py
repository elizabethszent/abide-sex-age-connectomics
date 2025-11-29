import pandas as pd

#Binary metrics you already made
binm = pd.read_csv("data/female/metrics_merged.csv")  #has globalclustering

#Weighted metrics from the script I gave you
w = pd.read_csv("data/female/weighted_clustering_subjects.csv")

#Merge and compare
m = pd.merge(binm[["FILE_ID","DX_GROUP","global_clustering"]],
             w[["FILE_ID","Cw_emp"]], on="FILE_ID")

print("Corr(binary vs weighted clustering):",
      m["global_clustering"].corr(m["Cw_emp"]))

#group means
m["group"] = m["DX_GROUP"].map({1:"ASD",2:"Control"})
print("\nBinary clustering means:\n", m.groupby("group")["global_clustering"].mean())
print("\nWeighted clustering means:\n", m.groupby("group")["Cw_emp"].mean())
