#scripts/group_test.py
import pandas as pd
from scipy import stats
from pathlib import Path

meta = pd.read_csv("results/female_metadata_included.csv")
metrics = pd.read_csv("results/subject_metrics_10pct.csv")
df = meta.merge(metrics, on="FILE_ID", how="inner")

asd = df[df["DX_GROUP"] == 1]#1 = ASD
ctl = df[df["DX_GROUP"] == 2]#2 = Control

for col in ["mean_degree","global_clustering","global_efficiency"]:
    t, p = stats.ttest_ind(asd[col], ctl[col], equal_var=False, nan_policy="omit")
    print(f"{col:18s} | ASD mean={asd[col].mean():.4f} | CTL mean={ctl[col].mean():.4f} | p={p:.4g}")

df.to_csv("results/metrics_merged.csv", index=False)
print("Saved: results/metrics_merged.csv")
