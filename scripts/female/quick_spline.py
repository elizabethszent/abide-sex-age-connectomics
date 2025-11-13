import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy import bs

#load the metrics you already created
df = pd.read_csv('data/female/metrics_merged.csv')

#keep rows with required fields and ensure numeric dtypes
df = df.dropna(subset=['DX_GROUP','AGE_AT_SCAN','func_mean_fd'])
for col in ['AGE_AT_SCAN','func_mean_fd','mean_degree','global_clustering','global_efficiency']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['mean_degree','global_clustering','global_efficiency'])

#treat group as categorical (1=ASD, 2=Control in ABIDE I)
df['DX_GROUP'] = df['DX_GROUP'].astype('category')

outcomes = ['mean_degree','global_clustering','global_efficiency']

for y in outcomes:
    #cubic B-spline for age (df=4), include interaction with group, control for motion
    formula = f"{y} ~ C(DX_GROUP) * bs(AGE_AT_SCAN, df=4) + func_mean_fd"
    m = smf.ols(formula=formula, data=df).fit()

    print("\nOutcome:", y)

    param_index = list(m.params.index)
    inter_cols = [c for c in param_index if ('C(DX_GROUP)' in c and 'bs(' in c and ':' in c)]
    if inter_cols:
        R = np.zeros((len(inter_cols), len(param_index)))
        for i, col in enumerate(inter_cols):
            R[i, param_index.index(col)] = 1.0
        ft = m.f_test(R)
        print("  group × age-spline: F p =", float(ft.pvalue))

    #main effect of group (overall ASD vs Control difference)
    grp_ps = [p for name, p in m.pvalues.items()
              if name.startswith("C(DX_GROUP)") and ":bs(" not in name]
    if grp_ps:
        print("  group main effect p =", float(min(grp_ps)))
