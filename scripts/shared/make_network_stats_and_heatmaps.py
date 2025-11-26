import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os

ROOT = Path("C:/Users/eliza/CPSC_599_CONNECTOMICS/TERMProject")
IN_DIR = ROOT / "results/network_level/2"
OUT_DIR = ROOT / "results/network_level/2"
OUT_DIR.mkdir(exist_ok=True, parents=True)

SEXES = ["female", "male"]
AGE_GROUPS = ["child", "teen", "young_adult"]

#number of modules in your Louvain solution
N_MOD = 7 

def cohen_d(x_asd, x_ctl):
    """Cohen's d (CTL - ASD)."""
    n1, n2 = len(x_asd), len(x_ctl)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = x_asd.mean(), x_ctl.mean()
    s1, s2 = x_asd.std(ddof=1), x_ctl.std(ddof=1)
    sp = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan
    return (m2 - m1) / sp #CTL - ASD

for sex in SEXES:
    for age in AGE_GROUPS:
        fname = f"{sex}_{age}_network_connectivity_wide.csv"
        fpath = IN_DIR / fname
        if not fpath.exists():
            print(f"[SKIP] {fpath} not found")
            continue

        print(f"\n=== {sex.upper()} | {age.upper()} ===")
        df = pd.read_csv(fpath)

        #identify network columns
        net_cols = [c for c in df.columns if c.startswith("net")]
        net_cols = sorted(net_cols)

        results = []
        pvals = []

        for net in net_cols:
            sub = df[["DX_GROUP", "AGE_AT_SCAN", "func_mean_fd", net]].dropna()
            if sub["DX_GROUP"].nunique() != 2:
                print(f"  [WARN] {net}: not both groups present, skipping")
                continue

            #fit OLS: DX_GROUP coded as categorical, 1 = ASD (reference), 2 = CTL
            formula = f"{net} ~ C(DX_GROUP) + AGE_AT_SCAN + func_mean_fd"
            m = smf.ols(formula, data=sub).fit()

            #coefficient for CTL vs ASD
            term = "C(DX_GROUP)[T.2]"
            beta = m.params.get(term, np.nan)
            p = m.pvalues.get(term, np.nan)

            asd = sub[sub["DX_GROUP"] == 1][net]
            ctl = sub[sub["DX_GROUP"] == 2][net]
            mean_asd = asd.mean()
            mean_ctl = ctl.mean()
            d = cohen_d(asd, ctl)

            res = {
                "network": net,
                "beta_CTL_minus_ASD": beta,
                "p_uncorrected": p,
                "mean_ASD": mean_asd,
                "mean_CTL": mean_ctl,
                "cohen_d_CTL_minus_ASD": d,
                "n_ASD": len(asd),
                "n_CTL": len(ctl),
            }
            results.append(res)
            pvals.append(p)

        if not results:
            print("  No valid networks, skipping.")
            continue

        stats_df = pd.DataFrame(results)

        #FDR across networks
        reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        stats_df["p_FDR"] = p_fdr
        stats_df["FDR_significant"] = reject

        out_stats = OUT_DIR / f"{sex}_{age}_network_stats.csv"
        stats_df.to_csv(out_stats, index=False)
        print(f"  Saved stats -> {out_stats}")

        #Heatmap of effect size (Cohen's d)

        #Build symmetric matrix from net{i}_{j}
        mat_d = np.zeros((N_MOD, N_MOD)) * np.nan

        for _, row in stats_df.iterrows():
            name = row["network"]  #e.g., "net1_3"
            try:
                _, pair = name.split("net")
                i_str, j_str = pair.split("_")
                i, j = int(i_str), int(j_str)
            except Exception:
                continue

            d = row["cohen_d_CTL_minus_ASD"]
            mat_d[i-1, j-1] = d
            mat_d[j-1, i-1] = d

        plt.figure(figsize=(5, 4))
        vmax = np.nanmax(np.abs(mat_d))
        if np.isnan(vmax) or vmax == 0:
            vmax = 0.2

        im = plt.imshow(mat_d, vmin=-vmax, vmax=vmax, cmap="coolwarm")
        plt.colorbar(im, label="Cohen's d (CTL - ASD)")
        plt.xticks(range(N_MOD), [f"M{i}" for i in range(1, N_MOD+1)])
        plt.yticks(range(N_MOD), [f"M{i}" for i in range(1, N_MOD+1)])
        plt.title(f"{sex.capitalize()} {age} – network connectivity\n(ASD vs Control)")

        plt.tight_layout()
        out_fig = OUT_DIR / f"{sex}_{age}_network_effectsizes_heatmap.png"
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"  Saved heatmap -> {out_fig}")

print("\nAll done.")