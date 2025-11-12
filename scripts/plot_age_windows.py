import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/age_window_tests.csv")
fig, ax = plt.subplots(figsize=(7,4))
for y, g in df.groupby("outcome"):
    ax.plot(g["center"], g["p"], marker="o", label=y)
ax.axhline(0.05, ls="--")
ax.axhline(0.10, ls=":")
ax.set_xlabel("Window center age (years)")
ax.set_ylabel("ASD vs Control p-value")
ax.set_title("Sliding-window group tests (6-year windows)")
ax.legend()
Path("results").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig("results/age_window_pvals.png", dpi=200)
print("Saved results/age_window_pvals.png")
