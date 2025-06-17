import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('Mrs_total_pronouns.csv')
counts = df['total'].values
n = len(counts)

mean_c = counts.mean()
var_c = counts.var(ddof=1)
dispersion_index = var_c / mean_c

# 検定統計量： (n-1)*var/mean は自由度 n-1 の χ² に従う
chi2_stat = (n-1) * dispersion_index
p_val = 1 - stats.chi2.cdf(chi2_stat, df=n-1)

print(f"[Dispersion test] mean={mean_c:.3f}, var={var_c:.3f}")
print(f"Dispersion index={dispersion_index:.3f}")
print(f"χ²-stat={chi2_stat:.2f}, df={n-1}, p-value={p_val:.3f}")
if p_val < 0.05:
    print("帰無仮説を棄却する。過分散がある。")
else:
    print("帰無仮説を棄却できない。過分散はない。")