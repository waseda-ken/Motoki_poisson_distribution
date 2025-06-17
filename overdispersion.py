import pandas as pd
import numpy as np
from scipy import stats

# 分割後の各行を 0.5曲として扱う
unit_time = 1.0

df = pd.read_csv('Mrs_total_pronouns.csv')
counts = df['total'].values
n = len(counts)

mean_full = counts.mean()
var_full = counts.var(ddof=1)

# 半曲分にスケーリング（Poisson なら mean,var 共に scale ）
mean_rt = mean_full * unit_time
var_rt = var_full * unit_time
dispersion_index = var_rt / mean_rt

chi2_stat = (n - 1) * dispersion_index
p_val = 1 - stats.chi2.cdf(chi2_stat, df=n - 1)

print(f"[Dispersion test] mean={mean_rt:.3f}, var={var_rt:.3f} (unit={unit_time}曲)")
print(f"Dispersion index={dispersion_index:.3f}")
print(f"χ²-stat={chi2_stat:.2f}, df={n-1}, p-value={p_val:.3f}")
print("過分散あり。" if p_val < 0.05 else "過分散なし。")
