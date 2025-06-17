import pandas as pd
import numpy as np
from scipy import stats

unit_time = 0.5

df = pd.read_csv('Mrs_total_pronouns.csv')
counts = df['total'].values
n = len(counts)

lambda_full = counts.mean()
lambda_rt = lambda_full * unit_time
print(f"[KS test] サンプル数={n}, 単位時間={unit_time}曲あたり λ̂={lambda_rt:.3f}")

# シミュレーション
np.random.seed(0)
sim = stats.poisson.rvs(mu=lambda_rt, size=n)

ks_stat, ks_p = stats.ks_2samp(counts, sim)
print(f"KS-statistic={ks_stat:.3f}, p-value={ks_p:.3f}")
print("ポアソンに従う。" if ks_p >= 0.05 else "ポアソンに従わない。")
