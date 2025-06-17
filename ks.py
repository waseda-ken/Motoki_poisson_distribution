import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('Mrs_total_pronouns.csv')
counts = df['total'].values
n = len(counts)

lambda_hat = counts.mean()
print(f"[KS test] サンプル数={n}, λ={lambda_hat:.3f}")

# ポアソン分布からシミュレーションデータ生成
np.random.seed(0)
sim = stats.poisson.rvs(mu=lambda_hat, size=n)

ks_stat, ks_p = stats.ks_2samp(counts, sim)
print(f"KS-statistic={ks_stat:.3f}, p-value={ks_p:.3f}")

if ks_p < 0.05:
    print("帰無仮説を棄却する。ポアソン分布に従わない。")
else:
    print("帰無仮説を棄却できない。ポアソン分布に従う。")