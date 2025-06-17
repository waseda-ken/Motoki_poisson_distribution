import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 分割後の各行を 0.5曲として扱う
unit_time = 1.0

df = pd.read_csv('Mrs_total_pronouns.csv')
counts = df['total'].values
n = len(counts)

lambda_full = counts.mean()
lambda_rt = lambda_full * unit_time
print(f"サンプル数={n}, 推定 λ̂={lambda_rt:.3f} (unit={unit_time}曲)")

# 観測度数
max_k = counts.max()
bins = np.arange(0, max_k + 2) - 0.5
obs_freq, _ = np.histogram(counts, bins=bins)

# 期待度数 (尾部まとめ込み)
k = np.arange(0, max_k + 1)
pmf = stats.poisson.pmf(k, mu=lambda_rt)
exp_freq = pmf * n
tail_prob = 1 - stats.poisson.cdf(max_k, mu=lambda_rt)
exp_tail = tail_prob * n

obs_grp = np.append(obs_freq[:-1], obs_freq[-1])
exp_grp = np.append(exp_freq[:-1], exp_freq[-1] + exp_tail)

# χ² 検定
chi2, p = stats.chisquare(f_obs=obs_grp, f_exp=exp_grp)
print(f"Grouped χ²={chi2:.2f}, p-value={p:.3f}")
print("ポアソンに従う。" if p >= 0.05 else "ポアソンに従わない。")

labels = list(k[:-1]) + [f"{max_k}+"]
x = np.arange(len(labels))
plt.bar(x, obs_grp, width=0.6, alpha=0.6, label='Observed')
plt.plot(x, exp_grp, 'o-', lw=2, label='Expected')
plt.xticks(x, labels)
plt.xlabel('Total pronouns per song')
plt.ylabel('Frequency')
plt.title(f'χ² Goodness-of-Fit (unit={unit_time}曲)')
plt.legend()
plt.tight_layout()
plt.show()
