import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('Mrs_total_pronouns.csv')
counts = df['total'].values
n = len(counts)
lambda_hat = counts.mean()

print(f"サンプル数={n}, 推定 λ={lambda_hat:.3f}")

# ヒストグラム
max_k = counts.max()
bins = np.arange(0, max_k+2) - 0.5
obs_freq, _ = np.histogram(counts, bins=bins)

# 理論度数
k = np.arange(0, max_k+1)
pmf = stats.poisson.pmf(k, mu=lambda_hat)
exp_freq = pmf * n

# 「尾部」にまとめる：最後のビン k>=max_k
tail_prob = 1 - stats.poisson.cdf(max_k, mu=lambda_hat)
exp_tail = tail_prob * n

# 尾部をまとめた観測・期待度数配列
obs_grp = np.append(obs_freq[:-1], obs_freq[-1])
exp_grp = np.append(exp_freq[:-1], exp_freq[-1] + exp_tail)

# χ² 検定
chi2, p = stats.chisquare(f_obs=obs_grp, f_exp=exp_grp)
print(f"Grouped χ²={chi2:.2f}, p-value={p:.3f}")
if p < 0.05:
    print("帰無仮説を棄却する。ポアソン分布に従わない。")
else:
    print("帰無仮説を棄却できない。ポアソン分布に従う。")

labels = list(k[:-1]) + [f"{max_k}+"]
x = np.arange(len(labels))
plt.bar(x, obs_grp, width=0.6, label='Observed counts', alpha=0.6)
plt.plot(x, exp_grp, 'o-', label='Expected counts (Poisson + tail)', lw=2)
plt.xticks(x, labels)
plt.xlabel('Total pronouns per song')
plt.ylabel('Frequency')
plt.title('Grouped χ² Goodness-of-Fit: Observed vs Poisson')
plt.legend()
plt.tight_layout()
plt.show()