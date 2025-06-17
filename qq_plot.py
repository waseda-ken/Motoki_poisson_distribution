import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 分割後の各行を 0.5曲として扱う
unit_time = 1.0

df = pd.read_csv('Mrs_total_pronouns.csv')
counts = np.sort(df['total'].values)
n = len(counts)

# 元データは「1曲あたり」の発話数 → 単位時間0.5曲あたりの期待値
lambda_hat_full = counts.mean()
lambda_hat = lambda_hat_full * unit_time

print(f"[Q-Q] サンプル数={n}, 単位時間={unit_time}曲あたり λ̂={lambda_hat:.3f}")

# 理論分位点（(i-0.5)/n のパーセント点）
probs = (np.arange(1, n+1) - 0.5) / n
theoretical_q = stats.poisson.ppf(probs, mu=lambda_hat)

if np.any(np.isnan(theoretical_q)):
    print("警告: 理論分位点に NaN が含まれています。")
    theoretical_q = theoretical_q[~np.isnan(theoretical_q)]
else:
    print("理論分位点はすべて有効です。")

# 1) 回帰でフィット度
slope, intercept, r_value, p_value, std_err = stats.linregress(theoretical_q, counts)
print(f"slope = {slope:.3f}, intercept = {intercept:.3f}, R² = {r_value**2:.3f}")
print("R²は良好です。" if r_value**2 >= 0.95 else "注意: R²が0.95未満です。")

# 2) 残差誤差
residuals = counts - theoretical_q
mae = np.mean(np.abs(residuals))
max_err = np.max(np.abs(residuals))
print(f"Mean absolute error = {mae:.3f}")
print(f"Max absolute error  = {max_err:.3f}")
print("最大誤差は許容範囲内です。" if max_err <= 0.1 else "注意: 最大誤差が0.1超え。")

plt.scatter(theoretical_q, counts)
lims = [min(theoretical_q.min(), counts.min()),
        max(theoretical_q.max(), counts.max())]
plt.plot(lims, lims, 'r--', label='y=x')
plt.xlabel(f'Theoretical Quantiles (Poisson, μ={lambda_hat:.2f})')
plt.ylabel('Sample Quantiles')
plt.title('Q-Q Plot (0.5-song unit)')
plt.legend()
plt.tight_layout()
plt.show()
