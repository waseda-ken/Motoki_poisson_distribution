import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv('Mrs_total_pronouns.csv')
counts = np.sort(df['total'].values)
n = len(counts)
lambda_hat = counts.mean()

# 理論分位点を計算（(i-0.5)/n のパーセント点）
probs = (np.arange(1, n+1) - 0.5) / n
theoretical_q = stats.poisson.ppf(probs, mu=lambda_hat)
if np.any(np.isnan(theoretical_q)):
    print("警告: 理論分位点に NaN が含まれています。入力データを確認してください。")
    theoretical_q = theoretical_q[~np.isnan(theoretical_q)]
else:
    print("理論分位点はすべて有効です。")

# --- 定量的評価 ---

# 1) 線形回帰でフィット度を見る
slope, intercept, r_value, p_value, std_err = stats.linregress(theoretical_q, counts)
print(f"slope = {slope:.3f}, intercept = {intercept:.3f}, R² = {r_value**2:.3f}")
if r_value**2 < 0.95:
    print("注意: R²が0.95未満です。データの適合度に問題がある可能性があります。")  
else:
    print("R²は許容範囲内です。データの適合度は良好です。")

# 2) 残差（counts - theoretical_q）による誤差指標
residuals = counts - theoretical_q
mae = np.mean(np.abs(residuals))
max_err = np.max(np.abs(residuals))
print(f"Mean absolute error = {mae:.3f}")
print(f"Max absolute error  = {max_err:.3f}")
if max_err > 0.1:
    print("注意: 最大誤差が0.1を超えています。データの適合度に問題がある可能性があります。")
else:
    print("最大誤差は許容範囲内です。データの適合度は良好です。")

# Q–Qプロット
plt.scatter(theoretical_q, counts)
lims = [min(theoretical_q.min(), counts.min()),
        max(theoretical_q.max(), counts.max())]
plt.plot(lims, lims, 'r--', label='y=x')
plt.xlabel('Theoretical Quantiles (Poisson)')
plt.ylabel('Sample Quantiles')
plt.title('Q-Q Plot')
plt.legend()
plt.tight_layout()
plt.show()
