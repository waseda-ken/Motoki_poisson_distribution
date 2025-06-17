import pandas as pd
import numpy as np
import statsmodels.api as sm

unit_time = 0.5

df = pd.read_csv('Mrs_total_pronouns.csv')
y = df['total'].values
n = len(y)
X = np.ones((n, 1))

# オフセット（log(exposure)）を設定
exposure = np.full(n, unit_time)
offset = np.log(exposure)

# Poisson GLM
poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
print("[AIC] Poisson AIC:", poisson_model.aic)

# Negative Binomial GLM
nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit()
print("[AIC] NegBinom AIC:", nb_model.aic)

best = "NegBinom" if nb_model.aic < poisson_model.aic else "Poisson"
print(f"{best} がより適合度良好です（単位時間={unit_time}曲）。")
