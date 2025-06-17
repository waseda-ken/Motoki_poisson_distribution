import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm

# データ読み込み
df = pd.read_csv('Mrs_total_pronouns.csv')
y = df['total'].values
n = len(y)
X = np.ones((n,1)) 

# Poisson GLM
poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
print("[AIC] Poisson AIC:", poisson_model.aic)

# Negative Binomial GLM
nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
print("[AIC] NegBinom AIC:", nb_model.aic)

if nb_model.aic < poisson_model.aic:
    print("NegBinomがPoissonよりも適合度が良い。")
else:
    print("PoissonがNegBinomよりも適合度が良い。")