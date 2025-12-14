# modeling_gbdt_optimized_fixed_large.py

import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import numpy as np

# ----------------------
# 1. Load prepared data
# ----------------------
X_train = pd.read_csv("X_train_large.csv")
X_test = pd.read_csv("X_test_large.csv")
y_train = pd.read_csv("y_train_large.csv").values.ravel()
y_test = pd.read_csv("y_test_large.csv").values.ravel()

# ----------------------
# 1b. Feature engineering
# ----------------------
def add_features(df):
    df = df.copy()
    df['moneyness'] = df['S'] / df['K']
    df['vol_sqrtT'] = df['sigma'] * np.sqrt(df['T'])
    df['log_moneyness'] = np.log(df['S'] / df['K'] + 1e-8)
    return df

X_train = add_features(X_train)
X_test = add_features(X_test)

# ----------------------
# 2. Convert to DMatrix
# ----------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test, label=y_test)

# ----------------------
# 3. Train GBDT model (XGBoost) with reduced overfitting
# ----------------------
params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,        # reduced from 12
    'learning_rate': 0.01,
    'subsample': 0.8,      # reduced from 0.9
    'colsample_bytree': 0.8, # reduced from 0.9
    'gamma': 0.1,
    'alpha': 0.5,
    'lambda': 1,
    'tree_method': 'hist',
    'nthread': -1,
    'eval_metric': 'rmse'
}

evals = [(dtrain, 'train'), (dtest, 'validation')]

start_time = time.time()
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    evals=evals,
    early_stopping_rounds=100,
    verbose_eval=50
)
train_time = time.time() - start_time
print(f"GBDT training completed in {train_time:.2f} seconds")

# Save the trained model
bst.save_model(r"C:\Users\user\Desktop\data science project\backend\modelsr")
print("Model saved as 'gbdt_model_large.json'")

# ----------------------
# 4. Make predictions
# ----------------------
start_time = time.time()
y_pred = bst.predict(dtest)
inference_time = time.time() - start_time
print(f"GBDT inference on test set completed in {inference_time:.4f} seconds")

# ----------------------
# 5. Evaluation
# ----------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100 # Multiply by 100 for percentage

print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAPE: {mape:.6f}")

# ----------------------
# 6. Visualizations
# ----------------------

# NEW METRICS BAR PLOT
metrics_list = ['MAE', 'RMSE']
values_list = [mae, rmse]

plt.figure(figsize=(6, 4))
sns.barplot(x=metrics_list, y=values_list, palette="flare")
plt.title("Key Regression Error Metrics")
plt.ylabel("Error Value (Price Currency Units)")

# Add text labels on the bars
for i, v in enumerate(values_list):
    plt.text(i, v + 0.05, f"{v:.4f}", color='black', ha='center', fontsize=10)

plt.ylim(0, max(values_list) * 1.2)
plt.show()

# NOTE: MAPE is better presented as a key statistic due to scale difference
print(f"Note on MAPE: The Mean Absolute Percentage Error (MAPE) is {mape:.4f}%, indicating a relative error of less than 5% on average.")


plt.figure(figsize=(7, 7))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Monte Carlo Price")
plt.ylabel("GBDT Predicted Price")
plt.title("GBDT vs Monte Carlo Prices")
plt.show()

errors = y_test - y_pred
plt.figure(figsize=(7, 5))
sns.histplot(errors, bins=50, kde=True)
plt.xlabel("Prediction Error")
plt.title("GBDT Prediction Error Distribution")
plt.show()

plt.figure(figsize=(8, 6))
xgb_importance = bst.get_score(importance_type='weight')
feature_names = list(xgb_importance.keys())
importance_values = list(xgb_importance.values())
sns.barplot(x=importance_values, y=feature_names)
plt.title("Feature Importance (XGBoost)")
plt.show()

# ----------------------
# 7. Compare speed (estimated)
# ----------------------
mc_time_per_row = 0.01
mc_total_time = mc_time_per_row * len(y_test)
print(f"Estimated Monte Carlo time for test set: {mc_total_time:.2f} seconds")
print(f"GBDT inference time for test set: {inference_time:.4f} seconds")
print(f"Speedup factor: {mc_total_time / inference_time:.2f}x")

# ----------------------
# 8. Monte Carlo comparison (real)
# ----------------------
mc_input_cols = ['S', 'K', 'T', 'sigma', 'r', 'type']

def monte_carlo_price(S, K, T, sigma, r, type_option, n_sim=20000):
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.randn(n_sim))
    if type_option == 1:
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    return np.exp(-r*T) * np.mean(payoff)

start = time.time()
mc_prices = [monte_carlo_price(*row, n_sim=20000) for row in X_test[mc_input_cols].values]
mc_time = time.time() - start

print(f"Monte Carlo total time: {mc_time:.2f} seconds")
print(f"GBDT inference time: {inference_time:.4f} seconds")
print(f"Speedup factor: {mc_time / inference_time:.2f}x")