import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("options_mc_dataset_large.csv")

# 2. Remove invalid Monte Carlo prices (zeros or negative)
df = df[df['mc_price'] > 0]

# 3. Select features and target
X = df[['S', 'K', 'T', 'sigma', 'r', 'type']].copy()  # .copy() avoids SettingWithCopyWarning
y = df['mc_price']

# 4. Optional: Feature engineering
# Add 'moneyness' feature
X['moneyness'] = X['S'] / X['K']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Optional: Save the prepared datasets for later use
X_train.to_csv("X_train_large.csv", index=False)
X_test.to_csv("X_test_large.csv", index=False)
y_train.to_csv("y_train_large.csv", index=False)
y_test.to_csv("y_test_large.csv", index=False)

print("Data preparation completed successfully!")
print(f"Training set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")