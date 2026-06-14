import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from feature_factory import FeatureFactory

print("--- KAGGLE AI TRAINING ENGINE ---")

# 1. Generate Features & Load Targets
print("[*] 1. Loading and Merging Data...")
factory = FeatureFactory()
features_df = factory.process_stock(stock_id=0)
targets_df = pd.read_csv("train.csv")
master_data = pd.merge(features_df, targets_df, on=['stock_id', 'time_id'], how='inner').dropna()

# 2. Prepare the AI Matrix (X) and the Answers (y)
print("[*] 2. Prepping Machine Learning Matrix...")
# We only give the AI the math clues. We hide time_id and stock_id so it doesn't cheat.
X = master_data[['realized_volatility', 'mean_spread']]
y = master_data['target']

# 3. Time-Series Split (80% Train, 20% Test)
# We train the AI on the past, and test it on the future.
split_index = int(len(master_data) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"[*]    -> Training on {len(X_train)} past rows.")
print(f"[*]    -> Testing on {len(X_test)} future rows.")

# 4. Train the Baseline XGBoost AI
print("[*] 3. Training XGBoost Algorithm...")
model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=4, 
    random_state=42
)
model.fit(X_train, y_train)

# 5. Grade the AI
print("[*] 4. Grading AI Performance...")
predictions = model.predict(X_test)

# Calculate RMSE (Root Mean Squared Error) - The standard Kaggle scoring metric
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"\n[✓] SUCCESS! Model RMSE Error: {rmse:.6f}")

# Show a real-world snapshot of how well it predicted
results = pd.DataFrame({
    'Actual Answer': y_test.values[:5],
    'AI Prediction': predictions[:5]
})
print("\n[Sneak Peek: Actual vs Predicted Volatility]")
print(results)