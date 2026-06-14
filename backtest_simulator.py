import pandas as pd
import numpy as np
import xgboost as xgb
from feature_factory import FeatureFactory
import warnings

warnings.filterwarnings('ignore')

print("--- AI RISK ENGINE & PORTFOLIO BACKTEST ---")
print("[*] 1. Initializing AI and Rebuilding Predictions...")

# 1. Quick Data Load & Train (Using the exact same logic as your engine)
factory = FeatureFactory()
features_df = factory.process_stock(stock_id=0)
targets_df = pd.read_csv("train.csv")
master_data = pd.merge(features_df, targets_df, on=['stock_id', 'time_id'], how='inner').dropna()

X = master_data[['realized_volatility', 'mean_spread']]
y = master_data['target']

split_index = int(len(master_data) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("[✓] AI Engine Ready. Commencing Backtest...\n")

# --- THE BACKTEST SIMULATION ---
# We simulate a market maker capturing the spread, but taking losses when volatility spikes.

capital = 1000000 # £1,000,000 starting capital
target_volatility = 0.002 # We want a safe, flat risk profile

portfolio_baseline = [capital]
portfolio_ai = [capital]

print(f"[*] Starting Capital: £{capital:,}")
print(f"[*] Simulating {len(X_test)} trading windows...\n")

# Loop through the future timeline
np.random.seed(42) # Lock random seed so your results are perfectly reproducible
for i in range(len(X_test)):
    actual_vol = y_test.iloc[i]
    predicted_vol = predictions[i]
    spread_edge = X_test['mean_spread'].iloc[i]
    
    # 1. The "Dumb" Strategy: Always trade £1,000,000.
    baseline_position = capital
    
    # 2. The AI Risk Strategy: Volatility Targeting
    # If predicted vol > target vol, this shrinks our position.
    # If predicted vol < target vol, it increases it safely.
    ai_position = capital * (target_volatility / predicted_vol)
    
    # Cap maximum leverage so we don't borrow crazy amounts of money
    ai_position = min(ai_position, capital * 2) 

    # Simulate PnL: We make money on the spread, but take a hit proportional to actual volatility shocks
    market_shock = np.random.normal(0, actual_vol)
    
    baseline_pnl = (baseline_position * spread_edge) - (baseline_position * abs(market_shock) * 0.1)
    ai_pnl = (ai_position * spread_edge) - (ai_position * abs(market_shock) * 0.1)
    
    portfolio_baseline.append(portfolio_baseline[-1] + baseline_pnl)
    portfolio_ai.append(portfolio_ai[-1] + ai_pnl)

# --- ANALYTICS ---
base_return = ((portfolio_baseline[-1] - capital) / capital) * 100
ai_return = ((portfolio_ai[-1] - capital) / capital) * 100

base_std = np.std(np.diff(portfolio_baseline))
ai_std = np.std(np.diff(portfolio_ai))

# Sharpe Ratio = Return per unit of Risk (Annualized for high-frequency)
base_sharpe = np.mean(np.diff(portfolio_baseline)) / base_std * np.sqrt(252 * 78) 
ai_sharpe = np.mean(np.diff(portfolio_ai)) / ai_std * np.sqrt(252 * 78)

print("==== FINAL INSTITUTIONAL TEARSHEET ====")
print("1. DUMB BASELINE (Constant Position Size)")
print(f"   Final Equity: £{portfolio_baseline[-1]:,.2f}")
print(f"   Total Return: {base_return:.2f}%")
print(f"   Sharpe Ratio: {base_sharpe:.2f}")

print("\n2. AI RISK ENGINE (Dynamic Sizing)")
print(f"   Final Equity: £{portfolio_ai[-1]:,.2f}")
print(f"   Total Return: {ai_return:.2f}%")
print(f"   Sharpe Ratio: {ai_sharpe:.2f}")
print("=======================================")

if ai_sharpe > base_sharpe:
    print("\n[✓] CONCLUSION: The AI successfully protected the portfolio, crushing the baseline Sharpe Ratio.")