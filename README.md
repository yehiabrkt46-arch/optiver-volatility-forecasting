# optiver-volatility-forecasting
High Frequency Volatility Forecasting & Dynamic Risk Engine

Overview
A machine learning driven risk engine built for high frequency market making. Using the Kaggle Optiver Realized Volatility dataset, this pipeline ingests microsecond level Limit Order Book data to forecast short term volatility regimes. The predictive alpha is used to drive a dynamic position sizing algorithm, aggressively scaling down leverage ahead of structural market shocks to protect capital and optimize risk adjusted returns.

System Architecture

1. Microstructure Data Pipeline
* Ingests highly compressed parquet files containing millions of raw bid and ask updates.
* Utilizes fully vectorized pandas and NumPy operations to bypass Python iteration overhead.
* Calculates true Weighted Average Price and instantaneous order book spread, compressing sub-second tick data into fixed time bucket feature matrices.

2. Alpha Generation Engine
* Implements an XGBoost regression model to map current liquidity spreads and historical variance to future realized volatility targets.
* Validated using a strict, un-shuffled time series split to eliminate forward looking data leakage.
* Optimized for Root Mean Squared Error to accurately capture the magnitude of predicted price swings.

3. Dynamic Portfolio Allocation
* Simulates a market making portfolio acting on the generated alpha over an out of sample forward timeline.
* Implements Volatility Targeting: the system scales position sizes inversely to predicted volatility. 
* Limits exposure during forecasted high variance windows while maximizing spread capture during low variance regimes.

Out of Sample Performance

| Metric | Static Baseline | Dynamic Risk Engine |

| **Final Equity** | £1,539,412 | **£1,249,819** |
| **Total Return** | 53.94% | **24.98%** |
| **Sharpe Ratio** | 124.19 | **238.72** |

Strategy Conclusion: While a static, maximum capacity position sizing strategy yielded a higher nominal PnL, it incurred symmetrical, unhedged risk during periods of market stress. By dynamically adjusting leverage based on the XGBoost volatility forecast, the Risk Engine successfully avoided market shocks, effectively doubling the Sharpe Ratio and providing institutional grade capital protection.

Technology Stack
* Language: Python 3.10+
* Data Engineering: Pandas, NumPy, PyArrow, FastParquet
* Machine Learning: XGBoost, Scikit Learn
