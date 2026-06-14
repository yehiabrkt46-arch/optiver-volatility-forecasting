import pandas as pd
import numpy as np
import os

class FeatureFactory:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir

    def process_stock(self, stock_id: int) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, "book_train.parquet", f"stock_id={stock_id}")
        df = pd.read_parquet(file_path)

        # 1. Calculate WAP
        df['wap'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])

        # 2. Calculate Instantaneous Log Returns
        df['log_return'] = np.log(df['wap']).groupby(df['time_id']).diff()

        # 3. Calculate Spread BEFORE grouping (Vectorized C-speed)
        df['spread'] = (df['ask_price1'] - df['bid_price1']) / df['wap']

        # 4. Compress into Features
        features = df.groupby('time_id').agg(
            realized_volatility=('log_return', lambda x: np.sqrt((x**2).sum())),
            mean_spread=('spread', 'mean')
        ).reset_index()

        features['stock_id'] = stock_id
        return features

if __name__ == "__main__":
    print("--- HIGH-FREQUENCY FEATURE FACTORY ---")
    factory = FeatureFactory()
    
    print("[*] Processing Stock 0 Limit Order Book...")
    features_df = factory.process_stock(stock_id=0)
    
    print(f"[✓] Success! Compressed raw ticks down to {len(features_df)} clean feature rows.")
    print("\n[Machine Learning Feature Matrix Snapshot]")
    print(features_df.head())