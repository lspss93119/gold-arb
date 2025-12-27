
import pandas as pd
import numpy as np
import glob
import statsmodels.api as sm

def load_and_process_data(pattern="market_data_*.csv"):
    """Load, merge, and resample data to 1-minute intervals."""
    files = glob.glob(pattern)
    if not files:
        print("No market data files found.")
        return None
    
    print(f"Loading {len(files)} files...")
    df_list = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_list: return None
    
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['readable_time'])
    full_df = full_df.sort_values('timestamp').set_index('timestamp')
    
    # Calculate Mid Prices
    full_df['paxg_mid'] = (full_df['paxg_best_bid'] + full_df['paxg_best_ask']) / 2
    full_df['xau_mid'] = (full_df['xau_best_bid'] + full_df['xau_best_ask']) / 2
    full_df['spread'] = full_df['paxg_mid'] - full_df['xau_mid']
    
    # Resample to 1 minute (OHLC for spread or simple mean)
    # Using last close for robustness in time-series
    df_1m = full_df['spread'].resample('1min').last().dropna()
    
    return df_1m

def calculate_half_life(spread_series):
    """
    Calculate Half-Life of Mean Reversion using Ornstein-Uhlenbeck process.
    d(x_t) = theta * (mu - x_t) * dt + sigma * dW_t
    Regression: x_t - x_{t-1} = a + b * x_{t-1} + error
    theta = -b
    Half-Life = -ln(2) / b
    """
    x_t = spread_series.values
    x_t_minus_1 = spread_series.shift(1).dropna().values
    x_t = x_t[1:] # Align length
    
    delta_x = x_t - x_t_minus_1
    
    # Regression
    # y = delta_x
    # x = x_t_minus_1
    # Add constant for 'a' (represents theta * mu)
    X = sm.add_constant(x_t_minus_1)
    
    model = sm.OLS(delta_x, X)
    results = model.fit()
    
    a, b = results.params
    
    # b is -theta * dt (dt = 1 for 1 unit step)
    # So reversion speed (theta) = -b
    # Half-life = ln(2) / theta = ln(2) / -b
    
    print("\n=== Regression Results ===")
    print(f"Slope (b): {b:.6f}")
    print(f"Intercept (a): {a:.6f}")
    print(f"R-squared: {results.rsquared:.4f}")
    
    if b >= 0:
        print("\n[WARNING] Slope is non-negative. Spread is NOT mean-reverting (it's trending or random walk).")
        return None
        
    half_life = -np.log(2) / b
    return half_life

def main():
    print("Processing data...")
    df_1m = load_and_process_data()
    
    if df_1m is None or len(df_1m) < 10:
        print("Not enough data to calculate half-life.")
        return

    print(f"\nData Points (1-min bars): {len(df_1m)}")
    print(f"Start: {df_1m.index[0]}")
    print(f"End:   {df_1m.index[-1]}")
    
    half_life = calculate_half_life(df_1m)
    
    if half_life:
        print(f"\n=== Result ===")
        print(f"Half-Life: {half_life:.2f} minutes")
        print(f"Interpretation: It takes approx {half_life:.1f} minutes for the spread to revert half-way to its mean.")
        
        # Recommendation
        print("\n=== Strategy Recommendation ===")
        print(f"Suggested Window Size: {int(half_life)} - {int(half_life * 3)} minutes")
        print(f"Currently in your bot: 11 Hours (660 mins)")
        if half_life * 5 < 660:
             print("-> Your current window (11h) might be too long relative to the actual reversion speed.")
        else:
             print("-> Your current window seems appropriate.")

if __name__ == "__main__":
    main()
