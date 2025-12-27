
import pandas as pd
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from decimal import Decimal

# Optimization Parameters
WINDOW_MIN = 1800  # 0.5 hours
WINDOW_MAX = 86400 # 24 hours
WINDOW_STEP = 1800
Z_MIN = 0.5
Z_MAX = 3.0
Z_STEP = 0.1

def load_data(pattern="data/market_data_*.csv"):
    """Load and merge all market data CSVs."""
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
    # Sort by timestamp
    full_df = full_df.sort_values('timestamp').reset_index(drop=True)
    
    # Filter Invalid Data (Market Closed / Stagnant)
    # 1. Flag-based filter (primary)
    if 'is_market_open' in full_df.columns:
        # Fill missing values (for old CSVs) with True (assume open, let Timestamp filter handle it)
        full_df['is_market_open'] = full_df['is_market_open'].fillna(True)
        # Handle string booleans and possible 'nan' strings
        full_df['is_market_open'] = full_df['is_market_open'].astype(str).map({'True': True, 'False': False, '1': True, '0': False, '1.0': True, '0.0': False, 'nan': True})
        
        # Filter: Keep only Open rows
        full_df = full_df[full_df['is_market_open'] == True]
    
    # 2. Timestamp-based filter (secondary, enforces rules on old data)
    # Market Open: Sun 23:00 UTC to Fri 22:00 UTC
    # Filter out: Fri 22:00 <= Time < Sun 23:00
    
    # Convert to UTC datetime
    full_df['dt_utc'] = pd.to_datetime(full_df['timestamp'], unit='s', utc=True)
    
    # Logic: Keep if NOT (Fri >= 22 or Sat or Sun < 23)
    # Weekday: Mon=0 ... Fri=4, Sat=5, Sun=6
    
    condition_fri_closed = (full_df['dt_utc'].dt.weekday == 4) & (full_df['dt_utc'].dt.hour >= 22)
    condition_sat_closed = (full_df['dt_utc'].dt.weekday == 5)
    condition_sun_closed = (full_df['dt_utc'].dt.weekday == 6) & (full_df['dt_utc'].dt.hour < 23)
    
    # Filter Holidays (Dec 25, Jan 1)
    condition_holiday = (
        ((full_df['dt_utc'].dt.month == 12) & (full_df['dt_utc'].dt.day == 25)) |
        ((full_df['dt_utc'].dt.month == 1) & (full_df['dt_utc'].dt.day == 1))
    )
    
    full_df = full_df[
        ~(condition_fri_closed | condition_sat_closed | condition_sun_closed | condition_holiday)
    ]
    
    full_df = full_df.reset_index(drop=True)
    full_df = full_df.drop(columns=['dt_utc']) # Clean up

    # Pre-calculate Mid Prices for Z-Score
    full_df['paxg_mid'] = (full_df['paxg_best_bid'] + full_df['paxg_best_ask']) / 2
    full_df['xau_mid'] = (full_df['xau_best_bid'] + full_df['xau_best_ask']) / 2
    
    # Spread = PAXG - XAU
    full_df['spread_diff'] = full_df['paxg_mid'] - full_df['xau_mid']
    
    print(f"Loaded {len(full_df)} valid rows.")
    return full_df

def calculate_slippage_cost(row, amount_usdc, side):
    """
    Calculate slippage cost based on orderbook depth.
    amount_usdc: Trade size in USDC
    side: 'buy_spread' (Buy PAXG, Sell XAU) or 'sell_spread' (Sell PAXG, Buy XAU)
    """
    try:
        # PAXG Depth Key
        paxg_depth_key = 'paxg_asks_depth' if side == 'buy_spread' else 'paxg_bids_depth' 
        xau_depth_key = 'xau_bids_depth' if side == 'buy_spread' else 'xau_asks_depth'    
        
        if side == 'sell_spread':
            paxg_depth_key = 'paxg_bids_depth' 
            xau_depth_key = 'xau_asks_depth'   
            
        paxg_depth = json.loads(row[paxg_depth_key])
        xau_depth = json.loads(row[xau_depth_key])
        
        def get_avg_price(depth, target_value):
            filled_val = 0.0
            total_qty = 0.0
            weighted_sum = 0.0
            
            for level in depth:
                p = float(level['p'])
                s = float(level['s'])
                
                level_val = p * s
                remaining_val = target_value - filled_val
                
                take_val = min(level_val, remaining_val)
                take_qty = take_val / p
                
                weighted_sum += take_val
                total_qty += take_qty
                filled_val += take_val
                
                if filled_val >= target_value * 0.999:
                    break
            
            if filled_val == 0: return None
            return weighted_sum / total_qty if total_qty > 0 else None

        paxg_avg = get_avg_price(paxg_depth, amount_usdc)
        xau_avg = get_avg_price(xau_depth, amount_usdc)
        
        if paxg_avg is None or xau_avg is None:
            return float('inf') 
            
        paxg_mid = (row['paxg_best_bid'] + row['paxg_best_ask']) / 2
        xau_mid = (row['xau_best_bid'] + row['xau_best_ask']) / 2
        
        slip_paxg = abs(paxg_avg - paxg_mid)
        slip_xau = abs(xau_avg - xau_mid)
        
        qty_paxg = amount_usdc / paxg_mid
        qty_xau = amount_usdc / xau_mid
        
        total_cost = (slip_paxg * qty_paxg) + (slip_xau * qty_xau)
        return total_cost

    except Exception:
        return 0.0 

def backtest(df, window_seconds, z_entry, trade_size_usdc=1000):
    """
    Backtest strategy for specific parameters.
    """
    df_test = df.copy()
    # Simple rolling mean/std calculation
    rolling = df_test['spread_diff'].rolling(window=int(window_seconds), min_periods=max(1, int(window_seconds/2)))
    
    # Note: Rolling on integer window assumes continuous data. 
    # Since we filter weekends, there might be jumps. 
    # Ideally should use time-based rolling, but integer is faster for estimation.
    
    df_test['mean'] = rolling.mean()
    df_test['std'] = rolling.std()
    
    # Z-Score
    df_test['z_score'] = (df_test['spread_diff'] - df_test['mean']) / df_test['std']
    
    df_test = df_test.dropna(subset=['z_score'])
    
    position = 0 
    pnl = 0.0
    entry_price = 0.0
    max_pnl = 0.0
    dd = 0.0
    
    records = df_test.to_dict('records')
    
    for row in records:
        z = row['z_score']
        spread = row['spread_diff']
        
        if position == 0:
            if z > z_entry:
                # Open Short Spread
                cost = calculate_slippage_cost(row, trade_size_usdc, 'sell_spread')
                position = -1
                entry_price = spread
                pnl -= cost 
                
            elif z < -z_entry:
                # Open Long Spread
                cost = calculate_slippage_cost(row, trade_size_usdc, 'buy_spread')
                position = 1
                entry_price = spread
                pnl -= cost 
                
        elif position == -1:
            if z <= 0: 
                # Close Short
                cost = calculate_slippage_cost(row, trade_size_usdc, 'buy_spread')
                qty = trade_size_usdc / 2700.0 
                profit = (entry_price - spread) * qty 
                pnl += (profit - cost)
                position = 0
                
        elif position == 1:
            if z >= 0:
                # Close Long
                cost = calculate_slippage_cost(row, trade_size_usdc, 'sell_spread')
                qty = trade_size_usdc / 2700.0
                profit = (spread - entry_price) * qty
                pnl += (profit - cost)
                position = 0
                
        if pnl > max_pnl: max_pnl = pnl
        if (pnl - max_pnl) < dd: dd = (pnl - max_pnl)
        
    return pnl, dd

def optimize():
    df = load_data()
    if df is None: return
    
    windows = range(WINDOW_MIN, WINDOW_MAX + 1, WINDOW_STEP)
    z_scores = np.arange(Z_MIN, Z_MAX + 0.01, Z_STEP)
    
    results = []
    
    print(f"Starting Grid Search ({len(windows)} x {len(z_scores)} iterations)...")
    
    for w, z in product(windows, z_scores):
        pnl, dd = backtest(df, w, z)
        
        sharpe = pnl / abs(dd) if dd != 0 else 0
        if pnl == 0: sharpe = 0
        
        results.append({
            'window': w,
            'z_score': z,
            'pnl': pnl,
            'max_dd': dd,
            'sharpe': sharpe
        })
        # print(f"W: {w}, Z: {z:.1f} -> PnL: {pnl:.2f}, Sharpe: {sharpe:.2f}")
        
    res_df = pd.DataFrame(results)
    
    # 1. Heatmap
    pivot = res_df.pivot(index='window', columns='z_score', values='sharpe')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=False, cmap='RdYlGn')
    plt.title("Sharpe Ratio Heatmap")
    plt.xlabel("Z-Score Threshold")
    plt.ylabel("Window Size (sec)")
    plt.savefig("data/optimizer_heatmap.png")
    print("\nSaved heatmap to data/optimizer_heatmap.png")
    
    # 2. Find Parameter Plateau (Robustness)
    # We look for the cell with the highest average Sharpe among its neighbors (3x3 grid)
    print("\nCalculating Robust Plateau...")
    
    best_plateau_score = -999
    best_config = None
    
    # Helper to clean grid values
    grid_values = pivot.fillna(0).values 
    rows = pivot.index
    cols = pivot.columns
    
    for r in range(1, len(rows)-1):
        for c in range(1, len(cols)-1):
            # 3x3 kernel
            kernel = grid_values[r-1:r+2, c-1:c+2]
            avg_score = np.mean(kernel)
            
            if avg_score > best_plateau_score:
                best_plateau_score = avg_score
                best_config = (rows[r], cols[c])
                
    # Fallback to absolute max if no plateau found (e.g. grid too small)
    best_single = res_df.sort_values('sharpe', ascending=False).iloc[0]
    
    print("\n=== OPTIMIZATION RESULT ===")
    
    if best_config:
        w_p, z_p = best_config
        # Fetch detailed stats for the center of the plateau
        plateau_stats = res_df[(res_df['window'] == w_p) & (np.isclose(res_df['z_score'], z_p))].iloc[0]
        
        print(f"--- Recommended Robust Parameters (Plateau) ---")
        print(f"Window Size : {int(w_p)}s ({w_p/3600:.1f}h)")
        print(f"Z-Score     : {z_p:.1f}")
        print(f"Avg Sharpe (Neighbors): {best_plateau_score:.4f}")
        print(f"Net Profit  : ${plateau_stats['pnl']:.2f}")
    else:
        print("No robust plateau found (grid too small or flat). Using single best.")
        
    print(f"\n--- Absolute Best Single Run ---")
    print(f"Window: {int(best_single['window'])}s, Z: {best_single['z_score']:.1f}")
    print(f"Profit: ${best_single['pnl']:.2f}, Sharpe: {best_single['sharpe']:.4f}")

if __name__ == "__main__":
    optimize()
