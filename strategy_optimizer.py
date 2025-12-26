
import pandas as pd
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from decimal import Decimal

# Optimization Parameters
WINDOW_MIN = 1800  # 30 mins
WINDOW_MAX = 86400 # 24 hours
WINDOW_STEP = 1800
Z_MIN = 0.5
Z_MAX = 3.0
Z_STEP = 0.1

def load_data(pattern="market_data_*.csv"):
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
    full_df = full_df.sort_values('timestamp').reset_index(drop=True)
    
    # Pre-calculate Mid Prices for Z-Score
    # Mid = (Bid + Ask) / 2
    full_df['paxg_mid'] = (full_df['paxg_best_bid'] + full_df['paxg_best_ask']) / 2
    full_df['xau_mid'] = (full_df['xau_best_bid'] + full_df['xau_best_ask']) / 2
    
    # Spread = PAXG - XAU
    full_df['spread_diff'] = full_df['paxg_mid'] - full_df['xau_mid']
    
    print(f"Loaded {len(full_df)} rows.")
    return full_df

def calculate_slippage_cost(row, amount_usdc, side):
    """
    Calculate slippage cost based on orderbook depth.
    amount_usdc: Trade size in USDC
    side: 'buy_spread' (Buy PAXG, Sell XAU) or 'sell_spread' (Sell PAXG, Buy XAU)
    
    Returns: Cost in Spread Points (or USDC? Strategy uses Spread Points for PnL usually, but let's stick to USDC cost).
    Actually, simpler: Calculate Average Execution Price for both legs and compare to Mid Price.
    Cost = |Exec - Mid| * Size.
    """
    try:
        # Parse JSON depth
        # PAXG
        paxg_depth_key = 'paxg_asks_depth' if side == 'buy_spread' else 'paxg_bids_depth' # Buy PAXG if buying spread
        xau_depth_key = 'xau_bids_depth' if side == 'buy_spread' else 'xau_asks_depth'    # Sell XAU if buying spread
        
        # Reverse logic for Sell Spread
        if side == 'sell_spread':
            paxg_depth_key = 'paxg_bids_depth' # Sell PAXG
            xau_depth_key = 'xau_asks_depth'   # Buy XAU
            
        paxg_depth = json.loads(row[paxg_depth_key])
        xau_depth = json.loads(row[xau_depth_key])
        
        def get_avg_price(depth, target_value):
            filled_val = 0.0
            total_qty = 0.0
            weighted_sum = 0.0
            
            for level in depth:
                p = float(level['p'])
                s = float(level['s'])
                
                # Check value capacity of this level
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

        # Price Impact
        paxg_avg = get_avg_price(paxg_depth, amount_usdc)
        xau_avg = get_avg_price(xau_depth, amount_usdc)
        
        if paxg_avg is None or xau_avg is None:
            return float('inf') # Infinite cost if no liquidity
            
        # Cost relative to Mid Price used in Z-Score
        # Spread PnL is based on Mid. Trade is based on Exec.
        # Slippage = |Exec - Mid|
        paxg_mid = (row['paxg_best_bid'] + row['paxg_best_ask']) / 2
        xau_mid = (row['xau_best_bid'] + row['xau_best_ask']) / 2
        
        slip_paxg = abs(paxg_avg - paxg_mid)
        slip_xau = abs(xau_avg - xau_mid)
        
        # Total Slippage per unit? Or total value?
        # Let's return Total Slippage Cost in USDC for this trade size
        # Approx qty = amount_usdc / price
        qty_paxg = amount_usdc / paxg_mid
        qty_xau = amount_usdc / xau_mid
        
        total_cost = (slip_paxg * qty_paxg) + (slip_xau * qty_xau)
        return total_cost

    except Exception:
        return 0.0 # Fallback

def backtest(df, window_seconds, z_entry, trade_size_usdc=1000):
    """
    Vectorized backtest is hard with sliding window + state. 
    Use Iteration with pre-calc rolling stats.
    """
    # 1. Calc Rolling Stats
    # Window in rows? We assume 1s frequency roughly.
    # If gaps exist, rolling('Xs') is better but requires index time.
    
    df_test = df.copy()
    df_test.set_index(pd.to_datetime(df_test['readable_time']), inplace=True)
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_seconds) # Wait, we need looking BACK.
    # rolling(str) is backward looking.
    
    rolling = df_test['spread_diff'].rolling(window=f'{window_seconds}s', min_periods=int(window_seconds/2))
    
    df_test['mean'] = rolling.mean()
    df_test['std'] = rolling.std()
    
    # Z-Score
    df_test['z_score'] = (df_test['spread_diff'] - df_test['mean']) / df_test['std']
    
    # Simulation
    position = 0 # 0, 1 (Long Spread), -1 (Short Spread)
    pnl = 0.0
    entry_price = 0.0
    max_pnl = 0.0
    dd = 0.0
    
    # We iterate nicely
    # Note: Vectorization is faster but logic is complex.
    # For optimization, we need speed.
    
    # Let's clean NA
    df_test = df_test.dropna(subset=['z_score'])
    
    # Extract arrays for speed
    z_vals = df_test['z_score'].values
    spread_vals = df_test['spread_diff'].values
    # timestamps = df_test['timestamp'].values # For duration
    # depths... we need raw rows for slippage.
    # Accessing dataframe inside loop is slow.
    # Pre-calc slippage? Slippage depends on side. 
    # Approx: Slippage is constant % or func of volatility? 
    # For accurate "Real Slippage", we must look up depth.
    # To optimize speed: Assume average slippage or look up only on trade.
    
    records = df_test.to_dict('records')
    
    trade_log = []
    
    for row, z, spread in zip(records, z_vals, spread_vals):
        # Entry/Exit Logic
        # Exit Z is usually mean reversion (0) or small buffer. Fixed at 0.0 or 0.2?
        # Prompt only asks to optimize Z-Score (Entry). Exit logic implicit?
        # Usually Exit at 0 or Z_Entry * 0.2. Let's assume Exit at 0 for simplicity or fixed parameter.
        # User defined Z Range 0.5 to 3.0. This is Entry.
        # Exit implicitly 0 or symmetrical?
        # I'll assume Exit at Z=0 for Sharpe maximization.
        
        # Slippage Cost (One-time on Entry + One-time on Exit)
        
        if position == 0:
            if z > z_entry:
                # Open Short Spread (Sell PAXG, Buy XAU)
                cost = calculate_slippage_cost(row, trade_size_usdc, 'sell_spread')
                position = -1
                entry_price = spread
                pnl -= cost 
                
            elif z < -z_entry:
                # Open Long Spread (Buy PAXG, Sell XAU)
                cost = calculate_slippage_cost(row, trade_size_usdc, 'buy_spread')
                position = 1
                entry_price = spread
                pnl -= cost 
                
        elif position == -1:
            if z <= 0: # Mean Reversion
                # Close Short
                cost = calculate_slippage_cost(row, trade_size_usdc, 'buy_spread') # Inverse action
                gross_profit = (entry_price - spread) * (trade_size_usdc / 2000) # Approx Notional Scaling? 
                # Spread is diff in Price. PnL ~ Spread_Diff * Quantity.
                # Quantity ~ trade_size / Price. ~ 1000 / 2700 ~ 0.37 unit.
                # Exact PnL = (Entry_Spread - Curr_Spread) * Qty
                # We approximate Qty = trade_size_usdc / MidPrice(approx 2700)
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
                
        # Track Max DD
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
        
        # Calc Sharpe (Approx based on single result? Need annualized vol of returns?
        # Simple Sharpe = PnL / |MaxDD| if valid? 
        # Or just Return / Risk
        sharpe = pnl / abs(dd) if dd != 0 else 0
        
        results.append({
            'window': w,
            'z_score': z,
            'pnl': pnl,
            'max_dd': dd,
            'sharpe': sharpe
        })
        print(f"W: {w}, Z: {z:.1f} -> PnL: {pnl:.2f}, Sharpe: {sharpe:.2f}")
        
    # Convert to DF
    res_df = pd.DataFrame(results)
    
    # 1. Heatmap
    pivot = res_df.pivot(index='window', columns='z_score', values='sharpe')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=False, cmap='RdYlGn')
    plt.title("Sharpe Ratio Heatmap")
    plt.xlabel("Z-Score Threshold")
    plt.ylabel("Window Size (sec)")
    plt.savefig("optimizer_heatmap.png")
    print("Saved heatmap to optimizer_heatmap.png")
    
    # 2. Recommendations
    best = res_df.sort_values('sharpe', ascending=False).iloc[0]
    print("\n=== OPTIMIZATION RESULT ===")
    print(f"Best Window: {int(best['window'])}s ({best['window']/3600:.1f}h)")
    print(f"Best Z-Score: {best['z_score']:.1f}")
    print(f"Net Profit: ${best['pnl']:.2f}")
    print(f"Sharpe Ratio: {best['sharpe']:.4f}")
    print(f"Max Drawdown: ${best['max_dd']:.2f}")
    
if __name__ == "__main__":
    optimize()
