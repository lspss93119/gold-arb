
import asyncio
import os
import json
import csv
import time
from datetime import datetime
import pandas as pd
from decimal import Decimal
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from lighter.api.order_api import OrderApi
from lighter.api_client import ApiClient
from lighter.configuration import Configuration
from exchanges.lighter import LighterClient

# Helper to mock config
class Config:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

async def fetch_extended_history(market_id: int, limit_total: int = 1000):
    """
    Fetch extended history for a market using authenticated calls if possible,
    or public API with pagination.
    """
    load_dotenv()
    
    # Setup LighterClient to reuse authentication logic
    config_dict = {
        "ticker": "TEMP",
        "contract_id": market_id,
        "tick_size": Decimal("0.01"),
        "close_order_side": "sell",
        "account_index": int(os.getenv('LIGHTER_ACCOUNT_INDEX', '0')),
    }
    client = LighterClient(Config(config_dict))
    
    try:
        await client.connect()
        
        # Access the underlying OrderApi
        # LighterClient -> lighter_client (SignerClient) -> order_api (OrderApi) ??
        # Let's inspect client.lighter_client structure
        signer_client = client.lighter_client
        
        # In lighter/signer_client.py, it usually has .order_api or we create one with its api_client
        if hasattr(signer_client, 'order_api'):
            order_api = signer_client.order_api
        else:
            # Create new OrderApi using the authenticated api_client from signer_client if available
            # Or creating fresh one
            print("Creating direct OrderApi instance...")
            # Note: Authenticated calls might need signatures. 
            # If get_trades is public but needs 'auth' param?
            # Let's try calling client.lighter_client.order_api.trades if it exists
            # Or manually calling it.
            order_api = OrderApi(signer_client.api_client)

        print(f"Fetching recent trades for Market {market_id} (Limit 100)...")
        
        # Use recent_trades as it works reliably without complex auth flows for now
        # Ideally we want more, but 100 is enough for a small sample optimization
        try:
            res = await order_api.recent_trades(market_id=market_id, limit=100)
            if hasattr(res, 'trades') and res.trades:
                all_trades = res.trades
                print(f"Fetched {len(all_trades)} trades.")
            else:
                print("No trades found.")
        except Exception as e:
            print(f"Error fetching trades: {e}")
            
        return all_trades

        
        return all_trades

    finally:
        await client.disconnect()

async def collect_and_save():
    paxg_trades = await fetch_extended_history(48, 1000)
    xau_trades = await fetch_extended_history(92, 1000)
    
    # Debug first item if available
    if paxg_trades:
        print("First PAXG Trade attrs:", dir(paxg_trades[0]))

    # Save to JSON
    # Safely get attributes
    def to_dict(t):
        return {
            "price": float(getattr(t, 'price', 0)),
            "timestamp": getattr(t, 'timestamp', 0),
            "size": float(getattr(t, 'amount', getattr(t, 'size', 0)))
        }

    data = {
        "paxg": [to_dict(t) for t in paxg_trades],
        "xau": [to_dict(t) for t in xau_trades]
    }
    
    with open("historical_trades.json", "w") as f:
        json.dump(data, f)
        
    print(f"Saved {len(data['paxg'])} PAXG trades and {len(data['xau'])} XAU trades to historical_trades.json")
    
    # Save to CSV (Matched Spreads)
    csv_file = "historical_trades.csv"
    
    print("Processing spreads with Pandas...")
    try:
        df_p = pd.DataFrame(data['paxg'])
        df_x = pd.DataFrame(data['xau'])
        
        if df_p.empty or df_x.empty:
            print("Insufficient data to generate spreads.")
            return

        df_p = df_p.rename(columns={'price': 'price_paxg', 'size': 'size_paxg'})
        df_x = df_x.rename(columns={'price': 'price_xau', 'size': 'size_xau'})
        
        # Ensure timestamp is float/int
        df_p['timestamp'] = df_p['timestamp'].astype(float)
        df_x['timestamp'] = df_x['timestamp'].astype(float)
        
        df_p = df_p.sort_values('timestamp')
        df_x = df_x.sort_values('timestamp')
        
        # Merge on nearest timestamp
        # Tolerance: 60 seconds (check timestamp unit!) 
        # API usually returns Seconds or Milliseconds.
        # If milliseconds, 60s = 60000.
        # The script fetch logic normalizes to seconds in `fetch_lighter_history` but here `fetch_extended_history` keeps raw?
        # Let's check a sample timestamp. If > 1e11, it's ms.
        avg_ts = df_p['timestamp'].mean()
        tolerance = 60000 if avg_ts > 1e11 else 60
        
        df = pd.merge_asof(
            df_p, df_x, 
            on='timestamp', 
            direction='nearest', 
            tolerance=tolerance
        ).dropna()
        
        df['spread'] = df['price_paxg'] - df['price_xau']
        
        # Format Time
        def format_ts(ts):
            ts = float(ts)
            sec = ts / 1000.0 if ts > 1e11 else ts
            return datetime.fromtimestamp(sec).strftime('%Y-%m-%d %H:%M:%S')
            
        df['readable_time'] = df['timestamp'].apply(format_ts)
        
        # Select columns
        cols = ['timestamp', 'readable_time', 'price_paxg', 'price_xau', 'spread']
        df[cols].sort_values('timestamp', ascending=False).to_csv(csv_file, index=False)
        
        print(f"Saved {len(df)} matched spread rows to {csv_file}")
        
    except Exception as e:
        print(f"Error generating CSV: {e}")

if __name__ == "__main__":
    asyncio.run(collect_and_save())
