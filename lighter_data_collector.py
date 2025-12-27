
import asyncio
import os
import glob
import csv
import json
import time
import logging
import traceback
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any

from dotenv import load_dotenv
from exchanges.lighter import LighterClient

# Load env vars
load_dotenv()

# Logger Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collector.log"),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger("DataCollector")

class LighterDataCollector:
    def __init__(self):
        self.paxg_id = 48
        self.xau_id = 92
        
        # Clients
        self.client_paxg = None
        self.client_xau = None
        
        self.running = False
        self.current_csv_file = None
        self.csv_file_handle = None
        self.csv_writer = None

        # Market Status State
        self.last_xau_price = None
        self.last_xau_update_time = time.time()

    async def setup_clients(self):
        """Setup and connect clients."""
        logger.info("Connecting to Lighter...")
        
        # Helper config
        class Config:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        # PAXG Setup
        cfg_paxg = {
            "ticker": "PAXG",
            "contract_id": self.paxg_id,
            "tick_size": Decimal("0.01"),
            "close_order_side": "sell",
            "account_index": int(os.getenv('LIGHTER_ACCOUNT_INDEX', '0')),
        }
        self.client_paxg = LighterClient(Config(cfg_paxg))
        
        # XAU Setup
        cfg_xau = {
            "ticker": "XAU",
            "contract_id": self.xau_id,
            "tick_size": Decimal("0.01"),
            "close_order_side": "sell",
            "account_index": int(os.getenv('LIGHTER_ACCOUNT_INDEX', '0')),
        }
        self.client_xau = LighterClient(Config(cfg_xau))
        
        await asyncio.gather(
            self.client_paxg.connect(),
            self.client_xau.connect()
        )
        logger.info("Connected.")

    def get_top_depth(self, client: LighterClient, n: int = 5):
        """Extract top N bids and asks."""
        if not client.ws_manager or not client.ws_manager.order_book:
            return [], []
            
        ob = client.ws_manager.order_book
        
        # Bids: Descending price
        bids = sorted(ob['bids'].items(), key=lambda x: float(x[0]), reverse=True)[:n]
        # Asks: Ascending price
        asks = sorted(ob['asks'].items(), key=lambda x: float(x[0]))[:n]
        
        # Format as list of dicts for JSON
        bids_fmt = [{"p": float(p), "s": float(s)} for p, s in bids]
        asks_fmt = [{"p": float(p), "s": float(s)} for p, s in asks]
        
        return bids_fmt, asks_fmt

    def get_best_price(self, depth_list):
        """Get best price from depth list."""
        if not depth_list: return None
        return depth_list[0]['p']

    def check_market_status(self, current_xau_price):
        """
        Check if market is effectively open.
        Returns: True (Open), False (Closed/Stagnant)
        """
        # 1. Update Stagnation Tracker
        if current_xau_price is not None:
            if self.last_xau_price is None or abs(current_xau_price - self.last_xau_price) > 1e-6:
                self.last_xau_price = current_xau_price
                self.last_xau_update_time = time.time()
        
        # Check Stagnation (> 10 mins = 600s)
        time_since_update = time.time() - self.last_xau_update_time
        if time_since_update > 600:
            return False # Stagnant
            
        # 2. Check Schedule (UTC)
        now_utc = datetime.utcnow()
        weekday = now_utc.weekday() # Mon=0, Sun=6
        hour = now_utc.hour
        
        # Close: Friday 22:00 UTC to Sunday 23:00 UTC
        # Saturday (5) -> Closed
        if weekday == 5:
            return False
            
        # Friday (4) >= 22:00 -> Closed
        if weekday == 4 and hour >= 22:
            return False
            
        # Sunday (6) < 23:00 -> Closed
        if weekday == 6 and hour < 23:
            return False
            
        # 3. Check Holidays (Dec 25, Jan 1)
        month = now_utc.month
        day = now_utc.day
        if (month == 12 and day == 25) or (month == 1 and day == 1):
            return False
            
        return True

    def cleanup_old_files(self):
        """Delete CSV files older than 48 hours to save disk space."""
        try:
            retention_hours = 48
            cutoff_time = time.time() - (retention_hours * 3600)
            
            files = glob.glob("market_data_*.csv")
            for f in files:
                try:
                    # simplistic check: modification time
                    mtime = os.path.getmtime(f)
                    if mtime < cutoff_time:
                        os.remove(f)
                        logger.info(f"Cleanup: Deleted old file {f}")
                except Exception as e:
                    logger.warning(f"Cleanup Error on {f}: {e}")
        except Exception as e:
            logger.error(f"Cleanup Failed: {e}")

    def update_csv_file(self):
        """Rotate CSV file based on current hour."""
        now = datetime.now()
        filename = f"market_data_{now.strftime('%Y%m%d_%H')}.csv"
        
        if filename != self.current_csv_file:
            # Trigger cleanup on rotation
            self.cleanup_old_files()
            
            # Close old
            if self.csv_file_handle:
                self.csv_file_handle.close()
                
            self.current_csv_file = filename
            
            # Check if exists to write header
            file_exists = os.path.isfile(filename)
            
            self.csv_file_handle = open(filename, 'a', newline='')
            fieldnames = [
                'timestamp', 'readable_time', 
                'paxg_best_bid', 'paxg_best_ask', 
                'xau_best_bid', 'xau_best_ask', 
                'paxg_spread', 'xau_spread',
                'paxg_bids_depth', 'paxg_asks_depth',
                'xau_bids_depth', 'xau_asks_depth',
                'is_market_open'
            ]
            self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=fieldnames)
            
            if not file_exists:
                self.csv_writer.writeheader()
                self.csv_file_handle.flush()
                
            logger.info(f"Rotated to CSV: {filename}")

    async def run(self):
        self.running = True
        
        while self.running:
            try:
                # Ensure connection
                if not self.client_paxg or not self.client_paxg.ws_manager:
                    await self.setup_clients()
                
                # Loop
                while self.running:
                    start_time = time.time()
                    
                    # 1. Rotate File
                    self.update_csv_file()
                    
                    # 2. Get Data
                    paxg_bids, paxg_asks = self.get_top_depth(self.client_paxg, 5)
                    xau_bids, xau_asks = self.get_top_depth(self.client_xau, 5)
                    
                    ts = time.time()
                    readable = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    
                    paxg_bb = self.get_best_price(paxg_bids)
                    paxg_ba = self.get_best_price(paxg_asks)
                    xau_bb = self.get_best_price(xau_bids)
                    xau_ba = self.get_best_price(xau_asks)
                    
                    # 3. Check Market Status
                    # Use Bid as price reference (if avail)
                    is_open = self.check_market_status(xau_bb)
                    
                    # Spreads (Bid-Ask Spread within the market)
                    paxg_spread_val = (paxg_ba - paxg_bb) if (paxg_bb and paxg_ba) else None
                    xau_spread_val = (xau_ba - xau_bb) if (xau_bb and xau_ba) else None
                    
                    # 4. Write Data
                    row = {
                        'timestamp': ts,
                        'readable_time': readable,
                        'paxg_best_bid': paxg_bb,
                        'paxg_best_ask': paxg_ba,
                        'xau_best_bid': xau_bb,
                        'xau_best_ask': xau_ba,
                        'paxg_spread': paxg_spread_val,
                        'xau_spread': xau_spread_val,
                        'paxg_bids_depth': json.dumps(paxg_bids),
                        'paxg_asks_depth': json.dumps(paxg_asks),
                        'xau_bids_depth': json.dumps(xau_bids),
                        'xau_asks_depth': json.dumps(xau_asks),
                        'is_market_open': is_open
                    }
                    
                    if self.csv_writer:
                        self.csv_writer.writerow(row)
                        self.csv_file_handle.flush()
                        
                    # 5. Wait for next second
                    elapsed = time.time() - start_time
                    delay = max(0, 1.0 - elapsed)
                    if delay > 0:
                        await asyncio.sleep(delay)
                        
            except Exception as e:
                logger.error(f"Collector Error: {e}")
                traceback.print_exc()
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
                # Re-init clients on error
                try:
                    if self.client_paxg: await self.client_paxg.disconnect()
                    if self.client_xau: await self.client_xau.disconnect()
                except:
                    pass
                self.client_paxg = None
                self.client_xau = None

    async def shutdown(self):
        self.running = False
        if self.csv_file_handle:
            self.csv_file_handle.close()
        if self.client_paxg: await self.client_paxg.disconnect()
        if self.client_xau: await self.client_xau.disconnect()

if __name__ == "__main__":
    collector = LighterDataCollector()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(collector.run())
    except KeyboardInterrupt:
        loop.run_until_complete(collector.shutdown())
