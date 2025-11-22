"""
binance_connector.py

BINANCE DATA CONNECTOR - Data Source Only
Handles real-time and historical data from Binance
NO TRADING - just pure data fetching

Features:
- WebSocket real-time data (prices, klines, trades)
- REST API for historical OHLC
- Automatic gap filling
- CSV file management
- Auto-recovery from disconnections
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import time
import threading
import websocket
from pathlib import Path
import logging
import warnings
import requests

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceDataConnector:
    """
    Binance Data Connector - Data Fetching Only
    No trading, just clean market data
    """
    
    # Symbol mapping
    ALL_PAIRS = [
        'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'ADA_USDT', 'DOT_USDT',
        'AVAX_USDT', 'MATIC_USDT', 'LINK_USDT', 'UNI_USDT', 'XRP_USDT',
        'BNB_USDT', 'DOGE_USDT', 'LTC_USDT', 'ATOM_USDT', 'ALGO_USDT'
    ]
    
    BINANCE_PAIR_MAPPING = {
        'BTC_USDT': 'BTCUSDT',
        'ETH_USDT': 'ETHUSDT',
        'SOL_USDT': 'SOLUSDT',
        'ADA_USDT': 'ADAUSDT',
        'DOT_USDT': 'DOTUSDT',
        'AVAX_USDT': 'AVAXUSDT',
        'MATIC_USDT': 'MATICUSDT',
        'LINK_USDT': 'LINKUSDT',
        'UNI_USDT': 'UNIUSDT',
        'XRP_USDT': 'XRPUSDT',
        'BNB_USDT': 'BNBUSDT',
        'DOGE_USDT': 'DOGEUSDT',
        'LTC_USDT': 'LTCUSDT',
        'ATOM_USDT': 'ATOMUSDT',
        'ALGO_USDT': 'ALGOUSDT'
    }
    
    BINANCE_REST_URL = 'https://api.binance.com'
    BINANCE_WS_URL = 'wss://stream.binance.com:9443/ws'
    
    def __init__(self, data_path: str = './data/raw/', update_existing_data: bool = True):
        """Initialize Binance Data Connector"""
        self.data_path = Path(data_path)
        self.update_existing_data = update_existing_data
        
        logger.info(" Binance Data Connector initialized")
        logger.info(f" Data path: {self.data_path}")
        
        # HTTP session
        self.session = requests.Session()
        
        # WebSocket
        self.ws = None
        self.ws_thread = None
        self.ws_running = False
        
        # Data storage
        self.latest_prices = {}
        self.orderbook = {}
        self.trades_stream = []
        
        # Callbacks
        self.price_callbacks = []
        self.kline_callbacks = []
        self.trade_callbacks = []
        
        # Thread lock
        self.data_lock = threading.Lock()
        
        # Health
        self.last_heartbeat = datetime.now()
        self.connection_healthy = True
    
    # ================== REST API ==================
    
    def fetch_ohlc(self, symbol: str, interval: str,
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None,
                   limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch OHLC data from Binance"""
        binance_symbol = self.BINANCE_PAIR_MAPPING.get(symbol)
        
        if not binance_symbol:
            logger.error(f"Symbol {symbol} not in mapping")
            return None
        
        try:
            url = f"{self.BINANCE_REST_URL}/api/v3/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return None
            
            candles = []
            for kline in data:
                candles.append({
                    'timestamp': pd.to_datetime(int(kline[0]), unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            df = pd.DataFrame(candles)
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f" Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLC: {e}")
            return None
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get latest COMPLETED candle - called by live_trader.py"""
        try:
            df = self.fetch_ohlc(symbol, timeframe, limit=2)
            
            if df is None or len(df) < 2:
                logger.warning(f"No candle data for {symbol} {timeframe}")
                return None
            
            candle = df.iloc[-2]  # Second-to-last (completed)
            
            return {
                'timestamp': candle.name,
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            }
            
        except Exception as e:
            logger.error(f"Error getting latest candle: {e}")
            return None
    
    def load_existing_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Load CSV data"""
        file_path = self.data_path / timeframe / f"{symbol}_{timeframe}.csv"
        
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df.columns = df.columns.str.lower()
            
            if all(col in df.columns for col in required_cols):
                return df[required_cols]
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def update_historical_data(self, symbol: str, new_data: pd.DataFrame, 
                              timeframe: str = '1h') -> None:
        """Update CSV with new data"""
        if not self.update_existing_data:
            return
        
        file_path = self.data_path / timeframe / f"{symbol}_{timeframe}.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            existing_df = self.load_existing_data(symbol, timeframe)
            
            if existing_df.empty:
                new_data.to_csv(file_path)
            else:
                combined_df = pd.concat([existing_df, new_data])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.sort_index(inplace=True)
                combined_df.to_csv(file_path)
                
        except Exception as e:
            logger.error(f"Error updating data: {e}")
    
    def fill_data_gaps(self, symbols: Optional[List[str]] = None, 
                      timeframes: Optional[List[str]] = None) -> Dict[str, int]:
        """Fill gaps in data"""
        if symbols is None:
            symbols = self.ALL_PAIRS
        
        if timeframes is None:
            timeframes = ['5m', '15m', '1h']
        
        results = {}
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    candles = self._fill_gap_for_pair(symbol, timeframe)
                    results[f"{symbol}_{timeframe}"] = candles
                    
                    if candles > 0:
                        logger.info(f" {symbol} {timeframe}: +{candles} candles")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error: {e}")
                    results[f"{symbol}_{timeframe}"] = 0
        
        return results
    
    def _fill_gap_for_pair(self, symbol: str, timeframe: str) -> int:
        """Fill gap for specific pair"""
        existing_df = self.load_existing_data(symbol, timeframe)
        
        if existing_df.empty:
            return 0
        
        last_timestamp = existing_df.index[-1]
        current_time = datetime.now()
        
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        time_diff_minutes = (current_time - last_timestamp).total_seconds() / 60
        
        if time_diff_minutes < timeframe_minutes * 2:
            return 0
        
        start_time = int(last_timestamp.timestamp() * 1000) + 1
        new_df = self.fetch_ohlc(symbol, timeframe, start_time=start_time, limit=1000)
        
        if new_df is not None and len(new_df) > 0:
            new_df = new_df[new_df.index > last_timestamp]
            
            if len(new_df) > 0:
                self.update_historical_data(symbol, new_df, timeframe)
                return len(new_df)
        
        return 0
    
    # ================== WEBSOCKET ==================
    
    def connect_websocket(self, pairs: Optional[List[str]] = None,
                         channels: Optional[List[str]] = None) -> None:
        """Connect to Binance WebSocket"""
        if pairs is None:
            pairs = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
        
        if channels is None:
            channels = ['ticker', 'kline']
        
        streams = []
        for pair in pairs:
            binance_symbol = self.BINANCE_PAIR_MAPPING.get(pair)
            if binance_symbol:
                symbol_lower = binance_symbol.lower()
                
                if 'ticker' in channels:
                    streams.append(f"{symbol_lower}@ticker")
                if 'kline' in channels:
                    # ✅ FIXED: Use correct kline format with timeframe
                    streams.append(f"{symbol_lower}@kline_5m")  # Match your trading timeframe
                if 'trade' in channels:
                    streams.append(f"{symbol_lower}@trade")
        
        # ✅ FIXED: Build proper stream URL
        if len(streams) == 1:
            # Single stream - use simple format
            stream_url = f"wss://stream.binance.com:9443/ws/{streams[0]}"
        else:
            # Multiple streams - use combined stream
            stream_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        logger.info(f"Connecting to: {stream_url}")
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                if 'stream' in data and 'data' in data:
                    stream_name = data['stream']
                    stream_data = data['data']
                    
                    with self.data_lock:
                        if '@ticker' in stream_name:
                            self._process_ticker(stream_data)
                        elif '@kline' in stream_name:
                            self._process_kline(stream_data)
                        elif '@trade' in stream_name:
                            self._process_trade(stream_data)
                        
                        self.last_heartbeat = datetime.now()
                        self.connection_healthy = True
                            
            except Exception as e:
                logger.error(f"Error processing message: {e}")
        
        def on_error(ws, error):
            logger.error(f" WebSocket error: {error}")
            self.connection_healthy = False
        
        def on_close(ws, close_status_code, close_msg):
            logger.warning(f" WebSocket closed: {close_status_code}")
            self.ws_running = False
        
        def on_open(ws):
            logger.info(" Binance WebSocket CONNECTED")
            logger.info(f" Streaming: {', '.join(pairs)}")
        
        self.ws = websocket.WebSocketApp(
            stream_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws_running = True
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()
    
    def _run_websocket(self) -> None:
        """Run WebSocket with reconnect"""
        while self.ws_running:
            try:
                self.ws.run_forever()
                if self.ws_running:
                    logger.warning("Reconnecting in 5s...")
                    time.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.ws_running:
                    time.sleep(5)
    
    def disconnect_websocket(self) -> None:
        """Disconnect WebSocket"""
        self.ws_running = False
        if self.ws:
            self.ws.close()
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        logger.info(" Binance disconnected")
    
    def _process_ticker(self, data: Dict) -> None:
        """Process ticker"""
        try:
            symbol = data['s']
            our_symbol = self._reverse_mapping(symbol)
            
            if our_symbol:
                price_data = {
                    'pair': our_symbol,
                    'bid': float(data['b']),
                    'ask': float(data['a']),
                    'last': float(data['c']),
                    'volume': float(data['v']),
                    'high': float(data['h']),
                    'low': float(data['l']),
                    'timestamp': pd.to_datetime(data['E'], unit='ms')
                }
                
                self.latest_prices[our_symbol] = price_data
                
                for callback in self.price_callbacks:
                    callback(price_data)
                
        except Exception as e:
            logger.error(f"Error processing ticker: {e}")
    
    def _process_kline(self, data: Dict) -> None:
        """Process kline"""
        try:
            kline = data['k']
            if not kline['x']:  # Not closed
                return
            
            symbol = kline['s']
            our_symbol = self._reverse_mapping(symbol)
            
            if our_symbol:
                candle_data = {
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }
                
                if our_symbol not in self.latest_prices:
                    self.latest_prices[our_symbol] = {}
                self.latest_prices[our_symbol].update({
                    'last': candle_data['close'],
                    'timestamp': candle_data['timestamp']
                })
                
                if self.update_existing_data:
                    new_df = pd.DataFrame([candle_data])
                    new_df.set_index('timestamp', inplace=True)
                    self.update_historical_data(our_symbol, new_df, '1m')
                
                for callback in self.kline_callbacks:
                    callback(our_symbol, candle_data)
                    
        except Exception as e:
            logger.error(f"Error processing kline: {e}")
    
    def _process_trade(self, data: Dict) -> None:
        """Process trade"""
        try:
            symbol = data['s']
            our_symbol = self._reverse_mapping(symbol)
            
            if our_symbol:
                trade_data = {
                    'pair': our_symbol,
                    'price': float(data['p']),
                    'volume': float(data['q']),
                    'time': pd.to_datetime(data['T'], unit='ms'),
                    'side': 'buy' if data['m'] else 'sell',
                    'timestamp': pd.to_datetime(data['T'], unit='ms')
                }
                
                self.trades_stream.append(trade_data)
                if len(self.trades_stream) > 1000:
                    self.trades_stream = self.trades_stream[-1000:]
                
                for callback in self.trade_callbacks:
                    callback(trade_data)
                    
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
    
    # ================== UTILITIES ==================
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current price"""
        if pair in self.latest_prices:
            return self.latest_prices[pair].get('last')
        
        df = self.load_existing_data(pair, '1h')
        if not df.empty:
            return df['close'].iloc[-1]
        
        return None
    
    def _reverse_mapping(self, binance_symbol: str) -> Optional[str]:
        """Convert Binance symbol to our format"""
        for our_pair, bn_pair in self.BINANCE_PAIR_MAPPING.items():
            if bn_pair == binance_symbol:
                return our_pair
        return None
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe to minutes"""
        mapping = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        return mapping.get(timeframe, 60)
    
    def register_price_callback(self, callback: Callable) -> None:
        """Register price callback"""
        self.price_callbacks.append(callback)
    
    def register_kline_callback(self, callback: Callable) -> None:
        """Register kline callback"""
        self.kline_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status"""
        return {
            'exchange': 'Binance',
            'purpose': 'Data Only',
            'websocket_connected': self.ws_running,
            'connection_healthy': self.connection_healthy,
            'pairs_tracked': len(self.latest_prices),
            'timestamp': datetime.now()
        }


if __name__ == "__main__":
    print(" BINANCE DATA CONNECTOR TEST")
    
    connector = BinanceDataConnector()
    
    # Test REST
    df = connector.fetch_ohlc('BTC_USDT', '1h', limit=5)
    if df is not None:
        print(f" Fetched {len(df)} candles")
    
    # Test latest candle
    candle = connector.get_latest_candle('BTC_USDT', '5m')
    if candle:
        print(f" BTC: ${candle['close']:.2f}")
    
    # Test WebSocket
    connector.connect_websocket(pairs=['BTC_USDT', 'ETH_USDT'])
    time.sleep(5)
    
    if connector.latest_prices:
        print(" Live prices:")
        for pair, data in connector.latest_prices.items():
            print(f"  {pair}: ${data.get('last', 0):.2f}")
    
    connector.disconnect_websocket()
    print(" Test complete!")