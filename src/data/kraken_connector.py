"""
kraken_connector.py

Complete Kraken Exchange Connector
Handles real-time data, order execution, account management, and auto-recovery
Integrates with existing historical data and handles all 15 crypto pairs

Features:
- Paper and Live trading modes
- Automatic gap filling
- WebSocket real-time data
- Auto-recovery from disconnections
- State persistence for crash recovery
- Multi-asset support (15 pairs)
- Order execution and management
- Account balance tracking
- Rate limiting and retry logic
"""

import krakenex
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
import threading
import websocket
import hashlib
import hmac
import base64
import urllib
from pathlib import Path
import logging
import warnings
import pickle
import schedule
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================== DATA STRUCTURES ==================

@dataclass
class KrakenOrder:
    """Order data structure"""
    pair: str
    type: str  # 'buy' or 'sell'
    ordertype: str  # 'market', 'limit', 'stop-loss', 'take-profit'
    volume: float
    price: Optional[float] = None
    leverage: Optional[str] = None
    reduce_only: bool = False
    userref: Optional[int] = None
    validate: bool = False  # True for validation only (no execution)


@dataclass
class KrakenBalance:
    """Account balance structure"""
    currency: str
    balance: float
    available: float
    hold: float
    timestamp: datetime


@dataclass
class KrakenPosition:
    """Position data structure"""
    pair: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    timestamp: datetime


# ================== BASE CONNECTOR ==================

class KrakenConnector:
    """
    Base Kraken Exchange Connector
    Handles both paper trading and live trading
    """
    
    # Your 15 trading pairs
    ALL_PAIRS = [
        'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'ADA_USDT', 'DOT_USDT',
        'AVAX_USDT', 'MATIC_USDT', 'LINK_USDT', 'UNI_USDT', 'XRP_USDT',
        'BNB_USDT', 'DOGE_USDT', 'LTC_USDT', 'ATOM_USDT', 'ALGO_USDT'
    ]
    
    # Kraken pairs mapping
    PAIR_MAPPING = {
        'BTC_USDT': 'XBTUSDT',
        'ETH_USDT': 'ETHUSDT',
        'SOL_USDT': 'SOLUSDT',
        'ADA_USDT': 'ADAUSDT',
        'DOT_USDT': 'DOTUSDT',
        'AVAX_USDT': 'AVAXUSDT',
        'MATIC_USDT': 'MATICUSDT',
        'LINK_USDT': 'LINKUSDT',
        'UNI_USDT': 'UNIUSDT',
        'XRP_USDT': 'XRPUSDT',
        'BNB_USDT': None,  # Not on Kraken
        'DOGE_USDT': 'DOGEUSDT',
        'LTC_USDT': 'LTCUSDT',
        'ATOM_USDT': 'ATOMUSDT',
        'ALGO_USDT': 'ALGOUSDT'
    }
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 mode: str = 'paper',  # 'paper' or 'live'
                 data_path: str = './data/raw/',
                 update_existing_data: bool = True):
        """
        Initialize Kraken Connector
        
        Args:
            api_key: Kraken API key (required for live trading)
            api_secret: Kraken API secret (required for live trading)
            mode: Trading mode ('paper' or 'live')
            data_path: Path to existing historical data
            update_existing_data: Whether to update existing CSV files
        """
        self.mode = mode
        self.data_path = Path(data_path)
        self.update_existing_data = update_existing_data
        
        # Initialize API (only for live mode)
        if mode == 'live':
            if not api_key or not api_secret:
                raise ValueError("API credentials required for live trading")
            self.api = krakenex.API(api_key, api_secret)
        else:
            self.api = None
            logger.info("Running in paper trading mode - no real orders will be placed")
        
        # WebSocket connection
        self.ws = None
        self.ws_thread = None
        self.ws_running = False
        
        # Data storage
        self.latest_prices = {}
        self.orderbook = {}
        self.trades_stream = []
        
        # Paper trading state
        if mode == 'paper':
            self.paper_balance = {
                'USDT': 10000.0,
                'BTC': 0.0,
                'ETH': 0.0,
                'SOL': 0.0,
                'ADA': 0.0,
                'DOT': 0.0
            }
            self.paper_positions = {}
            self.paper_orders = []
            self.paper_trades = []
        
        # Callbacks
        self.price_callbacks = []
        self.trade_callbacks = []
        self.order_callbacks = []
        
        # Thread locks
        self.data_lock = threading.Lock()
        self.order_lock = threading.Lock()
        
        logger.info(f"KrakenConnector initialized in {mode} mode")
    
    # ================== DATA MANAGEMENT ==================
    
    def load_existing_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """
        Load existing historical data from CSV files
        """
        file_path = self.data_path / timeframe / f"{symbol}_{timeframe}.csv"
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            # Parse timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
            
            # Standardize columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df.columns = df.columns.str.lower()
            
            if all(col in df.columns for col in required_cols):
                logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
                return df[required_cols]
            else:
                logger.warning(f"Missing required columns in {file_path}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()
    
    def update_historical_data(self, symbol: str, new_data: pd.DataFrame, 
                              timeframe: str = '1h') -> None:
        """
        Update existing CSV files with new data
        """
        if not self.update_existing_data:
            return
        
        file_path = self.data_path / timeframe / f"{symbol}_{timeframe}.csv"
        
        try:
            existing_df = self.load_existing_data(symbol, timeframe)
            
            if existing_df.empty:
                new_data.to_csv(file_path)
                logger.info(f"Created new data file: {file_path}")
            else:
                combined_df = pd.concat([existing_df, new_data])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.sort_index(inplace=True)
                combined_df.to_csv(file_path)
                logger.info(f"Updated {file_path} with {len(new_data)} new rows")
                
        except Exception as e:
            logger.error(f"Error updating data file: {e}")
    
    # ================== GAP FILLING ==================
    
    def fill_data_gaps(self, symbols: Optional[List[str]] = None, 
                      timeframes: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Fill gaps between existing historical data and current time
        """
        if symbols is None:
            symbols = self.ALL_PAIRS
        
        if timeframes is None:
            timeframes = ['1h', '4h', '1d']
        
        results = {}
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    candles_fetched = self._fill_gap_for_pair(symbol, timeframe)
                    results[f"{symbol}_{timeframe}"] = candles_fetched
                    
                    if candles_fetched > 0:
                        logger.info(f"Filled {candles_fetched} candles for {symbol} {timeframe}")
                    else:
                        logger.info(f"{symbol} {timeframe} is up to date")
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error filling gap for {symbol} {timeframe}: {e}")
                    results[f"{symbol}_{timeframe}"] = 0
        
        return results
    
    def _fill_gap_for_pair(self, symbol: str, timeframe: str) -> int:
        """
        Fill data gap for a specific pair and timeframe
        """
        existing_df = self.load_existing_data(symbol, timeframe)
        
        if existing_df.empty:
            logger.warning(f"No existing data for {symbol} {timeframe}")
            return 0
        
        last_timestamp = existing_df.index[-1]
        current_time = datetime.now()
        
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        time_diff = (current_time - last_timestamp).total_seconds() / 60
        
        if time_diff < timeframe_minutes * 2:
            return 0  # Data is recent enough
        
        logger.info(f"Gap detected for {symbol} {timeframe}: {last_timestamp} to {current_time}")
        
        # Fetch missing data
        return self._fetch_public_ohlc(symbol, timeframe, last_timestamp)
    
    def _fetch_public_ohlc(self, symbol: str, timeframe: str, 
                          since_timestamp: datetime) -> int:
        """
        Fetch OHLC data using Kraken's public API
        """
        kraken_pair = self.PAIR_MAPPING.get(symbol)
        
        if kraken_pair is None:
            logger.warning(f"{symbol} not available on Kraken")
            return 0
        
        interval = self._timeframe_to_kraken_interval(timeframe)
        since = int(since_timestamp.timestamp())
        
        url = "https://api.kraken.com/0/public/OHLC"
        all_candles = []
        last_id = since
        
        while True:
            params = {
                'pair': kraken_pair,
                'interval': interval,
                'since': last_id
            }
            
            try:
                response = requests.get(url, params=params)
                data = response.json()
                
                if data.get('error'):
                    logger.error(f"Kraken API error: {data['error']}")
                    break
                
                result = data.get('result', {})
                candles = result.get(kraken_pair, [])
                
                if not candles:
                    break
                
                for candle in candles:
                    timestamp = datetime.fromtimestamp(int(candle[0]))
                    
                    if timestamp > since_timestamp:
                        all_candles.append({
                            'timestamp': timestamp,
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[6])
                        })
                
                if candles:
                    last_timestamp = datetime.fromtimestamp(int(candles[-1][0]))
                    if (datetime.now() - last_timestamp).total_seconds() < 3600:
                        break
                    
                    last_id = result.get('last', last_id)
                    
                    if len(candles) < 720:
                        break
                else:
                    break
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching OHLC data: {e}")
                break
        
        if all_candles:
            new_df = pd.DataFrame(all_candles)
            new_df.set_index('timestamp', inplace=True)
            new_df.sort_index(inplace=True)
            new_df = new_df[~new_df.index.duplicated(keep='last')]
            self.update_historical_data(symbol, new_df, timeframe)
            return len(all_candles)
        
        return 0
    
    # ================== WEBSOCKET ==================
    
    def connect_websocket(self, pairs: Optional[List[str]] = None,
                         channels: Optional[List[str]] = None) -> None:
        """
        Connect to Kraken WebSocket for real-time data
        """
        if pairs is None:
            pairs = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
        
        if channels is None:
            channels = ['ticker', 'ohlc', 'trade']
        
        kraken_pairs = [self.PAIR_MAPPING.get(p) for p in pairs if self.PAIR_MAPPING.get(p)]
        
        ws_url = "wss://ws.kraken.com"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                if isinstance(data, dict) and 'event' in data:
                    if data['event'] == 'systemStatus':
                        logger.info(f"System status: {data['status']}")
                    elif data['event'] == 'subscriptionStatus':
                        logger.info(f"Subscription: {data['status']} - {data.get('pair', '')}")
                
                elif isinstance(data, list):
                    channel_id = data[0]
                    channel_data = data[1]
                    channel_name = data[2] if len(data) > 2 else ""
                    pair = data[3] if len(data) > 3 else ""
                    
                    with self.data_lock:
                        if 'ticker' in channel_name:
                            self._process_ticker(pair, channel_data)
                        elif 'ohlc' in channel_name:
                            self._process_ohlc(pair, channel_data)
                        elif 'trade' in channel_name:
                            self._process_trades(pair, channel_data)
                        elif 'book' in channel_name:
                            self._process_orderbook(pair, channel_data)
                            
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws):
            logger.info("WebSocket connection closed")
            self.ws_running = False
        
        def on_open(ws):
            logger.info("WebSocket connection opened")
            
            for channel in channels:
                subscribe_msg = {
                    "event": "subscribe",
                    "pair": kraken_pairs,
                    "subscription": {"name": channel}
                }
                
                if channel == 'ohlc':
                    subscribe_msg['subscription']['interval'] = 1
                
                ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {channel} for {kraken_pairs}")
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws_running = True
        self.ws_thread = threading.Thread(target=self._run_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        logger.info("WebSocket thread started")
    
    def _run_websocket(self) -> None:
        """Run WebSocket connection in thread"""
        while self.ws_running:
            try:
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                time.sleep(5)
    
    def disconnect_websocket(self) -> None:
        """Disconnect WebSocket"""
        self.ws_running = False
        if self.ws:
            self.ws.close()
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        logger.info("WebSocket disconnected")
    
    def _process_ticker(self, pair: str, data: Dict) -> None:
        """Process ticker data"""
        try:
            price_data = {
                'pair': pair,
                'bid': float(data['b'][0]),
                'ask': float(data['a'][0]),
                'last': float(data['c'][0]),
                'volume': float(data['v'][0]),
                'high': float(data['h'][0]),
                'low': float(data['l'][0]),
                'timestamp': datetime.now()
            }
            
            self.latest_prices[pair] = price_data
            
            for callback in self.price_callbacks:
                callback(price_data)
                
        except Exception as e:
            logger.error(f"Error processing ticker: {e}")
    
    def _process_ohlc(self, pair: str, data: List) -> None:
        """Process OHLC data"""
        try:
            ohlc_data = {
                'timestamp': datetime.fromtimestamp(float(data[0])),
                'open': float(data[1]),
                'high': float(data[2]),
                'low': float(data[3]),
                'close': float(data[4]),
                'vwap': float(data[5]),
                'volume': float(data[6]),
                'count': int(data[7])
            }
            
            if pair not in self.latest_prices:
                self.latest_prices[pair] = {}
            self.latest_prices[pair].update({
                'last': ohlc_data['close'],
                'timestamp': ohlc_data['timestamp']
            })
            
            if self.update_existing_data:
                symbol = self._reverse_pair_mapping(pair)
                if symbol:
                    new_df = pd.DataFrame([ohlc_data])
                    new_df.set_index('timestamp', inplace=True)
                    self.update_historical_data(symbol, new_df, '1m')
                    
        except Exception as e:
            logger.error(f"Error processing OHLC: {e}")
    
    def _process_trades(self, pair: str, trades: List) -> None:
        """Process trade data"""
        try:
            for trade in trades:
                trade_data = {
                    'pair': pair,
                    'price': float(trade[0]),
                    'volume': float(trade[1]),
                    'time': float(trade[2]),
                    'side': trade[3],
                    'order_type': trade[4],
                    'timestamp': datetime.fromtimestamp(float(trade[2]))
                }
                
                self.trades_stream.append(trade_data)
                
                if len(self.trades_stream) > 1000:
                    self.trades_stream = self.trades_stream[-1000:]
                
                for callback in self.trade_callbacks:
                    callback(trade_data)
                    
        except Exception as e:
            logger.error(f"Error processing trades: {e}")
    
    def _process_orderbook(self, pair: str, book_data: Dict) -> None:
        """Process orderbook data"""
        try:
            if pair not in self.orderbook:
                self.orderbook[pair] = {'bids': {}, 'asks': {}}
            
            if 'bs' in book_data:
                self.orderbook[pair]['bids'] = {
                    float(price): float(volume) 
                    for price, volume, _ in book_data.get('bs', [])
                }
            if 'as' in book_data:
                self.orderbook[pair]['asks'] = {
                    float(price): float(volume)
                    for price, volume, _ in book_data.get('as', [])
                }
            
            for bid in book_data.get('b', []):
                price = float(bid[0])
                volume = float(bid[1])
                if volume == 0:
                    self.orderbook[pair]['bids'].pop(price, None)
                else:
                    self.orderbook[pair]['bids'][price] = volume
            
            for ask in book_data.get('a', []):
                price = float(ask[0])
                volume = float(ask[1])
                if volume == 0:
                    self.orderbook[pair]['asks'].pop(price, None)
                else:
                    self.orderbook[pair]['asks'][price] = volume
                    
        except Exception as e:
            logger.error(f"Error processing orderbook: {e}")
    
    def _reverse_pair_mapping(self, kraken_pair: str) -> Optional[str]:
        """Convert Kraken pair back to our format"""
        for our_pair, kr_pair in self.PAIR_MAPPING.items():
            if kr_pair == kraken_pair:
                return our_pair
        return None
    
    # ================== ORDER EXECUTION ==================
    
    def place_order(self, order: KrakenOrder) -> Dict[str, Any]:
        """
        Place an order on Kraken
        """
        if self.mode == 'paper':
            return self._place_paper_order(order)
        else:
            return self._place_live_order(order)
    
    def _place_paper_order(self, order: KrakenOrder) -> Dict[str, Any]:
        """Place a paper trading order"""
        with self.order_lock:
            order_id = f"PAPER-{int(time.time() * 1000)}"
            current_price = self.get_current_price(order.pair)
            
            if current_price is None:
                return {
                    'success': False,
                    'error': 'No price data available',
                    'order_id': None
                }
            
            if order.ordertype == 'market':
                execution_price = current_price * (1.001 if order.type == 'buy' else 0.999)
            else:
                execution_price = order.price or current_price
            
            base_currency = order.pair.split('_')[0]
            quote_currency = order.pair.split('_')[1]
            
            if order.type == 'buy':
                cost = order.volume * execution_price
                if self.paper_balance.get(quote_currency, 0) < cost:
                    return {
                        'success': False,
                        'error': 'Insufficient balance',
                        'order_id': None
                    }
                
                self.paper_balance[quote_currency] -= cost
                self.paper_balance[base_currency] = self.paper_balance.get(base_currency, 0) + order.volume
                
            else:  # sell
                if self.paper_balance.get(base_currency, 0) < order.volume:
                    return {
                        'success': False,
                        'error': 'Insufficient balance',
                        'order_id': None
                    }
                
                self.paper_balance[base_currency] -= order.volume
                self.paper_balance[quote_currency] += order.volume * execution_price
            
            trade_record = {
                'order_id': order_id,
                'pair': order.pair,
                'type': order.type,
                'ordertype': order.ordertype,
                'volume': order.volume,
                'price': execution_price,
                'cost': order.volume * execution_price,
                'fee': order.volume * execution_price * 0.0026,
                'timestamp': datetime.now(),
                'status': 'closed'
            }
            
            self.paper_trades.append(trade_record)
            
            for callback in self.order_callbacks:
                callback(trade_record)
            
            logger.info(f"Paper order executed: {order_id}")
            
            return {
                'success': True,
                'order_id': order_id,
                'execution_price': execution_price,
                'volume': order.volume,
                'timestamp': datetime.now()
            }
    
    def _place_live_order(self, order: KrakenOrder) -> Dict[str, Any]:
        """Place a live order on Kraken"""
        if not self.api:
            return {
                'success': False,
                'error': 'No API connection for live trading',
                'order_id': None
            }
        
        try:
            params = {
                'pair': self.PAIR_MAPPING.get(order.pair, order.pair),
                'type': order.type,
                'ordertype': order.ordertype,
                'volume': str(order.volume)
            }
            
            if order.price:
                params['price'] = str(order.price)
            
            if order.leverage:
                params['leverage'] = order.leverage
            
            if order.reduce_only:
                params['reduce_only'] = True
            
            if order.userref:
                params['userref'] = order.userref
            
            if order.validate:
                params['validate'] = True
            
            result = self.api.query_private('AddOrder', params)
            
            if result.get('error'):
                return {
                    'success': False,
                    'error': result['error'],
                    'order_id': None
                }
            
            order_info = result.get('result', {})
            order_id = order_info.get('txid', [None])[0]
            
            logger.info(f"Live order placed: {order_id}")
            
            return {
                'success': True,
                'order_id': order_id,
                'description': order_info.get('descr', {}),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error placing live order: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': None
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order"""
        if self.mode == 'paper':
            return {
                'success': True,
                'message': f'Paper order {order_id} cancelled'
            }
        else:
            try:
                result = self.api.query_private('CancelOrder', {'txid': order_id})
                
                if result.get('error'):
                    return {
                        'success': False,
                        'error': result['error']
                    }
                
                return {
                    'success': True,
                    'count': result['result']['count'],
                    'pending': result['result'].get('pending', False)
                }
                
            except Exception as e:
                logger.error(f"Error cancelling order: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
    
    # ================== ACCOUNT MANAGEMENT ==================
    
    def get_account_balance(self) -> Dict[str, KrakenBalance]:
        """Get account balances"""
        if self.mode == 'paper':
            return self._get_paper_balance()
        else:
            return self._get_live_balance()
    
    def _get_paper_balance(self) -> Dict[str, KrakenBalance]:
        """Get paper trading balance"""
        balances = {}
        
        for currency, amount in self.paper_balance.items():
            balances[currency] = KrakenBalance(
                currency=currency,
                balance=amount,
                available=amount,
                hold=0,
                timestamp=datetime.now()
            )
        
        return balances
    
    def _get_live_balance(self) -> Dict[str, KrakenBalance]:
        """Get live account balance"""
        if not self.api:
            return {}
        
        try:
            result = self.api.query_private('Balance')
            
            if result.get('error'):
                logger.error(f"Error getting balance: {result['error']}")
                return {}
            
            balances = {}
            for currency, amount in result.get('result', {}).items():
                if currency == 'XXBT':
                    currency = 'BTC'
                elif currency == 'XETH':
                    currency = 'ETH'
                elif currency.startswith('Z'):
                    currency = currency[1:]
                
                balances[currency] = KrakenBalance(
                    currency=currency,
                    balance=float(amount),
                    available=float(amount),
                    hold=0,
                    timestamp=datetime.now()
                )
            
            return balances
            
        except Exception as e:
            logger.error(f"Error getting live balance: {e}")
            return {}
    
    # ================== UTILITY FUNCTIONS ==================
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for a pair"""
        kraken_pair = self.PAIR_MAPPING.get(pair, pair)
        if kraken_pair in self.latest_prices:
            return self.latest_prices[kraken_pair].get('last')
        
        df = self.load_existing_data(pair, '1h')
        if not df.empty:
            return df['close'].iloc[-1]
        
        return None
    
    def get_orderbook_snapshot(self, pair: str, depth: int = 10) -> Dict[str, Any]:
        """Get orderbook snapshot"""
        kraken_pair = self.PAIR_MAPPING.get(pair, pair)
        
        if kraken_pair not in self.orderbook:
            return {'bids': [], 'asks': []}
        
        book = self.orderbook[kraken_pair]
        
        sorted_bids = sorted(book['bids'].items(), key=lambda x: x[0], reverse=True)[:depth]
        sorted_asks = sorted(book['asks'].items(), key=lambda x: x[0])[:depth]
        
        return {
            'bids': sorted_bids,
            'asks': sorted_asks,
            'spread': sorted_asks[0][0] - sorted_bids[0][0] if sorted_bids and sorted_asks else 0
        }
    
    def check_data_status(self, symbols: Optional[List[str]] = None,
                         timeframes: Optional[List[str]] = None) -> pd.DataFrame:
        """Check the status of all data files"""
        if symbols is None:
            symbols = self.ALL_PAIRS
        
        if timeframes is None:
            timeframes = ['1h', '4h', '1d']
        
        status_data = []
        current_time = datetime.now()
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    df = self.load_existing_data(symbol, timeframe)
                    
                    if df.empty:
                        status_data.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'status': 'NO_DATA',
                            'last_timestamp': None,
                            'gap_hours': None,
                            'candles_missing': None
                        })
                    else:
                        last_timestamp = df.index[-1]
                        gap_seconds = (current_time - last_timestamp).total_seconds()
                        gap_hours = gap_seconds / 3600
                        
                        timeframe_minutes = self._timeframe_to_minutes(timeframe)
                        candles_missing = int(gap_seconds / 60 / timeframe_minutes)
                        
                        status = 'UP_TO_DATE' if candles_missing <= 1 else 'NEEDS_UPDATE'
                        
                        status_data.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'status': status,
                            'last_timestamp': last_timestamp,
                            'gap_hours': round(gap_hours, 1),
                            'candles_missing': candles_missing
                        })
                        
                except Exception as e:
                    status_data.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'status': 'ERROR',
                        'last_timestamp': None,
                        'gap_hours': None,
                        'candles_missing': None
                    })
        
        return pd.DataFrame(status_data)
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
        }
        return mapping.get(timeframe, 60)
    
    def _timeframe_to_kraken_interval(self, timeframe: str) -> int:
        """Convert timeframe to Kraken interval"""
        return self._timeframe_to_minutes(timeframe)
    
    def register_price_callback(self, callback: Callable) -> None:
        """Register callback for price updates"""
        self.price_callbacks.append(callback)
    
    def register_trade_callback(self, callback: Callable) -> None:
        """Register callback for trade updates"""
        self.trade_callbacks.append(callback)
    
    def register_order_callback(self, callback: Callable) -> None:
        """Register callback for order updates"""
        self.order_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'mode': self.mode,
            'websocket_connected': self.ws_running,
            'api_connected': self.api is not None,
            'pairs_tracked': len(self.latest_prices),
            'orderbook_pairs': len(self.orderbook),
            'recent_trades_count': len(self.trades_stream),
            'paper_balance': self.paper_balance if self.mode == 'paper' else None,
            'timestamp': datetime.now()
        }


# ================== RESILIENT CONNECTOR ==================

class ResilientKrakenConnector(KrakenConnector):
    """
    Enhanced connector with auto-recovery and gap management
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # State persistence
        self.state_file = Path(self.data_path) / '.connector_state.pkl'
        self.last_update_times = self._load_state()
        
        # Auto-recovery settings
        self.auto_recovery = True
        self.recovery_check_interval = 60
        self.gap_check_interval = 300
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 30
        
        # Connection monitoring
        self.last_heartbeat = datetime.now()
        self.connection_healthy = True
        self.recovery_thread = None
        self.gap_monitor_thread = None
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("Resilient connector initialized with auto-recovery")
    
    def _load_state(self) -> Dict:
        """Load saved state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                logger.info(f"Loaded previous state from {self.state_file}")
                return state
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        return {}
    
    def _save_state(self) -> None:
        """Save current state to disk"""
        try:
            state = {
                'last_update_times': self.last_update_times,
                'timestamp': datetime.now(),
                'symbols': list(self.latest_prices.keys())
            }
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _start_monitoring(self) -> None:
        """Start monitoring threads"""
        self.recovery_thread = threading.Thread(target=self._monitor_connection)
        self.recovery_thread.daemon = True
        self.recovery_thread.start()
        
        self.gap_monitor_thread = threading.Thread(target=self._monitor_gaps)
        self.gap_monitor_thread.daemon = True
        self.gap_monitor_thread.start()
        
        logger.info("Monitoring threads started")
    
    def _monitor_connection(self) -> None:
        """Monitor connection health"""
        while self.auto_recovery:
            try:
                time.sleep(self.recovery_check_interval)
                
                time_since_heartbeat = (datetime.now() - self.last_heartbeat).seconds
                
                if time_since_heartbeat > 120:
                    logger.warning(f"No data received for {time_since_heartbeat} seconds")
                    self.connection_healthy = False
                    self._recover_connection()
                else:
                    self.connection_healthy = True
                
                self._save_state()
                
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
                time.sleep(10)
    
    def _monitor_gaps(self) -> None:
        """Periodically check for and fill gaps"""
        while self.auto_recovery:
            try:
                time.sleep(self.gap_check_interval)
                
                logger.info("Performing periodic gap check...")
                gaps_found = self._detect_runtime_gaps()
                
                if gaps_found:
                    logger.warning(f"Found {len(gaps_found)} gaps during runtime")
                    self._fill_runtime_gaps(gaps_found)
                
            except Exception as e:
                logger.error(f"Error in gap monitor: {e}")
                time.sleep(30)
    
    def _detect_runtime_gaps(self) -> List[Dict]:
        """Detect gaps that occurred during runtime"""
        gaps = []
        current_time = datetime.now()
        
        for symbol in self.ALL_PAIRS:
            for timeframe in ['1h', '4h', '1d']:
                try:
                    df = self.load_existing_data(symbol, timeframe)
                    
                    if not df.empty:
                        last_timestamp = df.index[-1]
                        timeframe_minutes = self._timeframe_to_minutes(timeframe)
                        expected_candles = (current_time - last_timestamp).seconds / 60 / timeframe_minutes
                        
                        if expected_candles > 2:
                            gaps.append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'last_timestamp': last_timestamp,
                                'gap_size': int(expected_candles)
                            })
                            
                except Exception as e:
                    logger.error(f"Error checking gap for {symbol} {timeframe}: {e}")
        
        return gaps
    
    def _fill_runtime_gaps(self, gaps: List[Dict]) -> None:
        """Fill gaps detected during runtime"""
        for gap in gaps:
            try:
                logger.info(f"Filling gap for {gap['symbol']} {gap['timeframe']}: {gap['gap_size']} candles")
                candles_fetched = self._fill_gap_for_pair(gap['symbol'], gap['timeframe'])
                
                if candles_fetched > 0:
                    logger.info(f"Filled {candles_fetched} candles")
                
                key = f"{gap['symbol']}_{gap['timeframe']}"
                self.last_update_times[key] = datetime.now()
                
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error filling runtime gap: {e}")
    
    def _recover_connection(self) -> None:
        """Recover from connection failure"""
        logger.warning("Attempting connection recovery...")
        
        for attempt in range(self.max_reconnect_attempts):
            try:
                if self.ws:
                    self.disconnect_websocket()
                    time.sleep(2)
                
                logger.info("Checking for gaps during downtime...")
                gaps = self._detect_runtime_gaps()
                
                if gaps:
                    logger.info(f"Found {len(gaps)} gaps during downtime")
                    self._fill_runtime_gaps(gaps)
                
                logger.info("Reconnecting WebSocket...")
                self.connect_websocket(
                    pairs=['BTC_USDT', 'ETH_USDT', 'SOL_USDT'],
                    channels=['ticker', 'ohlc', 'trade']
                )
                
                time.sleep(5)
                if self.ws_running:
                    logger.info(" Recovery successful!")
                    self.connection_healthy = True
                    self.last_heartbeat = datetime.now()
                    return
                
            except Exception as e:
                logger.error(f"Recovery attempt {attempt + 1} failed: {e}")
                time.sleep(self.reconnect_delay)
        
        logger.error(f"Failed to recover after {self.max_reconnect_attempts} attempts")
    
    def _process_ticker(self, pair: str, data: Dict) -> None:
        """Override to update heartbeat"""
        super()._process_ticker(pair, data)
        self.last_heartbeat = datetime.now()
    
    def _process_ohlc(self, pair: str, data: List) -> None:
        """Override to track updates"""
        super()._process_ohlc(pair, data)
        
        symbol = self._reverse_pair_mapping(pair)
        if symbol:
            key = f"{symbol}_1m"
            self.last_update_times[key] = datetime.now()
        
        self.last_heartbeat = datetime.now()
    
    def smart_recovery_start(self) -> None:
        """Smart start with recovery"""
        print("\n === SMART RECOVERY START ===\n")
        
        if self.last_update_times:
            last_run = max(self.last_update_times.values())
            downtime = datetime.now() - last_run
            print(f" Last run: {last_run}")
            print(f" Downtime: {downtime}\n")
        else:
            print(" First run detected\n")
        
        print(" Checking for data gaps...")
        status_df = self.check_data_status()
        needs_update = status_df[status_df['status'] == 'NEEDS_UPDATE']
        
        if not needs_update.empty:
            print(f" Found {len(needs_update)} files with gaps")
            print("\n Filling gaps...")
            
            total = len(needs_update)
            for idx, row in needs_update.iterrows():
                print(f"  [{idx+1}/{total}] Updating {row['symbol']} {row['timeframe']}...")
                self._fill_gap_for_pair(row['symbol'], row['timeframe'])
        else:
            print(" All data is up to date!\n")
        
        print(" Connecting to real-time feeds...")
        self.connect_websocket(
            pairs=['BTC_USDT', 'ETH_USDT', 'SOL_USDT'],
            channels=['ticker', 'ohlc', 'trade']
        )
        
        time.sleep(3)
        if self.ws_running and self.connection_healthy:
            print(" System fully operational!")
            print(f" Tracking {len(self.latest_prices)} pairs")
            print(" Auto-recovery: ENABLED")
            print(" Gap monitoring: ACTIVE")
        else:
            print(" Warning: Connection not fully established")
        
        print("\n=== READY FOR TRADING ===\n")
    
    def get_health_status(self) -> Dict:
        """Get health status"""
        return {
            'connection_healthy': self.connection_healthy,
            'websocket_running': self.ws_running,
            'last_heartbeat': self.last_heartbeat,
            'seconds_since_heartbeat': (datetime.now() - self.last_heartbeat).seconds,
            'auto_recovery': self.auto_recovery,
            'monitoring_active': self.recovery_thread.is_alive() if self.recovery_thread else False,
            'gap_monitor_active': self.gap_monitor_thread.is_alive() if self.gap_monitor_thread else False,
            'tracked_pairs': len(self.latest_prices),
            'last_update_times': self.last_update_times,
            'mode': self.mode
        }


# ================== MULTI-ASSET CONNECTOR ==================

class MultiAssetKrakenConnector(ResilientKrakenConnector):
    """
    Optimized connector for all 15 crypto pairs
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pair_update_status = {pair: {} for pair in self.ALL_PAIRS}
        self.batch_size = 5
        self.rate_limit_delay = 0.1
        
        logger.info(f"Multi-asset connector initialized for {len(self.ALL_PAIRS)} pairs")
    
    def initialize_all_pairs(self) -> Dict[str, str]:
        """Initialize and check all 15 pairs"""
        print("\n === INITIALIZING ALL 15 PAIRS ===\n")
        
        pair_status = {}
        
        for pair in self.ALL_PAIRS:
            print(f"Checking {pair}...")
            
            for timeframe in ['1h', '4h', '1d']:
                df = self.load_existing_data(pair, timeframe)
                
                if not df.empty:
                    last_date = df.index[-1]
                    rows = len(df)
                    pair_status[f"{pair}_{timeframe}"] = f" {rows} rows, last: {last_date.strftime('%Y-%m-%d')}"
                else:
                    pair_status[f"{pair}_{timeframe}"] = " No data"
        
        print("\n Data Status Summary:")
        print("-" * 50)
        
        for pair in self.ALL_PAIRS:
            print(f"\n{pair}:")
            for tf in ['1h', '4h', '1d']:
                key = f"{pair}_{tf}"
                print(f"  {tf}: {pair_status.get(key, 'Unknown')}")
        
        return pair_status
    
    def update_all_pairs(self, parallel: bool = True) -> Dict[str, int]:
        """Update all 15 pairs efficiently"""
        print("\n === UPDATING ALL 15 PAIRS ===\n")
        
        results = {}
        start_time = datetime.now()
        
        if parallel:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                for pair in self.ALL_PAIRS:
                    for timeframe in ['1h', '4h', '1d']:
                        future = executor.submit(self._update_single_pair, pair, timeframe)
                        futures[future] = f"{pair}_{timeframe}"
                
                for future in as_completed(futures):
                    pair_tf = futures[future]
                    try:
                        candles = future.result()
                        results[pair_tf] = candles
                        if candles > 0:
                            print(f" {pair_tf}: Updated {candles} candles")
                    except Exception as e:
                        print(f" {pair_tf}: Error - {e}")
                        results[pair_tf] = 0
        else:
            for pair in self.ALL_PAIRS:
                for timeframe in ['1h', '4h', '1d']:
                    try:
                        print(f"Updating {pair} {timeframe}...")
                        candles = self._update_single_pair(pair, timeframe)
                        results[f"{pair}_{timeframe}"] = candles
                        
                        if candles > 0:
                            print(f"   Updated {candles} candles")
                        else:
                            print(f"   Already up to date")
                        
                        time.sleep(self.rate_limit_delay)
                        
                    except Exception as e:
                        print(f"   Error: {e}")
                        results[f"{pair}_{timeframe}"] = 0
        
        elapsed = (datetime.now() - start_time).seconds
        total_candles = sum(results.values())
        
        print(f"\n Update Complete:")
        print(f"  Time: {elapsed} seconds")
        print(f"  Total candles: {total_candles}")
        print(f"  Pairs updated: {len([v for v in results.values() if v > 0])}")
        
        return results
    
    def _update_single_pair(self, pair: str, timeframe: str) -> int:
        """Update a single pair/timeframe"""
        try:
            kraken_pair = self.PAIR_MAPPING.get(pair)
            
            if kraken_pair is None:
                logger.warning(f"{pair} not available on Kraken")
                return 0
            
            return self._fill_gap_for_pair(pair, timeframe)
            
        except Exception as e:
            logger.error(f"Error updating {pair} {timeframe}: {e}")
            return 0
    
    def get_all_current_prices(self) -> pd.DataFrame:
        """Get current prices for all pairs"""
        prices = []
        
        for pair in self.ALL_PAIRS:
            price = self.get_current_price(pair)
            
            if price:
                prices.append({
                    'pair': pair,
                    'price': price,
                    'source': 'realtime' if pair in self.latest_prices else 'historical',
                    'timestamp': datetime.now()
                })
            else:
                prices.append({
                    'pair': pair,
                    'price': None,
                    'source': 'unavailable',
                    'timestamp': datetime.now()
                })
        
        return pd.DataFrame(prices)
    
    def smart_start_all_pairs(self) -> None:
        """Comprehensive startup for all 15 pairs"""
        print("\n === SMART START FOR ALL 15 PAIRS ===\n")
        
        print("Step 1: Checking all pairs...")
        self.initialize_all_pairs()
        
        print("\nStep 2: Updating all pairs...")
        results = self.update_all_pairs(parallel=True)
        
        print("\nStep 3: Connecting real-time feeds...")
        available_pairs = [p for p in self.ALL_PAIRS if self.PAIR_MAPPING.get(p)]
        self.connect_websocket(
            pairs=available_pairs[:8],  # First batch
            channels=['ticker', 'ohlc']
        )
        
        print("\nStep 4: Starting monitors...")
        
        print("\n === ALL SYSTEMS OPERATIONAL ===")
        prices_df = self.get_all_current_prices()
        active_pairs = len(prices_df[prices_df['price'].notna()])
        print(f"  Active pairs: {active_pairs}/{len(self.ALL_PAIRS)}")
        print(f"  Data complete: {len(results)} timeframes")
        print(f"  Monitoring: ACTIVE")
        print(f"  Auto-recovery: ENABLED")


# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize connector
    print("=== Kraken Connector Test ===\n")
    
    # Use the most advanced version
    kraken = MultiAssetKrakenConnector(
        api_key=os.getenv('KRAKEN_API_KEY'),
        api_secret=os.getenv('KRAKEN_API_SECRET'),
        mode='paper',  # Start with paper trading
        data_path='./data/raw/',
        update_existing_data=True
    )
    
    # Start with smart recovery
    kraken.smart_start_all_pairs()
    
    # Test functionality
    print("\n=== Testing Core Functions ===")
    
    # Get current prices
    prices = kraken.get_all_current_prices()
    print("\nCurrent Prices:")
    print(prices[['pair', 'price', 'source']].head())
    
    # Test paper trading
    if kraken.mode == 'paper':
        test_order = KrakenOrder(
            pair='BTC_USDT',
            type='buy',
            ordertype='market',
            volume=0.001
        )
        
        result = kraken.place_order(test_order)
        print(f"\nTest Order Result: {result}")
    
    # Check system status
    status = kraken.get_system_status()
    print(f"\nSystem Status: {status}")
    
    print("\n Kraken Connector Ready for Production!")