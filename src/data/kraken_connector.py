"""
kraken_connector.py - FIXED VERSION

KRAKEN TRADING CONNECTOR - Trading Execution Only
Handles order execution, balance, and positions on Kraken
NO DATA FETCHING - just pure trading

CHANGES:
- Renamed class to KrakenConnector (was KrakenTradingConnector)
- Added compatibility methods for live_trader.py
- Fixed Kraken pair mapping
- Added better error messages
"""

import krakenex
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import time
import threading
import logging
import warnings
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
    validate: bool = False


@dataclass
class KrakenBalance:
    """Account balance"""
    currency: str
    balance: float
    available: float
    hold: float
    timestamp: datetime


@dataclass
class KrakenPosition:
    """Position data"""
    pair: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    timestamp: datetime


# ================== KRAKEN CONNECTOR ==================

class KrakenConnector:  # FIXED: Was KrakenTradingConnector
    """
    Kraken Connector - Trading Execution Only
    No data fetching, just order management
    
    NOTE: This connector is for TRADING only!
    Use BinanceDataConnector for market data.
    """
    
    # Kraken pair mapping (verified formats)
    KRAKEN_PAIR_MAPPING = {
        # Standard pairs
        'BTC_USDT': 'XBTUSD',   # Kraken uses XBT not BTC
        'ETH_USDT': 'ETHUSD',
        'SOL_USDT': 'SOLUSD',
        'ADA_USDT': 'ADAUSD',
        'DOT_USDT': 'DOTUSD',
        'AVAX_USDT': 'AVAXUSD',
        'MATIC_USDT': 'MATICUSD',
        'LINK_USDT': 'LINKUSD',
        'UNI_USDT': 'UNIUSD',
        'XRP_USDT': 'XRPUSD',
        'DOGE_USDT': 'DOGEUSD',
        'LTC_USDT': 'LTCUSD',
        'ATOM_USDT': 'ATOMUSD',
        'ALGO_USDT': 'ALGOUSD',
        # Aliases
        'BTCUSDT': 'XBTUSD',
        'ETHUSDT': 'ETHUSD',
    }
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 mode: str = 'paper',
                 data_path: str = './data/raw/'):  # NEW: Added for compatibility
        """
        Initialize Kraken Connector
        
        Args:
            api_key: Kraken API key (required for live trading)
            api_secret: Kraken API secret (required for live trading)
            mode: Trading mode ('paper' or 'live')
            data_path: Path to data files (for compatibility only)
        """
        self.mode = mode
        self.data_path = data_path  # Stored but not used (Kraken is trading-only)
        
        logger.info("="*60)
        logger.info(" Kraken Connector initialized")
        logger.info(f" Mode: {mode.upper()}")
        if mode == 'paper':
            logger.info(" Paper trading - no real orders")
        else:
            logger.info(" LIVE TRADING - real money at risk!")
        logger.info("="*60)
        
        # Initialize Kraken API
        try:
            self.api = krakenex.API()
            
            if api_key and api_secret:
                self.api.key = api_key
                self.api.secret = api_secret
                logger.info(" Kraken credentials loaded")
                self.has_credentials = True
            else:
                self.has_credentials = False
                if mode == 'live':
                    logger.warning(" No credentials - cannot execute live trades!")
                logger.info(" Running without credentials")
                
        except Exception as e:
            logger.error(f" Failed to initialize Kraken API: {e}")
            self.api = None
            self.has_credentials = False
        
        # Paper trading state
        if mode == 'paper':
            self.paper_balance = {
                'USDT': 100.0,  # FIXED: Match live_trader.py default
                'BTC': 0.0,
                'ETH': 0.0,
                'SOL': 0.0,
                'ADA': 0.0,
                'DOT': 0.0,
                'AVAX': 0.0
            }
            self.paper_positions = {}
            self.paper_orders = []
            self.paper_trades = []
            logger.info(f" Paper balance: ${self.paper_balance['USDT']:.2f} USDT")
        
        # Callbacks
        self.order_callbacks = []
        
        # Thread lock
        self.order_lock = threading.Lock()
        
        # Configuration
        self.slippage_pct = 0.001  # 0.1% slippage
    
    # ================== ORDER EXECUTION ==================
    
    def place_order(self, order: KrakenOrder, current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place order on Kraken
        
        Args:
            order: KrakenOrder object
            current_price: Current market price (for paper trading)
            
        Returns:
            Dict with order result
        """
        logger.info(f"\n{'='*60}")
        logger.info(f" PLACING ORDER")
        logger.info(f"{'='*60}")
        logger.info(f"  Pair: {order.pair}")
        logger.info(f"  Type: {order.type.upper()}")
        logger.info(f"  Volume: {order.volume}")
        logger.info(f"  Order Type: {order.ordertype}")
        logger.info(f"  Mode: {self.mode.upper()}")
        logger.info(f"{'='*60}")
        
        if self.mode == 'paper':
            return self._place_paper_order(order, current_price)
        else:
            return self._place_live_order(order)
    
    def _place_paper_order(self, order: KrakenOrder, current_price: Optional[float]) -> Dict[str, Any]:
        """Execute paper trading order"""
        with self.order_lock:
            order_id = f"PAPER-{int(time.time() * 1000)}"
            
            if current_price is None:
                logger.error(" No price provided for paper order")
                return {
                    'success': False,
                    'error': 'No price provided for paper trading',
                    'order_id': None
                }
            
            # Calculate execution price with slippage
            if order.ordertype == 'market':
                slippage_mult = (1 + self.slippage_pct) if order.type == 'buy' else (1 - self.slippage_pct)
                execution_price = current_price * slippage_mult
            else:
                execution_price = order.price or current_price
            
            base_currency = order.pair.split('_')[0]
            quote_currency = order.pair.split('_')[1]
            
            # Execute trade
            if order.type == 'buy':
                cost = order.volume * execution_price
                fee = cost * 0.0026  # Kraken maker/taker fee
                total_cost = cost + fee
                
                if self.paper_balance.get(quote_currency, 0) < total_cost:
                    logger.error(f" Insufficient {quote_currency}: need {total_cost:.2f}, have {self.paper_balance.get(quote_currency, 0):.2f}")
                    return {
                        'success': False,
                        'error': f'Insufficient {quote_currency} balance',
                        'order_id': None,
                        'required': total_cost,
                        'available': self.paper_balance.get(quote_currency, 0)
                    }
                
                self.paper_balance[quote_currency] -= total_cost
                self.paper_balance[base_currency] = self.paper_balance.get(base_currency, 0) + order.volume
                
                logger.info(f" BUY executed:")
                logger.info(f"   Cost: ${cost:.2f}")
                logger.info(f"   Fee: ${fee:.4f}")
                logger.info(f"   Total: ${total_cost:.2f}")
                
            else:  # sell
                if self.paper_balance.get(base_currency, 0) < order.volume:
                    logger.error(f" Insufficient {base_currency}: need {order.volume:.8f}, have {self.paper_balance.get(base_currency, 0):.8f}")
                    return {
                        'success': False,
                        'error': f'Insufficient {base_currency} balance',
                        'order_id': None,
                        'required': order.volume,
                        'available': self.paper_balance.get(base_currency, 0)
                    }
                
                self.paper_balance[base_currency] -= order.volume
                revenue = order.volume * execution_price
                fee = revenue * 0.0026
                net_revenue = revenue - fee
                self.paper_balance[quote_currency] = self.paper_balance.get(quote_currency, 0) + net_revenue
                
                logger.info(f" SELL executed:")
                logger.info(f"   Revenue: ${revenue:.2f}")
                logger.info(f"   Fee: ${fee:.4f}")
                logger.info(f"   Net: ${net_revenue:.2f}")
            
            # Record trade
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
                'status': 'closed',
                'exchange': 'paper',
                'mode': 'paper'
            }
            
            self.paper_trades.append(trade_record)
            
            # Callbacks
            for callback in self.order_callbacks:
                try:
                    callback(trade_record)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            logger.info(f"\n Paper order COMPLETED: {order_id}")
            logger.info(f"   {order.type.upper()} {order.volume:.8f} {base_currency} @ ${execution_price:.2f}")
            logger.info(f"   New balance: {base_currency}={self.paper_balance.get(base_currency, 0):.8f}, {quote_currency}=${self.paper_balance.get(quote_currency, 0):.2f}")
            logger.info(f"{'='*60}\n")
            
            return {
                'success': True,
                'order_id': order_id,
                'execution_price': execution_price,
                'volume': order.volume,
                'fee': trade_record['fee'],
                'timestamp': datetime.now(),
                'mode': 'paper'
            }
    
    def _place_live_order(self, order: KrakenOrder) -> Dict[str, Any]:
        """Execute LIVE order on Kraken"""
        if not self.api or not self.has_credentials:
            logger.error(" Kraken API credentials required for live trading!")
            return {
                'success': False,
                'error': 'Kraken API credentials required for live trading',
                'order_id': None
            }
        
        try:
            # Convert to Kraken pair format
            kraken_pair = self.KRAKEN_PAIR_MAPPING.get(order.pair)
            
            if not kraken_pair:
                logger.error(f" Unknown pair: {order.pair}")
                logger.error(f" Available pairs: {list(self.KRAKEN_PAIR_MAPPING.keys())}")
                return {
                    'success': False,
                    'error': f'Unknown pair: {order.pair}',
                    'order_id': None
                }
            
            # Build order parameters
            params = {
                'pair': kraken_pair,
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
            
            # Execute on Kraken
            logger.info(f" Executing LIVE order on Kraken...")
            logger.info(f"   {order.type.upper()} {order.volume} {kraken_pair} ({order.ordertype})")
            logger.info(f"   Parameters: {params}")
            
            result = self.api.query_private('AddOrder', params)
            
            if result.get('error'):
                error_msg = ', '.join(result['error']) if isinstance(result['error'], list) else str(result['error'])
                logger.error(f" Kraken order error: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'order_id': None,
                    'exchange': 'kraken'
                }
            
            order_info = result.get('result', {})
            txids = order_info.get('txid', [])
            order_id = txids[0] if txids else None
            
            logger.info(f" Kraken order placed successfully!")
            logger.info(f"   Order ID: {order_id}")
            logger.info(f"   Description: {order_info.get('descr', {})}")
            logger.info(f"{'='*60}\n")
            
            return {
                'success': True,
                'order_id': order_id,
                'description': order_info.get('descr', {}),
                'timestamp': datetime.now(),
                'exchange': 'kraken',
                'mode': 'live'
            }
            
        except Exception as e:
            logger.error(f" Error placing Kraken order: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'order_id': None,
                'exchange': 'kraken'
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order"""
        if self.mode == 'paper':
            logger.info(f" Paper order {order_id} cancelled")
            return {
                'success': True,
                'message': f'Paper order {order_id} cancelled'
            }
        else:
            if not self.has_credentials:
                return {
                    'success': False,
                    'error': 'Credentials required'
                }
            
            try:
                result = self.api.query_private('CancelOrder', {'txid': order_id})
                
                if result.get('error'):
                    return {
                        'success': False,
                        'error': result['error']
                    }
                
                logger.info(f" Kraken order {order_id} cancelled")
                
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
        """Get account balance"""
        if self.mode == 'paper':
            return self._get_paper_balance()
        else:
            return self._get_live_balance()
    
    def _get_paper_balance(self) -> Dict[str, KrakenBalance]:
        """Get paper balance"""
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
        """Get live balance from Kraken"""
        if not self.api or not self.has_credentials:
            logger.error("Kraken credentials required")
            return {}
        
        try:
            result = self.api.query_private('Balance')
            
            if result.get('error'):
                logger.error(f"Error getting balance: {result['error']}")
                return {}
            
            balances = {}
            for currency, amount in result.get('result', {}).items():
                # Standardize currency names
                if currency == 'XXBT':
                    currency = 'BTC'
                elif currency == 'XETH':
                    currency = 'ETH'
                elif currency.startswith('Z'):
                    currency = currency[1:]
                elif currency.startswith('X'):
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
            logger.error(f"Error getting balance: {e}")
            return {}
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get open orders"""
        if self.mode == 'paper':
            return []
        
        if not self.has_credentials:
            return []
        
        try:
            result = self.api.query_private('OpenOrders')
            
            if result.get('error'):
                logger.error(f"Error getting open orders: {result['error']}")
                return []
            
            return result.get('result', {}).get('open', {})
            
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def get_closed_orders(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get closed orders"""
        if self.mode == 'paper':
            return self.paper_trades[-count:]
        
        if not self.has_credentials:
            return []
        
        try:
            result = self.api.query_private('ClosedOrders', {'count': count})
            
            if result.get('error'):
                logger.error(f"Error getting closed orders: {result['error']}")
                return []
            
            return result.get('result', {}).get('closed', {})
            
        except Exception as e:
            logger.error(f"Error getting closed orders: {e}")
            return []
    
    # ================== COMPATIBILITY METHODS (FOR live_trader.py) ==================
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        COMPATIBILITY METHOD - DO NOT USE
        
        This method exists only for compatibility with live_trader.py
        Use BinanceDataConnector.get_current_price() instead!
        
        Kraken connector is for TRADING only, not data fetching.
        """
        logger.warning(f" get_current_price() called on Kraken connector!")
        logger.warning(f" Use BinanceDataConnector.get_current_price() instead!")
        logger.warning(f" Kraken is for TRADING only, not data fetching!")
        
        # Return None to force proper implementation
        return None
    
    def load_existing_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """
        COMPATIBILITY METHOD - DO NOT USE
        
        This method exists only for compatibility with live_trader.py
        Use BinanceDataConnector.load_existing_data() instead!
        
        Kraken connector is for TRADING only, not data fetching.
        """
        logger.warning(f" load_existing_data() called on Kraken connector!")
        logger.warning(f" Use BinanceDataConnector.load_existing_data() instead!")
        logger.warning(f" Kraken is for TRADING only, not data fetching!")
        
        # Return empty DataFrame to force proper implementation
        return pd.DataFrame()
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        COMPATIBILITY METHOD - DO NOT USE
        
        This method exists only for compatibility with live_trader.py
        Use BinanceDataConnector.get_latest_candle() instead!
        
        Kraken connector is for TRADING only, not data fetching.
        """
        logger.warning(f" get_latest_candle() called on Kraken connector!")
        logger.warning(f" Use BinanceDataConnector.get_latest_candle() instead!")
        logger.warning(f" Kraken is for TRADING only, not data fetching!")
        
        # Return None to force proper implementation
        return None
    
    # ================== UTILITIES ==================
    
    def register_order_callback(self, callback: Any) -> None:
        """Register callback for order events"""
        self.order_callbacks.append(callback)
    
    def get_paper_trade_history(self) -> List[Dict[str, Any]]:
        """Get paper trading history"""
        return self.paper_trades
    
    def get_paper_performance(self) -> Dict[str, Any]:
        """Get paper trading performance"""
        if not self.paper_trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'total_fees': 0,
                'current_balance_usdt': self.paper_balance.get('USDT', 0)
            }
        
        total_fees = sum(trade['fee'] for trade in self.paper_trades)
        
        return {
            'total_trades': len(self.paper_trades),
            'total_fees': total_fees,
            'current_balance_usdt': self.paper_balance.get('USDT', 0),
            'current_balance': self.paper_balance,
            'trades': self.paper_trades
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status"""
        return {
            'exchange': 'Kraken',
            'purpose': 'Trading Only (Use Binance for data)',
            'mode': self.mode,
            'api_connected': self.api is not None,
            'has_credentials': self.has_credentials,
            'paper_balance': self.paper_balance if self.mode == 'paper' else None,
            'paper_trades_count': len(self.paper_trades) if self.mode == 'paper' else 0,
            'timestamp': datetime.now()
        }


if __name__ == "__main__":
    load_dotenv()
    
    print("\n" + "="*60)
    print(" KRAKEN CONNECTOR TEST")
    print("="*60)
    
    # Test paper trading
    print("\n Testing Paper Trading...")
    connector = KrakenConnector(
        api_key=None,
        api_secret=None,
        mode='paper'
    )
    
    print(f"\n Initial Paper Balance:")
    for currency, amount in connector.paper_balance.items():
        if amount > 0:
            print(f"   {currency}: {amount}")
    
    # Test BUY order
    print("\n Testing BUY order...")
    order = KrakenOrder(
        pair='BTC_USDT',
        type='buy',
        ordertype='market',
        volume=0.001
    )
    
    result = connector.place_order(order, current_price=89500.0)
    print(f"\n Order result:")
    for key, value in result.items():
        print(f"   {key}: {value}")
    
    # Check balance after buy
    balances = connector.get_account_balance()
    print(f"\n Balance after BUY:")
    for currency, balance in balances.items():
        if balance.balance > 0:
            print(f"   {currency}: {balance.balance:.8f}")
    
    # Test SELL order
    print("\n Testing SELL order...")
    sell_order = KrakenOrder(
        pair='BTC_USDT',
        type='sell',
        ordertype='market',
        volume=0.001
    )
    
    result = connector.place_order(sell_order, current_price=90000.0)
    print(f"\n Order result:")
    for key, value in result.items():
        print(f"   {key}: {value}")
    
    # Final balance
    balances = connector.get_account_balance()
    print(f"\n Final Balance:")
    for currency, balance in balances.items():
        if balance.balance > 0:
            print(f"   {currency}: {balance.balance:.8f}")
    
    # Performance
    performance = connector.get_paper_performance()
    print(f"\n Performance Summary:")
    print(f"   Total Trades: {performance['total_trades']}")
    print(f"   Total Fees: ${performance['total_fees']:.4f}")
    print(f"   Final USDT: ${performance['current_balance_usdt']:.2f}")
    
    # Status
    status = connector.get_status()
    print(f"\n Connector Status:")
    for key, value in status.items():
        if key != 'timestamp' and key != 'paper_balance':
            print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print(" TEST COMPLETE")
    print("="*60 + "\n")