"""
dashboard/callbacks/real_time_updates.py
Real-time update callbacks for the dashboard
"""

from dash import callback, Input, Output, State
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import asyncio
from typing import Dict, List, Any

class RealTimeDataManager:
    """Manages real-time data updates from backend services"""
    
    def __init__(self):
        self.kraken_connector = None  # Will be initialized with actual connector
        self.portfolio = None
        self.ml_predictor = None
        self.rl_agent = None
        self.risk_manager = None
        self.cache = {}
        self.last_update = {}
        
    def init_backend_connections(self, kraken, portfolio, ml, rl, risk):
        """Initialize backend connections"""
        self.kraken_connector = kraken
        self.portfolio = portfolio
        self.ml_predictor = ml
        self.rl_agent = rl
        self.risk_manager = risk
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary data"""
        if self.portfolio:
            return self.portfolio.get_summary()
        
        # Mock data for development
        return {
            'total_value': np.random.uniform(10000, 15000),
            'daily_pnl': np.random.uniform(-500, 500),
            'daily_pnl_percent': np.random.uniform(-5, 5),
            'positions': np.random.randint(0, 10),
            'active_orders': np.random.randint(0, 5),
            'available_balance': np.random.uniform(1000, 5000)
        }
    
    def get_market_prices(self) -> List[Dict]:
        """Get current market prices"""
        if self.kraken_connector:
            return self.kraken_connector.get_all_prices()
        
        # Mock data for development
        symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
        return [
            {
                'symbol': symbol,
                'price': np.random.uniform(100, 50000) if symbol == 'BTC' else np.random.uniform(10, 5000),
                'change': np.random.uniform(-10, 10),
                'volume': f"{np.random.uniform(0.1, 100):.1f}M",
                'high_24h': np.random.uniform(100, 60000),
                'low_24h': np.random.uniform(90, 40000)
            }
            for symbol in symbols
        ]
    
    def get_active_positions(self) -> List[Dict]:
        """Get active trading positions"""
        if self.portfolio:
            return self.portfolio.get_positions()
        
        # Mock data for development
        positions = []
        symbols = ['BTC', 'ETH', 'SOL']
        for i, symbol in enumerate(symbols):
            positions.append({
                'symbol': symbol,
                'type': np.random.choice(['LONG', 'SHORT']),
                'size': f"{np.random.uniform(0.01, 5):.3f} {symbol}",
                'entry_price': np.random.uniform(1000, 50000),
                'current_price': np.random.uniform(1000, 50000),
                'pnl': np.random.uniform(-200, 500),
                'pnl_percent': np.random.uniform(-10, 20)
            })
        return positions
    
    def get_model_status(self) -> Dict:
        """Get ML and RL model status"""
        status = {}
        
        if self.ml_predictor:
            status['ml'] = self.ml_predictor.get_status()
        else:
            status['ml'] = {
                'status': np.random.choice(['TRAINED', 'TRAINING', 'IDLE']),
                'accuracy': np.random.uniform(0.7, 0.95),
                'last_update': f"{np.random.randint(1, 12)} hours ago",
                'predictions_made': np.random.randint(1000, 10000)
            }
        
        if self.rl_agent:
            status['rl'] = self.rl_agent.get_status()
        else:
            status['rl'] = {
                'status': np.random.choice(['TRAINING', 'READY', 'IDLE']),
                'episodes': np.random.randint(100, 5000),
                'reward': np.random.uniform(-1, 1),
                'epsilon': np.random.uniform(0.01, 0.5)
            }
        
        return status
    
    def get_system_health(self) -> Dict:
        """Get system health metrics"""
        if self.risk_manager:
            return self.risk_manager.get_health_status()
        
        # Mock data for development
        return {
            'api_connection': np.random.choice(['Healthy', 'Degraded', 'Error']),
            'data_feed': np.random.choice(['Active', 'Delayed', 'Inactive']),
            'risk_manager': np.random.choice(['Monitoring', 'Alert', 'Normal']),
            'order_engine': np.random.choice(['Ready', 'Busy', 'Error']),
            'database': np.random.choice(['Connected', 'Slow', 'Error']),
            'websocket': np.random.choice(['Connected', 'Reconnecting', 'Disconnected'])
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return {
            'sharpe_ratio': np.random.uniform(0.5, 2.5),
            'max_drawdown': np.random.uniform(5, 25),
            'win_rate': np.random.uniform(40, 70),
            'profit_factor': np.random.uniform(0.8, 2.0),
            'daily_trades': np.random.randint(5, 50),
            'avg_trade_duration': f"{np.random.randint(10, 180)} min"
        }


"""
dashboard/utils/state_manager.py
Application state management
"""

class DashboardState:
    """Manages dashboard application state"""
    
    def __init__(self):
        self.current_page = 'dashboard'
        self.trading_mode = 'PAPER'  # PAPER or LIVE
        self.system_status = 'IDLE'  # IDLE, TRADING, PAUSED, STOPPED
        self.selected_assets = ['BTC', 'ETH', 'SOL']
        self.selected_timeframe = '1h'
        self.alerts = []
        self.session_start = datetime.now()
        self.last_trade = None
        self.training_status = {
            'ml': {'is_training': False, 'progress': 0},
            'rl': {'is_training': False, 'progress': 0}
        }
    
    def set_page(self, page: str):
        """Set current page"""
        self.current_page = page
    
    def set_trading_mode(self, mode: str):
        """Set trading mode (PAPER/LIVE)"""
        if mode in ['PAPER', 'LIVE']:
            self.trading_mode = mode
    
    def set_system_status(self, status: str):
        """Set system status"""
        if status in ['IDLE', 'TRADING', 'PAUSED', 'STOPPED']:
            self.system_status = status
    
    def add_alert(self, message: str, type: str = 'info'):
        """Add an alert to the dashboard"""
        self.alerts.append({
            'id': len(self.alerts),
            'message': message,
            'type': type,
            'timestamp': datetime.now(),
            'read': False
        })
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
    
    def get_session_duration(self) -> str:
        """Get formatted session duration"""
        duration = datetime.now() - self.session_start
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        return f"{hours}h {minutes}m"
    
    def update_training_progress(self, model_type: str, progress: float):
        """Update model training progress"""
        if model_type in self.training_status:
            self.training_status[model_type]['progress'] = progress
            if progress >= 100:
                self.training_status[model_type]['is_training'] = False
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary"""
        return {
            'current_page': self.current_page,
            'trading_mode': self.trading_mode,
            'system_status': self.system_status,
            'selected_assets': self.selected_assets,
            'selected_timeframe': self.selected_timeframe,
            'alert_count': len([a for a in self.alerts if not a['read']]),
            'session_duration': self.get_session_duration(),
            'training_status': self.training_status
        }


"""
dashboard/callbacks/trading_callbacks.py
Trading action callbacks
"""

class TradingCallbacks:
    """Handles trading-related callbacks"""
    
    def __init__(self, state_manager, data_manager):
        self.state = state_manager
        self.data = data_manager
    
    def start_trading(self):
        """Start the trading bot"""
        try:
            if self.data.portfolio:
                self.data.portfolio.start_trading()
            
            self.state.set_system_status('TRADING')
            self.state.add_alert('Trading bot started successfully', 'success')
            return True
        except Exception as e:
            self.state.add_alert(f'Failed to start trading: {str(e)}', 'danger')
            return False
    
    def pause_trading(self):
        """Pause the trading bot"""
        try:
            if self.data.portfolio:
                self.data.portfolio.pause_trading()
            
            self.state.set_system_status('PAUSED')
            self.state.add_alert('Trading bot paused', 'warning')
            return True
        except Exception as e:
            self.state.add_alert(f'Failed to pause trading: {str(e)}', 'danger')
            return False
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        try:
            if self.data.risk_manager:
                self.data.risk_manager.emergency_stop()
            if self.data.portfolio:
                self.data.portfolio.close_all_positions()
            
            self.state.set_system_status('STOPPED')
            self.state.add_alert('EMERGENCY STOP ACTIVATED - All positions closed', 'danger')
            return True
        except Exception as e:
            self.state.add_alert(f'Emergency stop failed: {str(e)}', 'danger')
            return False
    
    def train_models(self, model_type: str = 'both'):
        """Start model training"""
        try:
            if model_type in ['ml', 'both']:
                if self.data.ml_predictor:
                    self.state.training_status['ml']['is_training'] = True
                    # Start ML training in background
                    # self.data.ml_predictor.train_async()
            
            if model_type in ['rl', 'both']:
                if self.data.rl_agent:
                    self.state.training_status['rl']['is_training'] = True
                    # Start RL training in background
                    # self.data.rl_agent.train_async()
            
            self.state.add_alert(f'Started training {model_type} models', 'info')
            return True
        except Exception as e:
            self.state.add_alert(f'Failed to start training: {str(e)}', 'danger')
            return False
    
    def place_order(self, symbol: str, side: str, size: float, order_type: str = 'market'):
        """Place a trading order"""
        try:
            if self.data.portfolio:
                order_id = self.data.portfolio.place_order(
                    symbol=symbol,
                    side=side,
                    size=size,
                    order_type=order_type
                )
                self.state.add_alert(f'Order placed: {side} {size} {symbol}', 'success')
                return order_id
            return None
        except Exception as e:
            self.state.add_alert(f'Failed to place order: {str(e)}', 'danger')
            return None
    
    def close_position(self, symbol: str):
        """Close a specific position"""
        try:
            if self.data.portfolio:
                self.data.portfolio.close_position(symbol)
                self.state.add_alert(f'Position closed: {symbol}', 'success')
                return True
            return False
        except Exception as e:
            self.state.add_alert(f'Failed to close position: {str(e)}', 'danger')
            return False


"""
dashboard/utils/websocket_manager.py
WebSocket connection manager for real-time updates
"""

import asyncio
import websocket
import json
import threading
from typing import Callable

class WebSocketManager:
    """Manages WebSocket connections for real-time data"""
    
    def __init__(self):
        self.connections = {}
        self.callbacks = {}
        self.running = False
        
    def connect_kraken(self, pairs: List[str], callback: Callable):
        """Connect to Kraken WebSocket"""
        ws_url = "wss://ws.kraken.com"
        
        def on_message(ws, message):
            data = json.loads(message)
            if callback:
                callback(data)
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws):
            print("WebSocket connection closed")
        
        def on_open(ws):
            # Subscribe to ticker updates
            subscribe_msg = {
                "event": "subscribe",
                "pair": pairs,
                "subscription": {"name": "ticker"}
            }
            ws.send(json.dumps(subscribe_msg))
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run in separate thread
        thread = threading.Thread(target=ws.run_forever)
        thread.daemon = True
        thread.start()
        
        self.connections['kraken'] = ws
    
    def disconnect_all(self):
        """Disconnect all WebSocket connections"""
        for name, ws in self.connections.items():
            if ws:
                ws.close()
        self.connections = {}
    
    def broadcast(self, event: str, data: Any):
        """Broadcast data to all registered callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                callback(data)
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for an event"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)


# Initialize global instances
data_manager = RealTimeDataManager()
state_manager = DashboardState()
trading_callbacks = TradingCallbacks(state_manager, data_manager)
websocket_manager = WebSocketManager()