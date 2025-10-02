"""
dashboard/components/data_center.py
Data Intelligence Center - Connected to Backend with Original UI Design
"""

from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import yaml
from dash.exceptions import PreventUpdate
import json

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import backend modules
from src.data.data_manager import DataManager
from src.data.validator import DataValidator
from src.data.kraken_connector import KrakenConnector
from src.features.feature_engineer import FeatureEngineer

class DataCenter:
    """Data Intelligence Center with full backend integration"""
    
    def __init__(self, config_path=None):
        """Initialize Data Center with backend connections"""
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(project_root, 'config.yaml')
        
        self.config = self._load_config(config_path)
        
        # Available assets (15 cryptocurrencies) - matching your original UI
        self.assets = [
            {'label': 'Bitcoin (BTC)', 'value': 'BTC_USDT'},
            {'label': 'Ethereum (ETH)', 'value': 'ETH_USDT'},
            {'label': 'Solana (SOL)', 'value': 'SOL_USDT'},
            {'label': 'Cardano (ADA)', 'value': 'ADA_USDT'},
            {'label': 'Polkadot (DOT)', 'value': 'DOT_USDT'},
            {'label': 'Avalanche (AVAX)', 'value': 'AVAX_USDT'},
            {'label': 'Polygon (MATIC)', 'value': 'MATIC_USDT'},
            {'label': 'Chainlink (LINK)', 'value': 'LINK_USDT'},
            {'label': 'Uniswap (UNI)', 'value': 'UNI_USDT'},
            {'label': 'Cosmos (ATOM)', 'value': 'ATOM_USDT'},
            {'label': 'Ripple (XRP)', 'value': 'XRP_USDT'},
            {'label': 'Litecoin (LTC)', 'value': 'LTC_USDT'},
            {'label': 'Algorand (ALGO)', 'value': 'ALGO_USDT'},
            {'label': 'Binance (BNB)', 'value': 'BNB_USDT'},  # Changed from NEAR to match your list
            {'label': 'Dogecoin (DOGE)', 'value': 'DOGE_USDT'}
        ]
        
        # Available timeframes including 30m
        self.timeframes = [
            {'value': '1m', 'label': '1M'},
            {'value': '5m', 'label': '5M'},
            {'value': '15m', 'label': '15M'},
            {'value': '30m', 'label': '30M'},  # Added 30m
            {'value': '1h', 'label': '1H'},
            {'value': '4h', 'label': '4H'},
            {'value': '1d', 'label': '1D'}
        ]
        
        # Initialize backend components
        self.data_manager = DataManager()
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        
        # Initialize Kraken connector
        api_key = self.config.get('kraken', {}).get('api_key', None)
        api_secret = self.config.get('kraken', {}).get('api_secret', None)
        
        self.kraken_connector = KrakenConnector(
            api_key=api_key,
            api_secret=api_secret,
            mode='paper',
            data_path=os.path.join(project_root, 'data', 'raw'),
            update_existing_data=True
        )
        
        # Cache for loaded data
        self.data_cache = {}
        self.quality_reports = {}
        
        # Current selection
        self.current_asset = 'BTC_USDT'
        self.current_timeframe = '1h'
        
        # Data path
        self.data_path = os.path.join(project_root, 'data', 'raw')
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {
                'data': {
                    'path': './data/raw/',
                    'database': './data/trading_bot.db'
                },
                'kraken': {
                    'api_key': None,
                    'api_secret': None
                },
                'trading': {
                    'initial_capital': 10000,
                    'max_positions': 5,
                    'risk_per_trade': 0.02
                }
            }
    
    def create_layout(self):
        """Create the redesigned Data Center layout matching original UI"""
        return html.Div([
            # Page Header
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-area", style={'fontSize': '24px', 'marginRight': '1rem', 'color': '#008394'}),
                    html.H1('Data Intelligence Center', style={
                        'fontSize': '2rem',
                        'fontWeight': '700',
                        'background': 'linear-gradient(45deg, #008394, #ffffff)',
                        'backgroundClip': 'text',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent',
                        'letterSpacing': '1px',
                        'margin': 0
                    })
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                
                # Live indicator
                html.Div([
                    html.Div(style={
                        'width': '8px',
                        'height': '8px',
                        'borderRadius': '50%',
                        'background': '#00ff88',
                        'boxShadow': '0 0 10px rgba(0, 255, 136, 0.8)',
                        'marginRight': '0.5rem'
                    }),
                    html.Span('LIVE DATA', style={
                        'color': '#00ff88',
                        'fontSize': '0.875rem',
                        'fontWeight': '600',
                        'letterSpacing': '1px'
                    })
                ], style={'display': 'flex', 'alignItems': 'center', 'position': 'absolute', 'right': '2rem'})
            ], style={
                'padding': '2rem',
                'borderBottom': '2px solid rgba(0, 131, 148, 0.3)',
                'marginBottom': '2rem',
                'position': 'relative'
            }),
            
            # Main Chart Section
            html.Div([
                # Top Controls Bar
                html.Div([
                    # Left: Crypto Dropdown
                    html.Div([
                        html.Label('Select Asset:', style={'color': '#888', 'fontSize': '0.875rem', 'marginBottom': '0.5rem'}),
                        dcc.Dropdown(
                            id='crypto-dropdown',
                            options=self.assets,
                            value='BTC_USDT',
                            style={
                                'width': '250px',
                                'backgroundColor': 'rgba(0, 0, 0, 0.4)',
                                'borderRadius': '8px'
                            },
                            className='custom-dropdown'
                        )
                    ], style={'display': 'flex', 'flexDirection': 'column'}),
                    
                    # Center: Current Price Display
                    html.Div([
                        html.Div(id='current-price-display', children=[
                            html.Span('BTC/USD', style={'color': '#888', 'fontSize': '0.875rem'}),
                            html.Div([
                                html.Span('$43,247.32', style={
                                    'fontSize': '2.5rem',
                                    'fontWeight': '700',
                                    'color': 'white',
                                    'marginRight': '1rem'
                                }),
                                html.Span('+2.34%', style={
                                    'fontSize': '1.25rem',
                                    'fontWeight': '600',
                                    'color': '#00ff88'
                                })
                            ])
                        ], style={'textAlign': 'center'})
                    ]),
                    
                    # Right: Chart Type Selector
                    html.Div([
                        html.Label('Chart Type:', style={'color': '#888', 'fontSize': '0.875rem', 'marginBottom': '0.5rem'}),
                        html.Div([
                            html.Button('Candlestick', id='chart-type-candle', className='chart-type-btn active'),
                            html.Button('Line', id='chart-type-line', className='chart-type-btn'),
                            html.Button('Area', id='chart-type-area', className='chart-type-btn')
                        ], style={'display': 'flex', 'gap': '0.5rem'})
                    ], style={'display': 'flex', 'flexDirection': 'column'})
                ], style={
                    'display': 'flex',
                    'justifyContent': 'space-between',
                    'alignItems': 'flex-end',
                    'marginBottom': '2rem',
                    'padding': '1rem 2rem',
                    'background': 'rgba(0, 0, 0, 0.3)',
                    'borderRadius': '12px',
                    'border': '1px solid rgba(0, 131, 148, 0.2)'
                }),
                
                # Timeframe Bar
                html.Div([
                    html.Div([
                        html.Button(
                            tf['label'], 
                            id=f'tf-btn-{tf["value"]}',
                            className='timeframe-btn active' if tf['value'] == '1h' else 'timeframe-btn',
                            style={'flex': '1'},
                            n_clicks=0
                        ) for tf in self.timeframes
                    ], style={
                        'display': 'flex',
                        'gap': '0.5rem',
                        'marginBottom': '1.5rem'
                    })
                ]),

                # Navigation controls for sliding window
                html.Div([
                    html.Button([
                        html.I(className="fas fa-chevron-left", style={'marginRight': '0.5rem'}),
                        'Previous'
                    ], id='prev-window-btn', className='nav-btn'),
                    
                    html.Span(id='data-range-display', style={
                        'margin': '0 1rem',
                        'color': '#888',
                        'fontSize': '0.875rem'
                    }),
                    
                    html.Button([
                        'Next',
                        html.I(className="fas fa-chevron-right", style={'marginLeft': '0.5rem'})
                    ], id='next-window-btn', className='nav-btn')
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'marginTop': '1rem'
                }),
                
                # Main Chart with Volume
                html.Div([
                    dcc.Graph(
                        id='main-price-volume-chart',
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'crypto_chart',
                                'height': 800,
                                'width': 1400,
                                'scale': 1
                            }
                        },
                        style={'height': '600px'}
                    )
                ], style={
                    'background': 'rgba(0, 0, 0, 0.4)',
                    'borderRadius': '12px',
                    'padding': '1rem',
                    'border': '1px solid rgba(0, 131, 148, 0.2)',
                    'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.5)'
                }),
                
                # Bottom Statistics Bar
                html.Div([
                    # 24h Volume
                    html.Div([
                        html.Span('24h Volume', style={'color': '#888', 'fontSize': '0.875rem'}),
                        html.Div('--', id='volume-24h', style={
                            'fontSize': '1.25rem',
                            'fontWeight': '600',
                            'color': '#008394'
                        })
                    ], style={'flex': '1', 'textAlign': 'center'}),
                    
                    # 24h High
                    html.Div([
                        html.Span('24h High', style={'color': '#888', 'fontSize': '0.875rem'}),
                        html.Div('--', id='high-24h', style={
                            'fontSize': '1.25rem',
                            'fontWeight': '600',
                            'color': '#00ff88'
                        })
                    ], style={'flex': '1', 'textAlign': 'center'}),
                    
                    # 24h Low
                    html.Div([
                        html.Span('24h Low', style={'color': '#888', 'fontSize': '0.875rem'}),
                        html.Div('--', id='low-24h', style={
                            'fontSize': '1.25rem',
                            'fontWeight': '600',
                            'color': '#ff4444'
                        })
                    ], style={'flex': '1', 'textAlign': 'center'}),
                    
                    # Market Cap
                    html.Div([
                        html.Span('Market Cap', style={'color': '#888', 'fontSize': '0.875rem'}),
                        html.Div('--', id='market-cap', style={
                            'fontSize': '1.25rem',
                            'fontWeight': '600',
                            'color': '#B98544'
                        })
                    ], style={'flex': '1', 'textAlign': 'center'}),
                    
                    # RSI
                    html.Div([
                        html.Span('RSI (14)', style={'color': '#888', 'fontSize': '0.875rem'}),
                        html.Div('--', id='rsi-value', style={
                            'fontSize': '1.25rem',
                            'fontWeight': '600',
                            'color': '#fbbf24'
                        })
                    ], style={'flex': '1', 'textAlign': 'center'})
                ], style={
                    'display': 'flex',
                    'justifyContent': 'space-around',
                    'marginTop': '2rem',
                    'padding': '1.5rem',
                    'background': 'rgba(0, 0, 0, 0.3)',
                    'borderRadius': '12px',
                    'border': '1px solid rgba(0, 131, 148, 0.2)'
                }),
                
                # Additional Info Cards
                html.Div([
                    # Data Quality Card
                    html.Div([
                        html.Div([
                            html.H3('Data Quality', style={'color': 'white', 'fontSize': '1rem', 'marginBottom': '1rem'}),
                            html.Div([
                                html.Div([
                                    html.Span('Status:', style={'color': '#888', 'fontSize': '0.875rem'}),
                                    html.Span(' --', id='data-quality-status', style={'color': '#00ff88', 'fontWeight': '600'})
                                ]),
                                html.Div([
                                    html.Span('Missing:', style={'color': '#888', 'fontSize': '0.875rem'}),
                                    html.Span(' --', id='data-missing', style={'color': '#008394', 'fontWeight': '600'})
                                ]),
                                html.Div([
                                    html.Span('Last Update:', style={'color': '#888', 'fontSize': '0.875rem'}),
                                    html.Span(' --', id='data-last-update', style={'color': '#B98544', 'fontWeight': '600'})
                                ])
                            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '0.5rem'})
                        ])
                    ], className='trading-card', style={'flex': '1'}),
                    
                    # Technical Indicators Card
                    html.Div([
                        html.Div([
                            html.H3('Technical Indicators', style={'color': 'white', 'fontSize': '1rem', 'marginBottom': '1rem'}),
                            html.Div([
                                html.Div([
                                    html.Span('MACD:', style={'color': '#888', 'fontSize': '0.875rem'}),
                                    html.Span(' --', id='macd-signal', style={'color': '#00ff88', 'fontWeight': '600'})
                                ]),
                                html.Div([
                                    html.Span('BB Position:', style={'color': '#888', 'fontSize': '0.875rem'}),
                                    html.Span(' --', id='bb-position', style={'color': '#fbbf24', 'fontWeight': '600'})
                                ]),
                                html.Div([
                                    html.Span('Trend:', style={'color': '#888', 'fontSize': '0.875rem'}),
                                    html.Span(' --', id='trend-direction', style={'color': '#00ff88', 'fontWeight': '600'})
                                ])
                            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '0.5rem'})
                        ])
                    ], className='trading-card', style={'flex': '1'}),
                    
                    # Export Controls Card
                    html.Div([
                        html.Div([
                            html.H3('Data Actions', style={'color': 'white', 'fontSize': '1rem', 'marginBottom': '1rem'}),
                            html.Div([
                                html.Button([
                                    html.I(className="fas fa-download", style={'marginRight': '0.5rem'}),
                                    'Fill Gaps'
                                ], id='fill-gaps-btn', className='action-btn-primary', style={'width': '100%', 'marginBottom': '0.5rem'}),
                                html.Button([
                                    html.I(className="fas fa-wrench", style={'marginRight': '0.5rem'}),
                                    'Fix Data'
                                ], id='fix-data-btn', className='action-btn-secondary', style={'width': '100%'})
                            ])
                        ])
                    ], className='trading-card', style={'flex': '1'})
                ], style={
                    'display': 'flex',
                    'gap': '1.5rem',
                    'marginTop': '2rem'
                })
                
            ], style={'maxWidth': '1600px', 'margin': '0 auto', 'padding': '0 2rem 2rem 2rem'}),
            
            # Hidden stores for data
            dcc.Store(id='current-data-store'),
            dcc.Store(id='quality-report-store'),
            dcc.Store(id='window-state-store'),
            
            # Interval for auto-refresh
            dcc.Interval(id='data-refresh-interval', interval=60000)  # 1 minute
        ])
    
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load data from CSV files - Fixed to handle subdirectory structure"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache first
        if cache_key in self.data_cache:
            print(f" Found {symbol} {timeframe} in cache")
            return self.data_cache[cache_key]
        
        # Build the correct file path with timeframe subdirectory
        filename = f"{symbol}_{timeframe}.csv"
        file_path = os.path.join(self.data_path, timeframe, filename)  # Added timeframe subdirectory
        
        print(f"Looking for file: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        if os.path.exists(file_path):
            try:
                # Load directly from CSV file
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                print(f" Loaded {len(df)} rows for {symbol} from {file_path}")
                self.data_cache[cache_key] = df
                return df
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
        
        # If direct load fails, try with DataManager (it might have different logic)
        formatted_symbol = symbol.replace('_', '/')
        print(f"Trying DataManager with {formatted_symbol} {timeframe}")
        
        df = self.data_manager.load_existing_data(formatted_symbol, timeframe)
        
        if df.empty:
            print(f" No data found for {symbol} {timeframe}")
            # List available files in the timeframe directory
            timeframe_path = os.path.join(self.data_path, timeframe)
            if os.path.exists(timeframe_path):
                files = os.listdir(timeframe_path)
                print(f"Available files in {timeframe_path}: {files[:5]}...")  # Show first 5
        else:
            print(f" Loaded {len(df)} rows for {symbol} via DataManager")
            self.data_cache[cache_key] = df
        
        return df
    
    def validate_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """Validate data quality using validator.py"""
        validation_result = self.validator.validate_and_fix(
            df, 
            symbol=symbol,
            timeframe=timeframe,
            auto_fix=False,
            comprehensive=True
        )
        
        # Calculate date range
        date_range = 'N/A'
        if not df.empty:
            start_date = df.index[0].strftime('%Y-%m-%d')
            end_date = df.index[-1].strftime('%Y-%m-%d')
            date_range = f"{start_date} to {end_date}"
        
        # Calculate average volume
        avg_volume = 'N/A'
        if not df.empty and 'volume' in df.columns:
            avg_vol = df['volume'].mean()
            if avg_vol > 1e6:
                avg_volume = f"{avg_vol/1e6:.2f}M"
            elif avg_vol > 1e3:
                avg_volume = f"{avg_vol/1e3:.2f}K"
            else:
                avg_volume = f"{avg_vol:.2f}"
        
        # Convert to dict for UI
        quality_report = {
            'is_valid': validation_result.is_valid,
            'quality_score': validation_result.quality_score,
            'errors': validation_result.errors,
            'warnings': validation_result.warnings,
            'statistics': validation_result.statistics,
            'total_rows': len(df),
            'missing_candles': validation_result.statistics.get('missing_candles', 0),
            'outliers': validation_result.statistics.get('outliers_detected', 0),
            'gaps': validation_result.statistics.get('gaps_found', 0),
            'date_range': date_range,
            'avg_volume': avg_volume
        }
        
        return quality_report
    
    def detect_gaps(self, df: pd.DataFrame, timeframe: str) -> list:
        """Detect gaps in data"""
        if df.empty:
            return []
        
        # Get expected frequency
        freq_map = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D'
        }
        
        freq = freq_map.get(timeframe, '1H')
        
        # Create expected date range
        expected_range = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            freq=freq
        )
        
        # Find missing timestamps
        missing = expected_range.difference(df.index)
        
        # Group consecutive gaps
        gaps = []
        if len(missing) > 0:
            missing_list = missing.tolist()
            current_gap = [missing_list[0]]
            
            for i in range(1, len(missing_list)):
                time_diff = (missing_list[i] - missing_list[i-1]).total_seconds()
                expected_diff = pd.Timedelta(freq).total_seconds()
                
                if time_diff <= expected_diff * 1.5:
                    current_gap.append(missing_list[i])
                else:
                    gaps.append({
                        'start': current_gap[0],
                        'end': current_gap[-1],
                        'count': len(current_gap)
                    })
                    current_gap = [missing_list[i]]
            
            if current_gap:
                gaps.append({
                    'start': current_gap[0],
                    'end': current_gap[-1],
                    'count': len(current_gap)
                })
        
        return gaps