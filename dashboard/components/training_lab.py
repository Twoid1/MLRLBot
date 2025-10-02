"""
dashboard/components/training_lab.py
AI Training Laboratory Component for the Rockets Trading Bot
Handles both ML predictor and RL agent training interfaces
Enhanced with all HIGH and MEDIUM priority features
"""

from dash import html, dcc, dash_table, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

class TrainingLab:
    """AI Training Laboratory component for ML and RL model training"""
    
    def __init__(self):
        """Initialize the Training Lab component"""
        self.assets = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.features = self._get_feature_list()
        self.ml_models = ['XGBoost', 'LightGBM', 'Random Forest']
        self.rl_agents = ['Standard DQN', 'Double DQN', 'Dueling DQN']
        
        # Mock training state
        self.ml_training_state = {
            'is_training': False,
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': 100,
            'metrics': {}
        }
        
        self.rl_training_state = {
            'is_training': False,
            'progress': 0,
            'current_episode': 0,
            'total_episodes': 1000,
            'metrics': {}
        }
    
    def _get_feature_list(self):
        """Get list of available features"""
        return {
            'Price-Based': [
                'Returns (1, 5, 10, 20, 50)',
                'Price Ratios (High/Low, Close/Open)',
                'Price Position in Range',
                'Distance from MAs'
            ],
            'Technical Indicators': [
                'RSI (14, 21)',
                'MACD & Signal',
                'Bollinger Bands',
                'ATR',
                'ADX',
                'Stochastic',
                'Williams %R'
            ],
            'Volume': [
                'OBV',
                'Volume SMA Ratio',
                'VWAP',
                'AD Line'
            ],
            'Pattern Recognition': [
                'Candlestick Patterns',
                'Support/Resistance',
                'Trend Detection',
                'Local Extrema'
            ],
            'Multi-Timeframe': [
                'Trend Alignment',
                'RSI Confluence',
                'Volume Patterns',
                'Volatility Regimes'
            ]
        }
    
    def create_layout(self):
        """Create the main training lab layout"""
        return html.Div([
            # Header
            self._create_header(),
            
            # Training Mode Selector
            self._create_mode_selector(),
            
            # Content area that changes based on mode
            html.Div(id='training-content', children=self._create_ml_section()),
            
            # Status Monitor (shared between modes)
            self._create_status_monitor(),
            
            # Model Management
            self._create_model_management()
            
        ], style={'padding': '2rem', 'maxWidth': '1600px', 'margin': '0 auto'})
    
    def _create_header(self):
        """Create the header section"""
        return html.Div([
            html.H1([
                html.I(className="fas fa-flask", style={'marginRight': '1rem', 'color': '#00a3b8'}),
                'AI Training Laboratory'
            ], style={'color': 'white', 'marginBottom': '0.5rem'}),
            html.P('Train and optimize ML predictors and RL agents for multi-asset trading', 
                  style={'color': '#b0b0b0', 'fontSize': '1.125rem'}),
            html.Hr(style={'borderColor': '#4a4a4a', 'marginBottom': '2rem'})
        ])
    
    def _create_mode_selector(self):
        """Create training mode selector tabs"""
        return html.Div([
            html.Button([
                html.I(className="fas fa-chart-line", style={'marginRight': '0.5rem'}),
                'ML Predictor'
            ], id='ml-mode-btn', className='training-mode-btn active'),
            
            html.Button([
                html.I(className="fas fa-robot", style={'marginRight': '0.5rem'}),
                'RL Agent'
            ], id='rl-mode-btn', className='training-mode-btn')
        ], style={
            'display': 'flex',
            'gap': '1rem',
            'marginBottom': '2rem',
            'background': 'rgba(0, 0, 0, 0.3)',
            'padding': '0.5rem',
            'borderRadius': '8px',
            'width': 'fit-content'
        })
    
    def _create_ml_section(self):
        """Create ML training section with all enhancements"""
        return html.Div([
            # Configuration Grid
            html.Div([
                # Left Column - Data Configuration
                html.Div([
                    self._create_enhanced_data_config_card(),
                    self._create_enhanced_feature_selection_card(),
                    self._create_triple_barrier_labeling_card()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'}),
                
                # Right Column - Model Configuration
                html.Div([
                    self._create_ml_model_config_card(),
                    self._create_risk_management_card(),
                    self._create_validation_config_card()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'})
            ], style={'display': 'flex', 'gap': '1.5rem', 'marginBottom': '2rem'}),
            
            # Training Control Panel
            self._create_training_control_panel('ml'),
            
            # Performance Visualization
            self._create_ml_performance_viz()
            
        ], id='ml-training-section')
    
    def _create_rl_section(self):
        """Create RL training section with enhancements"""
        return html.Div([
            # Configuration Grid
            html.Div([
                # Left Column - Environment Configuration
                html.Div([
                    self._create_environment_config_card(),
                    self._create_state_space_display(),
                    self._create_reward_config_card()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'}),
                
                # Right Column - Agent Configuration
                html.Div([
                    self._create_rl_agent_config_card(),
                    self._create_rl_training_params_card(),
                    self._create_validation_config_card()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'})
            ], style={'display': 'flex', 'gap': '1.5rem', 'marginBottom': '2rem'}),
            
            # Training Control Panel
            self._create_training_control_panel('rl'),
            
            # Performance Visualization
            self._create_rl_performance_viz()
            
        ], id='rl-training-section')
    
    def _create_enhanced_data_config_card(self):
        """Enhanced data configuration with multi-asset universal training"""
        return html.Div([
            html.H3('Data Configuration', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Universal Training Toggle - NEW
            html.Div([
                html.Label('Training Strategy', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.RadioItems(
                    id='training-strategy',
                    options=[
                        {'label': '🌍 Universal Multi-Asset (Recommended)', 'value': 'universal'},
                        {'label': '📍 Single Asset', 'value': 'single'}
                    ],
                    value='universal',
                    style={'marginBottom': '1rem', 'color': 'white'}
                ),
                html.Div(id='strategy-info', children=[
                    html.P('✓ Trains on 5+ assets × 5+ timeframes simultaneously', 
                          style={'color': '#00ff88', 'fontSize': '0.85rem', 'margin': '0.25rem 0'}),
                    html.P('✓ Patterns from BTC improve ETH predictions', 
                          style={'color': '#00ff88', 'fontSize': '0.85rem', 'margin': '0.25rem 0'}),
                    html.P('✓ Reduces overfitting through transfer learning', 
                          style={'color': '#00ff88', 'fontSize': '0.85rem', 'margin': '0.25rem 0'})
                ], style={
                    'background': 'rgba(0,255,136,0.1)', 
                    'padding': '0.75rem', 
                    'borderRadius': '4px',
                    'marginBottom': '1rem',
                    'border': '1px solid rgba(0,255,136,0.3)'
                })
            ]),
            
            # Select All Button - NEW
            html.Button('Select All Assets for Universal Training', 
                       id='select-all-assets', 
                       className='control-btn',
                       style={
                           'width': '100%',
                           'marginBottom': '1rem',
                           'background': 'linear-gradient(90deg, #008394 0%, #00a3b8 100%)',
                           'border': 'none',
                           'color': 'white',
                           'padding': '0.5rem',
                           'borderRadius': '4px',
                           'cursor': 'pointer'
                       }),
            
            # Asset Selection
            html.Div([
                html.Label('Select Assets', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Checklist(
                    id='ml-asset-checklist',
                    options=[{'label': asset, 'value': asset} for asset in self.assets],
                    value=['BTC', 'ETH', 'SOL'],
                    inline=True,
                    style={'color': 'white', 'marginBottom': '1rem'}
                )
            ]),
            
            # Timeframe Selection
            html.Div([
                html.Label('Select Timeframes', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Checklist(
                    id='ml-timeframe-checklist',
                    options=[{'label': tf, 'value': tf} for tf in self.timeframes],
                    value=['1h', '4h'],
                    inline=True,
                    style={'color': 'white', 'marginBottom': '1rem'}
                )
            ]),
            
            # Date Range
            html.Div([
                html.Label('Training Date Range', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.DatePickerRange(
                    id='ml-date-range',
                    start_date='2022-01-01',
                    end_date='2024-01-01',
                    display_format='YYYY-MM-DD',
                    style={'marginBottom': '1rem'}
                )
            ])
            
        ], className='training-card')
    
    def _create_enhanced_feature_selection_card(self):
        """Enhanced feature selection showing 100 → 50 features"""
        return html.Div([
            html.H3('Feature Engineering (100 → 50)', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Feature overview
            html.Div([
                html.Div([
                    html.H5('📊 Total Features: 100', style={'color': '#00ff88', 'margin': '0'}),
                    html.H5('🎯 Selected: 50', style={'color': '#00a3b8', 'margin': '0'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '1rem'}),
                
                # Feature importance method
                html.Label('Selection Method', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Dropdown(
                    id='feature-selection-method',
                    options=[
                        {'label': 'XGBoost Feature Importance', 'value': 'xgboost'},
                        {'label': 'Mutual Information', 'value': 'mutual_info'},
                        {'label': 'Random Forest Importance', 'value': 'rf'},
                        {'label': 'SHAP Values', 'value': 'shap'}
                    ],
                    value='xgboost',
                    style={'marginBottom': '1rem'}
                ),
                
                # Feature categories breakdown
                html.Div([
                    self._create_feature_category('Price-Based', 30, 15),
                    self._create_feature_category('Technical Indicators', 50, 25),
                    self._create_feature_category('Pattern Recognition', 15, 7),
                    self._create_feature_category('Multi-Timeframe', 5, 3)
                ])
            ])
        ], className='training-card')
    
    def _create_feature_category(self, name, total, selected):
        """Create feature category progress bar"""
        return html.Div([
            html.Label(f'{name}: {selected}/{total}', 
                      style={'fontSize': '0.85rem', 'color': '#b0b0b0'}),
            html.Div([
                html.Div(style={
                    'width': f'{(selected/total)*100}%',
                    'height': '8px',
                    'background': 'linear-gradient(90deg, #008394 0%, #00a3b8 100%)',
                    'borderRadius': '4px',
                    'transition': 'width 0.3s ease'
                })
            ], style={
                'background': 'rgba(0,0,0,0.3)', 
                'borderRadius': '4px',
                'marginBottom': '0.5rem'
            })
        ])
    
    def _create_triple_barrier_labeling_card(self):
        """Create triple-barrier labeling configuration - NEW"""
        return html.Div([
            html.H3('Triple-Barrier Labeling', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                # Profit Target
                html.Div([
                    html.Label('Profit Target (%)', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                    dcc.Slider(
                        id='profit-barrier',
                        min=0.5, max=5, step=0.1, value=2,
                        marks={i: f'{i}%' for i in range(1, 6)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], style={'marginBottom': '1.5rem'}),
                
                # Stop Loss
                html.Div([
                    html.Label('Stop Loss (%)', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                    dcc.Slider(
                        id='stop-barrier',
                        min=0.5, max=5, step=0.1, value=2,
                        marks={i: f'{i}%' for i in range(1, 6)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], style={'marginBottom': '1.5rem'}),
                
                # Time Horizon
                html.Div([
                    html.Label('Time Horizon (bars)', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                    dcc.Slider(
                        id='time-barrier',
                        min=10, max=100, step=10, value=50,
                        marks={i: str(i) for i in range(10, 101, 20)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], style={'marginBottom': '1rem'}),
                
                # Visual preview
                dcc.Graph(
                    id='barrier-preview',
                    figure=self._create_barrier_preview_chart(),
                    config={'displayModeBar': False},
                    style={'height': '200px'}
                )
            ])
        ], className='training-card')
    
    def _create_barrier_preview_chart(self):
        """Create a visual preview of triple-barrier setup"""
        # Sample price data
        x = list(range(50))
        price = 100 + np.cumsum(np.random.randn(50) * 0.5)
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=x, y=price,
            mode='lines',
            name='Price',
            line=dict(color='#00a3b8', width=2)
        ))
        
        # Upper barrier
        fig.add_trace(go.Scatter(
            x=x, y=[102] * 50,
            mode='lines',
            name='Profit Target',
            line=dict(color='#00ff88', width=1, dash='dash')
        ))
        
        # Lower barrier
        fig.add_trace(go.Scatter(
            x=x, y=[98] * 50,
            mode='lines',
            name='Stop Loss',
            line=dict(color='#ff6b6b', width=1, dash='dash')
        ))
        
        # Time barrier
        fig.add_vline(x=25, line_dash="dash", line_color="#ffd93d", opacity=0.5)
        
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            showlegend=True,
            legend=dict(orientation="h", y=1.1, x=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        return fig
    
    def _create_risk_management_card(self):
        """Create risk management settings card - NEW"""
        return html.Div([
            html.H3('Risk Management Settings', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                # Kelly Criterion
                html.Div([
                    dcc.Checklist(
                        id='use-kelly',
                        options=[{'label': ' Use Kelly Criterion (25% safety factor)', 'value': 'kelly'}],
                        value=['kelly'],
                        style={'marginBottom': '1rem', 'color': 'white'}
                    )
                ]),
                
                # Position sizing
                html.Div([
                    html.Label('Max Position Size (% of capital)', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                    dcc.Slider(
                        id='max-position-size',
                        min=5, max=50, step=5, value=10,
                        marks={i: f'{i}%' for i in [5, 10, 20, 30, 40, 50]},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], style={'marginBottom': '1.5rem'}),
                
                # Stop loss and take profit
                html.Div([
                    html.Div([
                        html.Label('Stop Loss (%)', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                        dcc.Input(
                            id='stop-loss-pct',
                            type='number',
                            value=5,
                            min=1, max=20, step=0.5,
                            style={'width': '100%', 'padding': '0.5rem', 'borderRadius': '4px'}
                        )
                    ], style={'flex': '1', 'marginRight': '1rem'}),
                    
                    html.Div([
                        html.Label('Take Profit (%)', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                        dcc.Input(
                            id='take-profit-pct',
                            type='number',
                            value=10,
                            min=1, max=50, step=0.5,
                            style={'width': '100%', 'padding': '0.5rem', 'borderRadius': '4px'}
                        )
                    ], style={'flex': '1'})
                ], style={'display': 'flex', 'marginBottom': '1.5rem'}),
                
                # Max drawdown limit
                html.Div([
                    html.Label('Max Drawdown Limit (%)', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                    dcc.Slider(
                        id='max-drawdown',
                        min=5, max=30, step=5, value=15,
                        marks={i: f'{i}%' for i in [5, 10, 15, 20, 25, 30]},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ])
            ])
        ], className='training-card')
    
    def _create_validation_config_card(self):
        """Create validation strategy configuration - NEW"""
        return html.Div([
            html.H3('Validation Strategy', style={'color': 'white', 'marginBottom': '1rem'}),
            
            dcc.RadioItems(
                id='validation-method',
                options=[
                    {'label': '📊 Standard Train/Test Split', 'value': 'standard'},
                    {'label': '🔄 Walk-Forward Analysis (Recommended)', 'value': 'walk_forward'}
                ],
                value='walk_forward',
                style={'marginBottom': '1rem', 'color': 'white'}
            ),
            
            html.Div(id='walk-forward-config', children=[
                html.Label('Number of Splits', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Slider(
                    id='n-splits',
                    min=3, max=10, step=1, value=5,
                    marks={i: str(i) for i in range(3, 11)},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                ),
                html.P('Each split trains on expanding window, tests on next period', 
                      style={'fontSize': '0.85rem', 'color': '#888', 'marginTop': '0.5rem'}),
                
                # Visual representation of walk-forward
                html.Div([
                    html.Div([
                        html.Div(style={
                            'background': 'linear-gradient(90deg, #008394 0%, #008394 60%, #00ff88 60%, #00ff88 80%, rgba(255,255,255,0.1) 80%)',
                            'height': '20px',
                            'borderRadius': '4px',
                            'marginBottom': '0.25rem'
                        }),
                        html.P('Train → Test → Future', 
                              style={'fontSize': '0.75rem', 'color': '#888', 'textAlign': 'center'})
                    ])
                ], style={'marginTop': '1rem'})
            ])
        ], className='training-card')
    
    def _create_state_space_display(self):
        """Create state space configuration display - NEW"""
        return html.Div([
            html.H3('State Space Configuration (63 dims)', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                # Visual breakdown of state dimensions
                html.Div([
                    self._create_state_component('Market Features', 50, '#00ff88'),
                    self._create_state_component('ML Predictions', 3, '#ff6b6b'),
                    self._create_state_component('Position Info', 5, '#4ecdc4'),
                    self._create_state_component('Account State', 5, '#95e1d3')
                ], style={'marginBottom': '1.5rem'}),
                
                # Action space
                html.Div([
                    html.Label('Action Space (3 actions)', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                    html.Div([
                        html.Span('0: Hold/Flat', style={
                            'background': 'rgba(255,255,255,0.1)',
                            'padding': '0.25rem 0.75rem',
                            'borderRadius': '4px',
                            'marginRight': '0.5rem',
                            'fontSize': '0.875rem'
                        }),
                        html.Span('1: Buy/Long', style={
                            'background': 'rgba(0,255,136,0.2)',
                            'padding': '0.25rem 0.75rem',
                            'borderRadius': '4px',
                            'marginRight': '0.5rem',
                            'fontSize': '0.875rem',
                            'color': '#00ff88'
                        }),
                        html.Span('2: Sell/Short', style={
                            'background': 'rgba(255,107,107,0.2)',
                            'padding': '0.25rem 0.75rem',
                            'borderRadius': '4px',
                            'fontSize': '0.875rem',
                            'color': '#ff6b6b'
                        })
                    ])
                ])
            ])
        ], className='training-card')
    
    def _create_state_component(self, name, dims, color):
        """Create a visual component for state space"""
        return html.Div([
            html.Div([
                html.Label(name, style={'color': '#b0b0b0', 'fontSize': '0.85rem'}),
                html.Span(f'{dims} dims', style={'color': color, 'fontSize': '0.85rem', 'float': 'right'})
            ]),
            html.Div([
                html.Div(style={
                    'width': f'{(dims/63)*100}%',
                    'height': '12px',
                    'background': f'linear-gradient(90deg, {color} 0%, {color}88 100%)',
                    'borderRadius': '4px'
                })
            ], style={
                'background': 'rgba(0,0,0,0.3)',
                'borderRadius': '4px',
                'marginBottom': '0.75rem'
            })
        ])
    
    # Keep existing methods unchanged
    def _create_ml_model_config_card(self):
        """Create ML model configuration card"""
        return html.Div([
            html.H3('Model Configuration', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                html.Label('Model Type', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Dropdown(
                    id='ml-model-type',
                    options=[{'label': model, 'value': model} for model in self.ml_models],
                    value='XGBoost',
                    style={'marginBottom': '1rem'}
                ),
                
                html.Label('Hyperparameter Optimization', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.RadioItems(
                    id='ml-hyperparam-opt',
                    options=[
                        {'label': 'None', 'value': 'none'},
                        {'label': 'Grid Search', 'value': 'grid'},
                        {'label': 'Random Search', 'value': 'random'},
                        {'label': 'Bayesian', 'value': 'bayesian'}
                    ],
                    value='random',
                    inline=True,
                    style={'color': 'white', 'marginBottom': '1rem'}
                )
            ])
        ], className='training-card')
    
    def _create_environment_config_card(self):
        """Create environment configuration card"""
        return html.Div([
            html.H3('Environment Configuration', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                html.Label('Initial Balance', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Input(
                    id='rl-initial-balance',
                    type='number',
                    value=10000,
                    style={'width': '100%', 'marginBottom': '1rem'}
                ),
                
                html.Label('Commission (%)', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Input(
                    id='rl-commission',
                    type='number',
                    value=0.26,
                    step=0.01,
                    style={'width': '100%', 'marginBottom': '1rem'}
                ),
                
                html.Label('Slippage (%)', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Input(
                    id='rl-slippage',
                    type='number',
                    value=0.1,
                    step=0.01,
                    style={'width': '100%'}
                )
            ])
        ], className='training-card')
    
    def _create_reward_config_card(self):
        """Create reward configuration card"""
        return html.Div([
            html.H3('Reward Configuration', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                html.Label('Reward Function', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Dropdown(
                    id='rl-reward-function',
                    options=[
                        {'label': 'Profit & Loss', 'value': 'pnl'},
                        {'label': 'Sharpe Ratio', 'value': 'sharpe'},
                        {'label': 'Risk-Adjusted Return', 'value': 'risk_adjusted'},
                        {'label': 'Custom Hybrid', 'value': 'hybrid'}
                    ],
                    value='sharpe',
                    style={'marginBottom': '1rem'}
                ),
                
                html.Label('Risk Penalty Weight', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Slider(
                    id='rl-risk-penalty',
                    min=0, max=1, step=0.1, value=0.3,
                    marks={i/10: str(i/10) for i in range(11)},
                    tooltip={'placement': 'bottom'}
                )
            ])
        ], className='training-card')
    
    def _create_rl_agent_config_card(self):
        """Create RL agent configuration card"""
        return html.Div([
            html.H3('Agent Configuration', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                html.Label('Agent Type', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.Dropdown(
                    id='rl-agent-type',
                    options=[{'label': agent, 'value': agent} for agent in self.rl_agents],
                    value='Double DQN',
                    style={'marginBottom': '1rem'}
                ),
                
                html.Label('Network Architecture', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                dcc.RadioItems(
                    id='rl-network-arch',
                    options=[
                        {'label': 'Simple (256-128)', 'value': 'simple'},
                        {'label': 'Standard (256-256-128)', 'value': 'standard'},
                        {'label': 'Deep (512-256-128-64)', 'value': 'deep'}
                    ],
                    value='standard',
                    inline=True,
                    style={'color': 'white'}
                )
            ])
        ], className='training-card')
    
    def _create_rl_training_params_card(self):
        """Create RL training parameters card"""
        return html.Div([
            html.H3('Training Parameters', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                html.Div([
                    html.Label('Episodes', style={'color': '#b0b0b0'}),
                    dcc.Input(id='rl-episodes', type='number', value=1000, style={'width': '100%'})
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                html.Div([
                    html.Label('Batch Size', style={'color': '#b0b0b0'}),
                    dcc.Input(id='rl-batch-size', type='number', value=32, style={'width': '100%'})
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'marginBottom': '1rem'}),
            
            html.Div([
                html.Div([
                    html.Label('Learning Rate', style={'color': '#b0b0b0'}),
                    dcc.Input(id='rl-learning-rate', type='number', value=0.001, step=0.0001, style={'width': '100%'})
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                html.Div([
                    html.Label('Gamma', style={'color': '#b0b0b0'}),
                    dcc.Input(id='rl-gamma', type='number', value=0.99, step=0.01, style={'width': '100%'})
                ], style={'flex': '1'})
            ], style={'display': 'flex'})
        ], className='training-card')
    
    def _create_training_control_panel(self, mode):
        """Create training control panel"""
        return html.Div([
            html.H3('Training Control', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Control buttons
            html.Div([
                html.Button([
                    html.I(className="fas fa-play", style={'marginRight': '0.5rem'}),
                    'Start Training'
                ], id=f'{mode}-start-btn', className='control-btn start'),
                
                html.Button([
                    html.I(className="fas fa-pause", style={'marginRight': '0.5rem'}),
                    'Pause'
                ], id=f'{mode}-pause-btn', className='control-btn pause', disabled=True),
                
                html.Button([
                    html.I(className="fas fa-stop", style={'marginRight': '0.5rem'}),
                    'Stop'
                ], id=f'{mode}-stop-btn', className='control-btn stop', disabled=True),
                
                html.Button([
                    html.I(className="fas fa-redo", style={'marginRight': '0.5rem'}),
                    'Reset'
                ], id=f'{mode}-reset-btn', className='control-btn reset'),
                
                html.Button([
                    html.I(className="fas fa-save", style={'marginRight': '0.5rem'}),
                    'Save Model'
                ], id='save-model-btn', className='control-btn save', 
                   style={'marginLeft': 'auto'})
                
            ], style={'display': 'flex', 'gap': '0.5rem', 'marginBottom': '1.5rem'}),
            
            # Progress bar
            html.Div([
                html.Div([
                    html.Label('Training Progress', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                    html.Span(id=f'{mode}-progress-text', children='0%', 
                             style={'float': 'right', 'color': 'white'})
                ]),
                html.Div([
                    html.Div(id=f'{mode}-progress-bar', style={
                        'width': '0%',
                        'height': '100%',
                        'background': 'linear-gradient(90deg, #008394 0%, #00a3b8 100%)',
                        'borderRadius': '4px',
                        'transition': 'width 0.3s ease'
                    })
                ], style={
                    'width': '100%',
                    'height': '8px',
                    'background': 'rgba(0, 131, 148, 0.2)',
                    'borderRadius': '4px'
                })
            ])
            
        ], style={
            'background': 'rgba(0, 0, 0, 0.3)',
            'border': '1px solid #4a4a4a',
            'borderRadius': '8px',
            'padding': '1.5rem',
            'marginBottom': '2rem'
        })
    
    def _create_ml_performance_viz(self):
        """Create ML performance visualization"""
        return html.Div([
            html.H3('Performance Metrics', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Metrics Grid
            html.Div([
                # Accuracy Chart
                html.Div([
                    dcc.Graph(
                        id='ml-accuracy-chart',
                        figure=self._create_metric_chart('Accuracy', 'Training Accuracy'),
                        config={'displayModeBar': False}
                    )
                ], style={'flex': '1'}),
                
                # Loss Chart
                html.Div([
                    dcc.Graph(
                        id='ml-loss-chart',
                        figure=self._create_metric_chart('Loss', 'Training Loss', ascending=False),
                        config={'displayModeBar': False}
                    )
                ], style={'flex': '1'}),
                
                # F1 Score Chart
                html.Div([
                    dcc.Graph(
                        id='ml-f1-chart',
                        figure=self._create_metric_chart('F1 Score', 'F1 Score'),
                        config={'displayModeBar': False}
                    )
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'gap': '1rem', 'marginBottom': '1.5rem'}),
            
            # Feature Importance
            html.Div([
                html.H4('Top 10 Feature Importance', style={'color': 'white', 'marginBottom': '1rem', 'fontSize': '1rem'}),
                dcc.Graph(
                    id='ml-feature-importance',
                    figure=self._create_feature_importance_chart(),
                    config={'displayModeBar': False}
                )
            ])
        ], className='training-card')
    
    def _create_rl_performance_viz(self):
        """Create RL performance visualization"""
        return html.Div([
            html.H3('Training Performance', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Metrics Grid
            html.Div([
                # Episode Reward Chart
                html.Div([
                    dcc.Graph(
                        id='rl-reward-chart',
                        figure=self._create_metric_chart('Episode Reward', 'Total Reward per Episode'),
                        config={'displayModeBar': False}
                    )
                ], style={'flex': '1'}),
                
                # Loss Chart
                html.Div([
                    dcc.Graph(
                        id='rl-loss-chart',
                        figure=self._create_metric_chart('Loss', 'Q-Network Loss', ascending=False),
                        config={'displayModeBar': False}
                    )
                ], style={'flex': '1'}),
                
                # Win Rate Chart
                html.Div([
                    dcc.Graph(
                        id='rl-winrate-chart',
                        figure=self._create_metric_chart('Win Rate', 'Profitable Trades %'),
                        config={'displayModeBar': False}
                    )
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'gap': '1rem'})
        ], className='training-card')
    
    def _create_status_monitor(self):
        """Create training status monitor"""
        return html.Div([
            html.H3('Training Status Monitor', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                # Live Metrics
                html.Div([
                    self._create_live_metric('Current Epoch', '0/100', 'epoch-metric'),
                    self._create_live_metric('Training Time', '00:00:00', 'time-metric'),
                    self._create_live_metric('ETA', '--:--:--', 'eta-metric'),
                    self._create_live_metric('GPU Memory', '0/8 GB', 'gpu-metric'),
                    self._create_live_metric('Learning Rate', '0.001', 'lr-metric'),
                    self._create_live_metric('Best Score', 'N/A', 'best-metric')
                ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(6, 1fr)', 'gap': '1rem', 'marginBottom': '1.5rem'}),
                
                # Training Log
                html.Div([
                    html.H4('Training Log', style={'color': '#b0b0b0', 'marginBottom': '0.5rem', 'fontSize': '0.875rem'}),
                    html.Div(id='training-log', children=[
                        self._create_log_entry('System', 'Training environment initialized', 'info'),
                        self._create_log_entry('Data', 'Loaded 147,320 samples', 'success'),
                        self._create_log_entry('Model', 'XGBoost model created', 'info'),
                        self._create_log_entry('Ready', 'Waiting for training to start...', 'warning')
                    ], style={
                        'background': 'rgba(0, 0, 0, 0.5)',
                        'border': '1px solid #4a4a4a',
                        'borderRadius': '4px',
                        'padding': '1rem',
                        'height': '150px',
                        'overflowY': 'auto',
                        'fontFamily': 'JetBrains Mono, monospace',
                        'fontSize': '0.875rem'
                    })
                ])
            ])
            
        ], className='training-card')
    
    def _create_model_management(self):
        """Create model management section"""
        return html.Div([
            html.H3('Model Management', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                # Saved Models Table
                html.Div([
                    html.H4('Saved Models', style={'color': '#b0b0b0', 'marginBottom': '1rem', 'fontSize': '1rem'}),
                    dash_table.DataTable(
                        id='saved-models-table',
                        columns=[
                            {'name': 'Name', 'id': 'name'},
                            {'name': 'Type', 'id': 'type'},
                            {'name': 'Accuracy', 'id': 'accuracy'},
                            {'name': 'Date', 'id': 'date'},
                            {'name': 'Status', 'id': 'status'}
                        ],
                        data=[
                            {'name': 'XGBoost_v1', 'type': 'XGBoost', 'accuracy': '82.3%', 'date': '2024-01-15', 'status': 'Production'},
                            {'name': 'LightGBM_v2', 'type': 'LightGBM', 'accuracy': '81.7%', 'date': '2024-01-14', 'status': 'Testing'},
                            {'name': 'DQN_Agent_v3', 'type': 'Double DQN', 'accuracy': '79.5%', 'date': '2024-01-13', 'status': 'Archive'}
                        ],
                        style_cell={
                            'backgroundColor': 'rgba(0, 0, 0, 0.3)',
                            'color': 'white',
                            'border': '1px solid #4a4a4a'
                        },
                        style_header={
                            'backgroundColor': 'rgba(0, 131, 148, 0.3)',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'status', 'filter_query': '{status} = "Production"'},
                                'color': '#00ff88'
                            },
                            {
                                'if': {'column_id': 'status', 'filter_query': '{status} = "Testing"'},
                                'color': '#ffd93d'
                            }
                        ]
                    )
                ])
            ])
        ], className='training-card')
    
    def _create_metric_chart(self, metric_name, title, ascending=True):
        """Create a metric chart"""
        # Generate sample data
        x = list(range(100))
        if ascending:
            y = [np.random.uniform(0.5, 0.9) * (1 + i/100) for i in x]
        else:
            y = [np.random.uniform(0.5, 0.3) * (1 - i/200) for i in x]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=metric_name,
            line=dict(color='#00a3b8', width=2)
        ))
        
        fig.update_layout(
            title=title,
            height=250,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white', size=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        return fig
    
    def _create_feature_importance_chart(self):
        """Create feature importance chart"""
        features = ['RSI_14', 'MACD', 'Volume_Ratio', 'ATR', 'Price_MA_Distance', 
                   'Bollinger_Width', 'OBV', 'ADX', 'Stochastic_K', 'Williams_R']
        importance = [0.15, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Teal',
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            )
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=100, r=20, t=20, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white', size=10),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=False)
        )
        
        return fig
    
    def _create_live_metric(self, label, value, metric_id):
        """Create a live metric display"""
        return html.Div([
            html.Label(label, style={'color': '#b0b0b0', 'fontSize': '0.75rem', 'marginBottom': '0.25rem'}),
            html.Div(id=metric_id, children=value, 
                    style={'color': 'white', 'fontSize': '1.125rem', 'fontWeight': 'bold'})
        ], style={
            'background': 'rgba(0, 0, 0, 0.3)',
            'padding': '0.75rem',
            'borderRadius': '4px',
            'border': '1px solid #4a4a4a',
            'textAlign': 'center'
        })
    
    def _create_log_entry(self, source, message, level='info'):
        """Create a log entry"""
        colors = {
            'info': '#00a3b8',
            'success': '#00ff88',
            'warning': '#ffd93d',
            'error': '#ff6b6b'
        }
        
        return html.Div([
            html.Span(f'[{datetime.now().strftime("%H:%M:%S")}]', 
                     style={'color': '#666', 'marginRight': '0.5rem'}),
            html.Span(f'[{source}]', 
                     style={'color': colors.get(level, '#00a3b8'), 'marginRight': '0.5rem'}),
            html.Span(message, style={'color': '#b0b0b0'})
        ], style={'marginBottom': '0.25rem'})