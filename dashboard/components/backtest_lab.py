"""
dashboard/components/backtest_lab.py
Backtesting Laboratory for the Rockets Trading Bot
Comprehensive strategy testing, walk-forward analysis, and performance evaluation
Ready for backend integration with backtester.py, walk_forward.py, and metrics.py
"""

from dash import html, dcc, dash_table, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

class BacktestLab:
    """Backtesting Laboratory component for strategy testing and analysis"""
    
    def __init__(self):
        """Initialize the Backtesting Laboratory"""
        self.assets = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        # Strategy configurations
        self.strategies = [
            {'label': 'ML Predictor + DQN Agent', 'value': 'ml_dqn'},
            {'label': 'ML Predictor Only', 'value': 'ml_only'},
            {'label': 'DQN Agent Only', 'value': 'dqn_only'},
            {'label': 'Moving Average Crossover', 'value': 'ma_cross'},
            {'label': 'RSI Mean Reversion', 'value': 'rsi_reversion'},
            {'label': 'Bollinger Bands', 'value': 'bb_strategy'},
            {'label': 'Custom Strategy', 'value': 'custom'}
        ]
        
        # Mock backtest results
        self.backtest_results = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0
        }
    
    def create_layout(self):
        """Create the main backtesting laboratory layout"""
        return html.Div([
            # Header
            self._create_header(),
            
            # Strategy Configuration Section
            self._create_strategy_config(),
            
            # Backtest Control Panel
            self._create_control_panel(),
            
            # Results Section
            html.Div([
                # Left Column: Performance Metrics
                html.Div([
                    self._create_metrics_dashboard(),
                    self._create_trade_analysis()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'}),
                
                # Right Column: Visualizations
                html.Div([
                    self._create_equity_curve(),
                    self._create_drawdown_chart(),
                    self._create_returns_distribution()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'})
            ], style={'display': 'flex', 'gap': '1.5rem', 'marginBottom': '2rem'}),
            
            # Walk-Forward Analysis Section
            self._create_walk_forward_section(),
            
            # Strategy Comparison Section
            self._create_strategy_comparison(),
            
            # Trade Log
            self._create_trade_log()
            
        ], style={'padding': '2rem', 'maxWidth': '1600px', 'margin': '0 auto'})
    
    def _create_header(self):
        """Create the header section"""
        return html.Div([
            html.H1([
                html.I(className="fas fa-history", style={'marginRight': '1rem', 'color': '#00a3b8'}),
                'Backtesting Laboratory'
            ], style={'color': 'white', 'marginBottom': '0.5rem'}),
            html.P('Test strategies, analyze performance, and optimize parameters', 
                  style={'color': '#b0b0b0', 'fontSize': '1.125rem'}),
            html.Hr(style={'borderColor': '#4a4a4a', 'marginBottom': '2rem'})
        ])
    
    def _create_strategy_config(self):
        """Create strategy configuration section"""
        return html.Div([
            html.H3('Strategy Configuration', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                # Strategy Selection
                html.Div([
                    html.Label('Select Strategy', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.Dropdown(
                        id='strategy-select',
                        options=self.strategies,
                        value='ml_dqn',
                        style={'background': '#1a1a1a'}
                    )
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                # Asset Selection
                html.Div([
                    html.Label('Assets', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.Dropdown(
                        id='backtest-assets',
                        options=[{'label': asset, 'value': asset} for asset in self.assets],
                        value=['BTC', 'ETH'],
                        multi=True,
                        style={'background': '#1a1a1a'}
                    )
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                # Timeframe
                html.Div([
                    html.Label('Timeframe', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.Dropdown(
                        id='backtest-timeframe',
                        options=[{'label': tf, 'value': tf} for tf in self.timeframes],
                        value='1h',
                        style={'background': '#1a1a1a'}
                    )
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'marginBottom': '1.5rem'}),
            
            # Date Range Selection
            html.Div([
                html.Div([
                    html.Label('Start Date', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.DatePickerSingle(
                        id='backtest-start-date',
                        date='2024-01-01',
                        style={'background': '#1a1a1a'}
                    )
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                html.Div([
                    html.Label('End Date', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.DatePickerSingle(
                        id='backtest-end-date',
                        date='2024-12-31',
                        style={'background': '#1a1a1a'}
                    )
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                html.Div([
                    html.Label('Initial Capital', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.Input(
                        id='initial-capital',
                        type='number',
                        value=10000,
                        style={'width': '100%', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white', 'padding': '0.5rem'}
                    )
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'marginBottom': '1.5rem'}),
            
            # Advanced Settings
            self._create_advanced_settings()
            
        ], className='backtest-card')
    
    def _create_advanced_settings(self):
        """Create advanced settings panel"""
        return html.Details([
            html.Summary('Advanced Settings', style={'color': '#00a3b8', 'cursor': 'pointer', 'marginBottom': '1rem'}),
            
            html.Div([
                # Risk Parameters
                html.Div([
                    html.H4('Risk Parameters', style={'color': 'white', 'fontSize': '1rem', 'marginBottom': '0.5rem'}),
                    
                    html.Div([
                        html.Div([
                            html.Label('Position Size (%)', style={'color': '#b0b0b0', 'fontSize': '0.75rem'}),
                            dcc.Slider(
                                id='position-size-slider',
                                min=1, max=100, step=1, value=10,
                                marks={1: '1%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
                                tooltip={'placement': 'bottom', 'always_visible': True}
                            )
                        ], style={'marginBottom': '1rem'}),
                        
                        html.Div([
                            html.Label('Stop Loss (%)', style={'color': '#b0b0b0', 'fontSize': '0.75rem'}),
                            dcc.Input(
                                id='stop-loss-input',
                                type='number',
                                value=2,
                                min=0.1, max=10, step=0.1,
                                style={'width': '100px', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white'}
                            )
                        ], style={'display': 'inline-block', 'marginRight': '2rem'}),
                        
                        html.Div([
                            html.Label('Take Profit (%)', style={'color': '#b0b0b0', 'fontSize': '0.75rem'}),
                            dcc.Input(
                                id='take-profit-input',
                                type='number',
                                value=5,
                                min=0.1, max=20, step=0.1,
                                style={'width': '100px', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white'}
                            )
                        ], style={'display': 'inline-block'})
                    ])
                ], style={'flex': '1', 'marginRight': '2rem'}),
                
                # ML/RL Parameters
                html.Div([
                    html.H4('ML/RL Parameters', style={'color': 'white', 'fontSize': '1rem', 'marginBottom': '0.5rem'}),
                    
                    html.Div([
                        dcc.Checklist(
                            id='use-ml-predictions',
                            options=[{'label': ' Use ML Predictions', 'value': 'ml'}],
                            value=['ml'],
                            style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}
                        ),
                        
                        dcc.Checklist(
                            id='use-rl-agent',
                            options=[{'label': ' Use RL Agent', 'value': 'rl'}],
                            value=['rl'],
                            style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}
                        ),
                        
                        html.Div([
                            html.Label('Model Confidence Threshold', style={'color': '#b0b0b0', 'fontSize': '0.75rem'}),
                            dcc.Slider(
                                id='confidence-threshold',
                                min=0.5, max=0.95, step=0.05, value=0.7,
                                marks={0.5: '0.5', 0.7: '0.7', 0.9: '0.9'},
                                tooltip={'placement': 'bottom', 'always_visible': True}
                            )
                        ])
                    ])
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'padding': '1rem', 'background': 'rgba(0, 0, 0, 0.3)', 'borderRadius': '8px'})
        ])
    
    def _create_control_panel(self):
        """Create backtest control panel"""
        return html.Div([
            html.Div([
                # Control Buttons
                html.Div([
                    html.Button([
                        html.I(className="fas fa-play", style={'marginRight': '0.5rem'}),
                        'RUN BACKTEST'
                    ], id='run-backtest-btn', className='control-btn-primary'),
                    
                    html.Button([
                        html.I(className="fas fa-pause", style={'marginRight': '0.5rem'}),
                        'PAUSE'
                    ], id='pause-backtest-btn', className='control-btn', disabled=True),
                    
                    html.Button([
                        html.I(className="fas fa-stop", style={'marginRight': '0.5rem'}),
                        'STOP'
                    ], id='stop-backtest-btn', className='control-btn', disabled=True),
                    
                    html.Button([
                        html.I(className="fas fa-sync", style={'marginRight': '0.5rem'}),
                        'RESET'
                    ], id='reset-backtest-btn', className='control-btn')
                ], style={'display': 'flex', 'gap': '1rem', 'flex': '1'}),
                
                # Progress Bar
                html.Div([
                    html.Div('Progress', style={'color': '#b0b0b0', 'fontSize': '0.875rem', 'marginBottom': '0.5rem'}),
                    html.Div([
                        html.Div(id='backtest-progress-bar', style={
                            'width': '0%',
                            'height': '100%',
                            'background': 'linear-gradient(90deg, #008394, #00a3b8)',
                            'borderRadius': '4px',
                            'transition': 'width 0.3s ease'
                        })
                    ], style={
                        'width': '300px',
                        'height': '8px',
                        'background': 'rgba(0, 131, 148, 0.2)',
                        'borderRadius': '4px'
                    }),
                    html.Span('0%', id='backtest-progress-text', style={'color': '#00a3b8', 'fontSize': '0.75rem', 'marginLeft': '0.5rem'})
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1rem'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
            
            # Status Messages
            html.Div(id='backtest-status', children='Ready to start backtesting', style={
                'marginTop': '1rem',
                'padding': '0.75rem',
                'background': 'rgba(0, 131, 148, 0.1)',
                'border': '1px solid rgba(0, 131, 148, 0.3)',
                'borderRadius': '4px',
                'color': '#00a3b8',
                'fontSize': '0.875rem'
            })
        ], className='backtest-card', style={'marginTop': '1.5rem', 'marginBottom': '1.5rem'})
    
    def _create_metrics_dashboard(self):
        """Create metrics dashboard"""
        return html.Div([
            html.H3('Performance Metrics', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Key Metrics Grid
            html.Div([
                self._create_metric_card('Total Return', '0.00%', 'total-return-metric', 'fas fa-chart-line'),
                self._create_metric_card('Sharpe Ratio', '0.00', 'sharpe-ratio-metric', 'fas fa-balance-scale'),
                self._create_metric_card('Max Drawdown', '0.00%', 'max-drawdown-metric', 'fas fa-chart-area'),
                self._create_metric_card('Win Rate', '0.00%', 'win-rate-metric', 'fas fa-trophy'),
                self._create_metric_card('Profit Factor', '0.00', 'profit-factor-metric', 'fas fa-coins'),
                self._create_metric_card('Total Trades', '0', 'total-trades-metric', 'fas fa-exchange-alt')
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(3, 1fr)',
                'gap': '1rem',
                'marginBottom': '1rem'
            }),
            
            # Additional Metrics
            html.Div([
                html.H4('Risk Metrics', style={'color': '#00a3b8', 'fontSize': '1rem', 'marginBottom': '0.5rem'}),
                html.Div([
                    self._create_small_metric('Sortino Ratio', '0.00', 'sortino-metric'),
                    self._create_small_metric('Calmar Ratio', '0.00', 'calmar-metric'),
                    self._create_small_metric('Avg Win/Loss', '0.00', 'avg-win-loss-metric'),
                    self._create_small_metric('Recovery Factor', '0.00', 'recovery-metric')
                ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '0.5rem'})
            ])
        ], className='backtest-card')
    
    def _create_metric_card(self, label, value, metric_id, icon):
        """Create a metric card"""
        return html.Div([
            html.I(className=icon, style={'color': '#00a3b8', 'fontSize': '1.25rem', 'marginBottom': '0.5rem'}),
            html.Div(label, style={'color': '#b0b0b0', 'fontSize': '0.75rem', 'marginBottom': '0.25rem'}),
            html.Div(value, id=metric_id, style={
                'color': 'white',
                'fontSize': '1.25rem',
                'fontWeight': '600',
                'fontFamily': 'JetBrains Mono, monospace'
            })
        ], style={
            'textAlign': 'center',
            'padding': '1rem',
            'background': 'rgba(0, 0, 0, 0.3)',
            'borderRadius': '8px',
            'border': '1px solid rgba(0, 131, 148, 0.2)'
        })
    
    def _create_small_metric(self, label, value, metric_id):
        """Create a small metric display"""
        return html.Div([
            html.Span(label, style={'color': '#b0b0b0', 'fontSize': '0.75rem'}),
            html.Span(value, id=metric_id, style={
                'color': 'white',
                'fontSize': '0.875rem',
                'fontWeight': '600',
                'marginLeft': '0.5rem'
            })
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '0.5rem'})
    
    def _create_trade_analysis(self):
        """Create trade analysis panel"""
        return html.Div([
            html.H3('Trade Analysis', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Win/Loss Distribution
            html.Div([
                html.Div([
                    html.Div('Winning Trades', style={'color': '#00ff88', 'fontSize': '0.875rem'}),
                    html.Div('0', id='winning-trades', style={'color': '#00ff88', 'fontSize': '1.5rem', 'fontWeight': '600'})
                ], style={'flex': '1', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div('Losing Trades', style={'color': '#ff6b6b', 'fontSize': '0.875rem'}),
                    html.Div('0', id='losing-trades', style={'color': '#ff6b6b', 'fontSize': '1.5rem', 'fontWeight': '600'})
                ], style={'flex': '1', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div('Break Even', style={'color': '#ffd93d', 'fontSize': '0.875rem'}),
                    html.Div('0', id='breakeven-trades', style={'color': '#ffd93d', 'fontSize': '1.5rem', 'fontWeight': '600'})
                ], style={'flex': '1', 'textAlign': 'center'})
            ], style={'display': 'flex', 'marginBottom': '1rem'}),
            
            # Trade Duration Stats
            html.Div([
                html.H4('Trade Duration', style={'color': '#00a3b8', 'fontSize': '1rem', 'marginBottom': '0.5rem'}),
                html.Div([
                    self._create_small_metric('Avg Duration', '0h 0m', 'avg-duration'),
                    self._create_small_metric('Max Duration', '0h 0m', 'max-duration'),
                    self._create_small_metric('Min Duration', '0h 0m', 'min-duration')
                ])
            ])
        ], className='backtest-card')
    
    def _create_equity_curve(self):
        """Create equity curve chart"""
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        equity = [10000]
        for _ in range(len(dates) - 1):
            change = np.random.randn() * 100
            equity.append(max(0, equity[-1] + change))
        
        fig = go.Figure()
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00a3b8', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 163, 184, 0.1)'
        ))
        
        # Buy/Hold benchmark
        benchmark = [10000 * (1 + 0.0003) ** i for i in range(len(dates))]
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#888', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Equity Curve',
            height=400,
            margin=dict(l=50, r=30, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickformat='$,.0f',
                title='Portfolio Value'
            ),
            hovermode='x unified',
            legend=dict(x=0, y=1)
        )
        
        return html.Div([
            html.H3('Equity Curve', style={'color': 'white', 'marginBottom': '1rem'}),
            html.Div([
                dcc.Graph(
                    id='equity-curve-chart',
                    figure=fig,
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '400px'}
                )
            ], style={'width': '100%', 'height': '400px', 'position': 'relative'})
        ], className='backtest-card')
    
    def _create_drawdown_chart(self):
        """Create drawdown chart"""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        drawdown = np.cumsum(np.random.randn(len(dates)) * 0.5)
        drawdown = np.minimum(0, drawdown - np.maximum.accumulate(drawdown))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff6b6b', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ))
        
        fig.update_layout(
            title='Drawdown',
            height=300,
            margin=dict(l=50, r=30, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickformat='.1%',
                title='Drawdown %'
            ),
            hovermode='x unified'
        )
        
        return html.Div([
            html.H3('Drawdown Analysis', style={'color': 'white', 'marginBottom': '1rem'}),
            html.Div([
                dcc.Graph(
                    id='drawdown-chart',
                    figure=fig,
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '300px'}
                )
            ], style={'width': '100%', 'height': '300px', 'position': 'relative'})
        ], className='backtest-card')
    
    def _create_returns_distribution(self):
        """Create returns distribution histogram"""
        returns = np.random.randn(1000) * 0.02
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker=dict(
                color='#00a3b8',
                line=dict(color='#008394', width=1)
            )
        ))
        
        # Add normal distribution overlay
        x_range = np.linspace(returns.min(), returns.max(), 100)
        y_normal = ((1 / np.sqrt(2 * np.pi * returns.std()**2)) * 
                   np.exp(-0.5 * ((x_range - returns.mean()) / returns.std())**2))
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_normal * len(returns) * (returns.max() - returns.min()) / 50,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#ffd93d', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Returns Distribution',
            height=300,
            margin=dict(l=50, r=30, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white'),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickformat='.1%',
                title='Daily Returns'
            ),
            yaxis=dict(
                showgrid=False,
                title='Frequency'
            ),
            yaxis2=dict(
                overlaying='y',
                side='right',
                showgrid=False,
                showticklabels=False
            ),
            bargap=0.1,
            hovermode='x unified',
            showlegend=True
        )
        
        return html.Div([
            html.H3('Returns Distribution', style={'color': 'white', 'marginBottom': '1rem'}),
            html.Div([
                dcc.Graph(
                    id='returns-distribution-chart',
                    figure=fig,
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '300px'}
                )
            ], style={'width': '100%', 'height': '300px', 'position': 'relative'})
        ], className='backtest-card')
    
    def _create_walk_forward_section(self):
        """Create walk-forward analysis section"""
        return html.Div([
            html.H3('Walk-Forward Analysis', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Configuration
            html.Div([
                html.Div([
                    html.Label('Training Period (days)', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.Input(
                        id='training-period',
                        type='number',
                        value=180,
                        style={'width': '100%', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white', 'padding': '0.5rem'}
                    )
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                html.Div([
                    html.Label('Test Period (days)', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.Input(
                        id='test-period',
                        type='number',
                        value=30,
                        style={'width': '100%', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white', 'padding': '0.5rem'}
                    )
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                html.Div([
                    html.Label('Number of Folds', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.Input(
                        id='num-folds',
                        type='number',
                        value=10,
                        style={'width': '100%', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white', 'padding': '0.5rem'}
                    )
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                html.Button([
                    html.I(className="fas fa-forward", style={'marginRight': '0.5rem'}),
                    'RUN WALK-FORWARD'
                ], id='run-walk-forward-btn', style={
                    'background': 'linear-gradient(45deg, #008394, #00a3b8)',
                    'border': 'none',
                    'color': 'white',
                    'padding': '0.75rem 1.5rem',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontWeight': '600'
                })
            ], style={'display': 'flex', 'alignItems': 'flex-end', 'marginBottom': '1rem'}),
            
            # Results Table
            dash_table.DataTable(
                id='walk-forward-results',
                columns=[
                    {'name': 'Fold', 'id': 'fold'},
                    {'name': 'Train Start', 'id': 'train_start'},
                    {'name': 'Train End', 'id': 'train_end'},
                    {'name': 'Test Start', 'id': 'test_start'},
                    {'name': 'Test End', 'id': 'test_end'},
                    {'name': 'Train Return', 'id': 'train_return'},
                    {'name': 'Test Return', 'id': 'test_return'},
                    {'name': 'Sharpe', 'id': 'sharpe'}
                ],
                data=[],
                style_cell={
                    'backgroundColor': 'rgba(0, 0, 0, 0.2)',
                    'color': 'white',
                    'border': '1px solid rgba(255, 255, 255, 0.1)',
                    'fontSize': '0.875rem'
                },
                style_header={
                    'backgroundColor': 'rgba(0, 131, 148, 0.2)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'test_return'},
                        'color': '#00ff88',
                        'fontWeight': 'bold'
                    }
                ],
                page_size=5
            )
        ], className='backtest-card')
    
    def _create_strategy_comparison(self):
        """Create strategy comparison section"""
        return html.Div([
            html.H3('Strategy Comparison', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Comparison Table
            dash_table.DataTable(
                id='strategy-comparison-table',
                columns=[
                    {'name': 'Strategy', 'id': 'strategy'},
                    {'name': 'Total Return', 'id': 'total_return'},
                    {'name': 'Sharpe Ratio', 'id': 'sharpe_ratio'},
                    {'name': 'Max DD', 'id': 'max_dd'},
                    {'name': 'Win Rate', 'id': 'win_rate'},
                    {'name': 'Trades', 'id': 'trades'},
                    {'name': 'Profit Factor', 'id': 'profit_factor'}
                ],
                data=[],
                style_cell={
                    'backgroundColor': 'rgba(0, 0, 0, 0.2)',
                    'color': 'white',
                    'border': '1px solid rgba(255, 255, 255, 0.1)'
                },
                style_header={
                    'backgroundColor': 'rgba(0, 131, 148, 0.2)',
                    'fontWeight': 'bold'
                },
                sort_action='native',
                row_selectable='multi',
                selected_rows=[]
            ),
            
            # Export Button
            html.Div([
                html.Button([
                    html.I(className="fas fa-download", style={'marginRight': '0.5rem'}),
                    'Export Results'
                ], id='export-backtest-btn', style={
                    'background': 'rgba(0, 131, 148, 0.2)',
                    'border': '1px solid #008394',
                    'color': '#00a3b8',
                    'padding': '0.5rem 1rem',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'marginTop': '1rem'
                }),
                
                dcc.Download(id='download-backtest-results')
            ])
        ], className='backtest-card')
    
    def _create_trade_log(self):
        """Create trade log table"""
        return html.Div([
            html.H3('Trade Log', style={'color': 'white', 'marginBottom': '1rem'}),
            
            dash_table.DataTable(
                id='trade-log-table',
                columns=[
                    {'name': 'Time', 'id': 'time'},
                    {'name': 'Symbol', 'id': 'symbol'},
                    {'name': 'Side', 'id': 'side'},
                    {'name': 'Entry', 'id': 'entry'},
                    {'name': 'Exit', 'id': 'exit'},
                    {'name': 'Quantity', 'id': 'quantity'},
                    {'name': 'P&L', 'id': 'pnl'},
                    {'name': 'P&L %', 'id': 'pnl_pct'},
                    {'name': 'Duration', 'id': 'duration'},
                    {'name': 'Signal', 'id': 'signal'}
                ],
                data=[],
                style_cell={
                    'backgroundColor': 'rgba(0, 0, 0, 0.2)',
                    'color': 'white',
                    'border': '1px solid rgba(255, 255, 255, 0.1)',
                    'fontSize': '0.875rem'
                },
                style_header={
                    'backgroundColor': 'rgba(0, 131, 148, 0.2)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'pnl', 'filter_query': '{pnl} > 0'},
                        'color': '#00ff88'
                    },
                    {
                        'if': {'column_id': 'pnl', 'filter_query': '{pnl} < 0'},
                        'color': '#ff6b6b'
                    }
                ],
                page_size=10,
                filter_action='native',
                sort_action='native'
            )
        ], className='backtest-card', style={'marginTop': '2rem'})