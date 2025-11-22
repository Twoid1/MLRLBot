"""
dashboard/components/portfolio_center.py
Portfolio Command Center - COMPLETE FIX
Fixed graph expansion and clipping issues
"""

from dash import html, dcc, dash_table, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

class PortfolioCenter:
    """Portfolio Command Center component for trading and position management"""
    
    def __init__(self):
        """Initialize the Portfolio Center component"""
        self.assets = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
        self.trading_pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD']
        
        # Mock portfolio state (will be replaced with backend connection)
        self.portfolio_state = {
            'mode': 'paper',  # 'paper' or 'live'
            'balance': 10000.0,
            'initial_balance': 10000.0,
            'positions': [],
            'pending_orders': [],
            'trade_history': []
        }
        
        # Performance metrics
        self.performance_metrics = {
            'total_return': 0.0,
            'daily_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0
        }
    
    def create_layout(self):
        """Create the main portfolio center layout"""
        return html.Div([
            # Header
            self._create_header(),
            
            # Trading Mode Selector (Paper vs Live)
            self._create_mode_selector(),
            
            # Top Row: Overview Cards
            self._create_overview_cards(),
            
            # Main Content Grid
            html.Div([
                # Left Column: Trading Panel & Positions
                html.Div([
                    self._create_trading_panel(),
                    self._create_positions_table(),
                    self._create_pending_orders_table()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'}),
                
                # Right Column: Performance & Risk
                html.Div([
                    self._create_performance_chart(),
                    self._create_asset_allocation_chart(),
                    self._create_risk_metrics_panel()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'})
            ], style={'display': 'flex', 'gap': '1.5rem', 'marginBottom': '2rem'}),
            
            # Bottom: Trade History
            self._create_trade_history_table(),
            
            # Risk Alerts Panel
            self._create_risk_alerts_panel()
            
        ], style={'padding': '2rem', 'maxWidth': '1600px', 'margin': '0 auto'})
    
    def _create_header(self):
        """Create the header section"""
        return html.Div([
            html.H1([
                html.I(className="fas fa-briefcase", style={'marginRight': '1rem', 'color': '#00a3b8'}),
                'Portfolio Command Center'
            ], style={'color': 'white', 'marginBottom': '0.5rem'}),
            html.P('Manage positions, execute trades, and monitor performance', 
                  style={'color': '#b0b0b0', 'fontSize': '1.125rem'}),
            html.Hr(style={'borderColor': '#4a4a4a', 'marginBottom': '2rem'})
        ])
    
    def _create_mode_selector(self):
        """Create trading mode selector (Paper vs Live)"""
        return html.Div([
            html.Div([
                html.Button([
                    html.I(className="fas fa-graduation-cap", style={'marginRight': '0.5rem'}),
                    'Paper Trading'
                ], id='paper-mode-btn', className='mode-btn active', style={
                    'background': 'linear-gradient(90deg, #008394 0%, #00a3b8 100%)',
                    'border': 'none',
                    'color': 'white',
                    'padding': '0.75rem 1.5rem',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'marginRight': '1rem'
                }),
                
                html.Button([
                    html.I(className="fas fa-rocket", style={'marginRight': '0.5rem'}),
                    'Live Trading'
                ], id='live-mode-btn', className='mode-btn', style={
                    'background': 'rgba(255, 107, 107, 0.2)',
                    'border': '1px solid #ff6b6b',
                    'color': '#ff6b6b',
                    'padding': '0.75rem 1.5rem',
                    'borderRadius': '4px',
                    'cursor': 'pointer'
                }),
                
                html.Span(' Live trading requires API keys and involves real money', 
                         id='live-warning',
                         style={
                             'color': '#ffd93d',
                             'fontSize': '0.85rem',
                             'marginLeft': '1rem',
                             'display': 'none'
                         })
            ], style={'marginBottom': '2rem'})
        ])
    
    def _create_overview_cards(self):
        """Create overview metric cards"""
        return html.Div([
            # Total Balance Card
            self._create_metric_card(
                'Total Balance',
                '$10,000.00',
                '+$0.00 (0.00%)',
                'fas fa-wallet',
                'balance-card',
                trend='neutral'
            ),
            
            # Daily P&L Card
            self._create_metric_card(
                "Today's P&L",
                '+$0.00',
                '0.00%',
                'fas fa-chart-line',
                'daily-pnl-card',
                trend='neutral'
            ),
            
            # Open Positions Card
            self._create_metric_card(
                'Open Positions',
                '0',
                '$0.00 exposure',
                'fas fa-coins',
                'positions-card',
                trend='neutral'
            ),
            
            # Win Rate Card
            self._create_metric_card(
                'Win Rate',
                '0.0%',
                '0 trades today',
                'fas fa-trophy',
                'winrate-card',
                trend='neutral'
            ),
            
            # Risk Level Card
            self._create_metric_card(
                'Risk Level',
                'Low',
                'All systems normal',
                'fas fa-shield-alt',
                'risk-card',
                trend='good'
            )
            
        ], style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(5, 1fr)',
            'gap': '1rem',
            'marginBottom': '2rem'
        })
    
    def _create_metric_card(self, title, value, subtitle, icon, card_id, trend='neutral'):
        """Create a metric card"""
        trend_colors = {
            'good': '#00ff88',
            'bad': '#ff6b6b',
            'neutral': '#b0b0b0'
        }
        
        return html.Div([
            html.Div([
                html.I(className=icon, style={
                    'fontSize': '1.5rem',
                    'color': '#00a3b8',
                    'marginBottom': '0.5rem'
                }),
                html.H4(title, style={
                    'color': '#b0b0b0',
                    'fontSize': '0.875rem',
                    'marginBottom': '0.5rem',
                    'fontWeight': 'normal'
                }),
                html.H2(value, id=f'{card_id}-value', style={
                    'color': 'white',
                    'fontSize': '1.5rem',
                    'marginBottom': '0.25rem'
                }),
                html.P(subtitle, id=f'{card_id}-subtitle', style={
                    'color': trend_colors[trend],
                    'fontSize': '0.75rem',
                    'margin': '0'
                })
            ], style={'padding': '1.25rem'})
        ], className='portfolio-metric-card')
    
    def _create_trading_panel(self):
        """Create the trading panel"""
        return html.Div([
            html.H3('Execute Trade', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Trading controls grid
            html.Div([
                # Row 1: Asset and Order Type
                html.Div([
                    html.Div([
                        html.Label('Trading Pair', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                        dcc.Dropdown(
                            id='trading-pair-select',
                            options=[{'label': pair, 'value': pair} for pair in self.trading_pairs],
                            value='BTC/USD',
                            style={'background': '#1a1a1a'}
                        )
                    ], style={'flex': '1', 'marginRight': '1rem'}),
                    
                    html.Div([
                        html.Label('Order Type', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                        dcc.Dropdown(
                            id='order-type-select',
                            options=[
                                {'label': 'Market', 'value': 'MARKET'},
                                {'label': 'Limit', 'value': 'LIMIT'}
                            ],
                            value='MARKET',
                            style={'background': '#1a1a1a'}
                        )
                    ], style={'flex': '1'})
                ], style={'display': 'flex', 'marginBottom': '1rem'}),
                
                # Row 2: Amount and Price
                html.Div([
                    html.Div([
                        html.Label('Amount', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                        dcc.Input(
                            id='trade-amount',
                            type='number',
                            placeholder='0.0',
                            style={'width': '100%', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white', 'padding': '0.5rem'}
                        )
                    ], style={'flex': '1', 'marginRight': '1rem'}),
                    
                    html.Div([
                        html.Label('Limit Price', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                        dcc.Input(
                            id='limit-price',
                            type='number',
                            placeholder='0.00',
                            disabled=True,
                            style={'width': '100%', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white', 'padding': '0.5rem', 'opacity': '0.5'}
                        )
                    ], style={'flex': '1'})
                ], style={'display': 'flex', 'marginBottom': '1rem'}),
                
                # Row 3: Risk Management
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id='use-stop-loss',
                            options=[{'label': ' Stop Loss', 'value': 'sl'}],
                            value=[],
                            style={'color': '#b0b0b0', 'display': 'inline-block'}
                        ),
                        dcc.Input(
                            id='stop-loss-pct',
                            type='number',
                            placeholder='%',
                            style={'width': '60px', 'marginLeft': '0.5rem', 'display': 'none'}
                        )
                    ], style={'marginRight': '2rem'}),
                    
                    html.Div([
                        dcc.Checklist(
                            id='use-take-profit',
                            options=[{'label': ' Take Profit', 'value': 'tp'}],
                            value=[],
                            style={'color': '#b0b0b0', 'display': 'inline-block'}
                        ),
                        dcc.Input(
                            id='take-profit-pct',
                            type='number',
                            placeholder='%',
                            style={'width': '60px', 'marginLeft': '0.5rem', 'display': 'none'}
                        )
                    ])
                ], style={'display': 'flex', 'marginBottom': '1rem'}),
                
                # Row 4: ML/RL Signals
                html.Div([
                    html.Div([
                        html.Span('ML Signal: ', style={'color': '#b0b0b0'}),
                        html.Span('NEUTRAL', id='ml-prediction', style={'color': '#ffd93d', 'fontWeight': 'bold'})
                    ], style={'marginRight': '2rem'}),
                    html.Div([
                        html.Span('RL Action: ', style={'color': '#b0b0b0'}),
                        html.Span('HOLD', id='rl-action', style={'color': '#ffd93d', 'fontWeight': 'bold'})
                    ])
                ], style={'display': 'flex', 'marginBottom': '1rem'}),
                
                # Execute button
                html.Button('EXECUTE TRADE', id='execute-trade-btn', style={
                    'background': 'linear-gradient(45deg, #B98544, #E13A3E)',
                    'border': 'none',
                    'color': 'white',
                    'padding': '1rem',
                    'width': '100%',
                    'borderRadius': '4px',
                    'fontWeight': '600',
                    'cursor': 'pointer'
                })
            ])
        ], className='trading-card')
    
    def _create_positions_table(self):
        """Create the open positions table"""
        return html.Div([
            html.Div([
                html.H3('Open Positions', style={'color': 'white'}),
                html.Button('Close Selected', id='close-position-btn', style={
                    'background': '#ff6b6b',
                    'border': 'none',
                    'color': 'white',
                    'padding': '0.5rem 1rem',
                    'borderRadius': '4px',
                    'fontSize': '0.875rem',
                    'cursor': 'pointer'
                })
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '1rem'}),
            
            dash_table.DataTable(
                id='positions-table',
                columns=[
                    {'name': 'Symbol', 'id': 'symbol'},
                    {'name': 'Side', 'id': 'side'},
                    {'name': 'Entry', 'id': 'entry'},
                    {'name': 'Current', 'id': 'current'},
                    {'name': 'Size', 'id': 'size'},
                    {'name': 'P&L', 'id': 'pnl'},
                    {'name': 'P&L %', 'id': 'pnl_pct'},
                    {'name': 'SL', 'id': 'sl'},
                    {'name': 'TP', 'id': 'tp'}
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
                style_data_conditional=[
                    {
                        'if': {'column_id': 'pnl'},
                        'color': '#00ff88',
                        'fontWeight': 'bold'
                    }
                ],
                editable=False,
                row_selectable='single'
            )
        ], className='trading-card')
    
    def _create_pending_orders_table(self):
        """Create the pending orders table"""
        return html.Div([
            html.Div([
                html.H3('Pending Orders', style={'color': 'white'}),
                html.Button('Cancel Selected', id='cancel-order-btn', style={
                    'background': '#ffd93d',
                    'border': 'none',
                    'color': '#000',
                    'padding': '0.5rem 1rem',
                    'borderRadius': '4px',
                    'fontSize': '0.875rem',
                    'cursor': 'pointer'
                })
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '1rem'}),
            
            dash_table.DataTable(
                id='pending-orders-table',
                columns=[
                    {'name': 'Time', 'id': 'time'},
                    {'name': 'Symbol', 'id': 'symbol'},
                    {'name': 'Type', 'id': 'type'},
                    {'name': 'Side', 'id': 'side'},
                    {'name': 'Price', 'id': 'price'},
                    {'name': 'Amount', 'id': 'amount'},
                    {'name': 'Status', 'id': 'status'}
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
                editable=False,
                row_selectable='single'
            )
        ], className='trading-card')
    
    def _create_performance_chart(self):
        """Create the portfolio performance chart with graph wrapper"""
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        values = 10000 + np.cumsum(np.random.randn(len(dates)) * 50)
        
        fig = go.Figure()
        
        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00a3b8', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 163, 184, 0.1)'
        ))
        
        # Add benchmark line
        fig.add_trace(go.Scatter(
            x=dates,
            y=[10000] * len(dates),
            mode='lines',
            name='Initial Value',
            line=dict(color='#888', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Portfolio Performance',
            height=400,
            margin=dict(l=50, r=30, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat='$,.0f'),
            hovermode='x unified',
            autosize=True
        )
        
        return html.Div([
            html.H3('Portfolio Performance', style={'color': 'white', 'marginBottom': '1rem'}),
            # Wrapper div with explicit style to prevent overflow
            html.Div([
                dcc.Graph(
                    id='performance-chart',
                    figure=fig,
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '400px'}  # Fixed height for the graph
                )
            ], style={'width': '100%', 'height': '400px', 'position': 'relative'})  # Container with fixed height
        ], className='trading-card')
    
    def _create_asset_allocation_chart(self):
        """Create the asset allocation pie chart with graph wrapper"""
        # Sample allocation data
        labels = ['BTC', 'ETH', 'SOL', 'Cash']
        values = [30, 25, 20, 25]
        colors = ['#f7931a', '#627eea', '#00d4b5', '#888888']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='#000', width=2)),
            textfont=dict(color='white'),
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Asset Allocation',
            height=400,
            margin=dict(l=30, r=100, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            ),
            autosize=True
        )
        
        return html.Div([
            html.H3('Asset Allocation', style={'color': 'white', 'marginBottom': '1rem'}),
            # Wrapper div with explicit style to prevent overflow
            html.Div([
                dcc.Graph(
                    id='allocation-chart',
                    figure=fig,
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '400px'}  # Fixed height for the graph
                )
            ], style={'width': '100%', 'height': '400px', 'position': 'relative'})  # Container with fixed height
        ], className='trading-card')
    
    def _create_risk_metrics_panel(self):
        """Create the risk metrics panel"""
        return html.Div([
            html.H3('Risk Metrics', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                self._create_risk_metric('Sharpe Ratio', '0.00', 'neutral'),
                self._create_risk_metric('Max Drawdown', '0.00%', 'good'),
                self._create_risk_metric('Value at Risk', '$0.00', 'good'),
                self._create_risk_metric('Position Risk', '0.00%', 'good'),
                self._create_risk_metric('Correlation', '0.00', 'neutral'),
                self._create_risk_metric('Beta', '0.00', 'neutral')
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(2, 1fr)',
                'gap': '1rem',
                'paddingBottom': '1rem'
            })
        ], className='trading-card')
    
    def _create_risk_metric(self, label, value, status='neutral'):
        """Create a single risk metric"""
        status_colors = {
            'good': '#00ff88',
            'warning': '#ffd93d',
            'bad': '#ff6b6b',
            'neutral': '#b0b0b0'
        }
        
        return html.Div([
            html.Span(label, style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
            html.H4(value, style={'color': status_colors[status], 'margin': '0.25rem 0'})
        ])
    
    def _create_trade_history_table(self):
        """Create the trade history table"""
        return html.Div([
            html.Div([
                html.H3('Trade History', style={'color': 'white'}),
                html.Button([
                    html.I(className="fas fa-download", style={'marginRight': '0.5rem'}),
                    'Export CSV'
                ], id='export-trades-btn', style={
                    'background': 'rgba(0, 131, 148, 0.2)',
                    'border': '1px solid #008394',
                    'color': '#00a3b8',
                    'padding': '0.5rem 1rem',
                    'borderRadius': '4px',
                    'fontSize': '0.875rem',
                    'cursor': 'pointer'
                })
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '1rem'}),
            
            dash_table.DataTable(
                id='trade-history-table',
                columns=[
                    {'name': 'Time', 'id': 'time'},
                    {'name': 'Symbol', 'id': 'symbol'},
                    {'name': 'Side', 'id': 'side'},
                    {'name': 'Price', 'id': 'price'},
                    {'name': 'Amount', 'id': 'amount'},
                    {'name': 'Total', 'id': 'total'},
                    {'name': 'Fee', 'id': 'fee'},
                    {'name': 'P&L', 'id': 'pnl'}
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
                page_size=10,
                style_table={'overflowX': 'auto'}
            ),
            
            # Download component
            dcc.Download(id='download-trades')
        ], className='trading-card', style={'marginTop': '2rem'})
    
    def _create_risk_alerts_panel(self):
        """Create the risk alerts panel"""
        return html.Div([
            html.H3('Risk Alerts', style={'color': 'white', 'marginBottom': '1rem'}),
            html.Div(id='risk-alerts-container', children=[
                html.Div([
                    html.I(className="fas fa-info-circle", style={'marginRight': '0.5rem', 'color': '#00a3b8'}),
                    'No active alerts'
                ], style={'padding': '0.75rem', 'background': 'rgba(0, 131, 148, 0.1)', 'borderRadius': '4px', 'color': '#00a3b8'})
            ])
        ], className='trading-card', style={'marginTop': '2rem'})