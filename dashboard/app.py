"""
ROCKETS TRADING - Main Dashboard Application
Dash implementation of the trading bot UI with real-time updates
Fixed version with original design and working navigation
"""

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
import json

# Import your Data Center component
from components.data_center import DataCenter
from components.training_lab import TrainingLab
from callbacks.training_callbacks import register_training_callbacks
from components.portfolio_center import PortfolioCenter
from callbacks.portfolio_callbacks import register_portfolio_callbacks
from components.backtest_lab import BacktestLab
from callbacks.backtest_callbacks import register_backtest_callbacks
from components.risk_center import RiskCenter
from callbacks.risk_callbacks import register_risk_callbacks
from dashboard.callbacks.data_callbacks import register_data_callbacks

# Initialize Data Center
data_center = DataCenter()
training_lab = TrainingLab()
portfolio_center = PortfolioCenter()
backtest_lab = BacktestLab()
risk_center = RiskCenter()

# Initialize Dash app with dark theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)


# Mock data for demonstration (replace with real backend connections)
class MockDataProvider:
    """Mock data provider - replace with real backend connections"""
    
    @staticmethod
    def get_portfolio_data():
        return {
            'total_value': 12847.32,
            'daily_pnl': 247.85,
            'daily_pnl_percent': 1.97,
            'positions': 3,
            'active_orders': 1
        }
    
    @staticmethod
    def get_active_positions():
        return [
            {'symbol': 'BTC', 'type': 'LONG', 'size': '0.025 BTC', 'pnl': 127.45},
            {'symbol': 'ETH', 'type': 'LONG', 'size': '1.2 ETH', 'pnl': 89.23},
            {'symbol': 'SOL', 'type': 'SHORT', 'size': '5.0 SOL', 'pnl': -23.11}
        ]
    
    @staticmethod
    def get_market_data():
        return [
            {'symbol': 'BTC', 'price': 43247.32, 'change': 2.34, 'volume': '1.2M'},
            {'symbol': 'ETH', 'price': 2634.87, 'change': -0.87, 'volume': '890K'},
            {'symbol': 'SOL', 'price': 98.42, 'change': 4.21, 'volume': '2.1M'},
            {'symbol': 'ADA', 'price': 0.487, 'change': 1.23, 'volume': '45M'},
            {'symbol': 'DOT', 'price': 7.23, 'change': -1.45, 'volume': '12M'}
        ]
    
    @staticmethod
    def get_model_status():
        return {
            'ml_model': {'status': 'TRAINED', 'accuracy': 0.847, 'last_update': '2 hours ago'},
            'rl_agent': {'status': 'TRAINING', 'episodes': 1247, 'reward': 0.23}
        }
    
    @staticmethod
    def get_system_health():
        return {
            'api_connection': 'Healthy',
            'data_feed': 'Active',
            'risk_manager': 'Monitoring',
            'order_engine': 'Ready'
        }

# Initialize data provider
data_provider = MockDataProvider()

# Layout components
def create_header():
    """Create the dashboard header"""
    return html.Div([
        html.Div([
            # Logo section
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line", style={'color': 'white'})
                ], className='logo-icon'),
                html.Span('ROCKETS TRADING', className='app-title')
            ], style={'display': 'inline-flex', 'alignItems': 'center'}),
            
            # System status
            html.Div([
                html.Div(className='status-dot'),
                html.Span('ACTIVE', id='system-status-text')
            ], className='status-indicator', style={'marginLeft': '2rem'})
        ], style={'display': 'flex', 'alignItems': 'center'}),
        
        # Time display
        html.Div(id='live-time', style={
            'fontFamily': 'JetBrains Mono, monospace',
            'fontSize': '1.125rem',
            'fontWeight': '600',
            'color': '#008394',
            'textShadow': '0 0 10px rgba(0, 131, 148, 0.5)'
        }),
        
        # Connection status
        html.Div([
            html.I(className="fas fa-desktop", style={'marginRight': '0.5rem'}),
            html.Span('CONNECTED', id='connection-status-text')
        ], style={'display': 'flex', 'alignItems': 'center', 'color': '#008394'})
    ], className='dashboard-header', style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'})

def create_navigation():
    """Create the navigation tabs"""
    return html.Div([
        html.Div([
            html.Button([
                html.I(className="fas fa-tachometer-alt", style={'marginRight': '0.5rem'}),
                'Dashboard'
            ], className='nav-tab active', id='nav-dashboard'),
            html.Button([
                html.I(className="fas fa-chart-area", style={'marginRight': '0.5rem'}),
                'Data Center'
            ], className='nav-tab', id='nav-data'),
            html.Button([
                html.I(className="fas fa-brain", style={'marginRight': '0.5rem'}),
                'AI Training'
            ], className='nav-tab', id='nav-training'),
            html.Button([
                html.I(className="fas fa-wallet", style={'marginRight': '0.5rem'}),
                'Portfolio'
            ], className='nav-tab', id='nav-portfolio'),
            html.Button([
                html.I(className="fas fa-history", style={'marginRight': '0.5rem'}),
                'Backtesting'
            ], className='nav-tab', id='nav-backtest'),
            html.Button([
                html.I(className="fas fa-shield-alt", style={'marginRight': '0.5rem'}),
                'Risk Management'
            ], className='nav-tab', id='nav-risk')
        ], style={'display': 'flex', 'gap': '0.5rem'})
    ], className='page-navigation')

def create_portfolio_card():
    """Create portfolio overview card"""
    portfolio_data = data_provider.get_portfolio_data()
    
    return html.Div([
        html.Div([
            html.H3('Portfolio Overview'),
            html.I(className="fas fa-dollar-sign", style={'color': '#b0b0b0'})
        ], className='card-header'),
        
        html.Div([
            # Total Value
            html.Div([
                html.Span('Total Value', style={'fontSize': '0.875rem', 'color': '#b0b0b0'}),
                html.Div(f"${portfolio_data['total_value']:,.2f}", className='metric-large')
            ], style={'marginBottom': '1rem'}),
            
            # Metrics row
            html.Div([
                html.Div([
                    html.Span('Daily P&L', style={'fontSize': '0.75rem', 'color': '#888', 'display': 'block'}),
                    html.Span(
                        f"${portfolio_data['daily_pnl']:.2f} ({portfolio_data['daily_pnl_percent']:+.2f}%)",
                        className='value-positive' if portfolio_data['daily_pnl'] >= 0 else 'value-negative',
                        style={'fontWeight': '700'}
                    )
                ], style={'flex': '1'}),
                html.Div([
                    html.Span('Positions', style={'fontSize': '0.75rem', 'color': '#888', 'display': 'block'}),
                    html.Span(str(portfolio_data['positions']), style={'fontWeight': '700', 'color': 'white'})
                ], style={'flex': '1'}),
                html.Div([
                    html.Span('Active Orders', style={'fontSize': '0.75rem', 'color': '#888', 'display': 'block'}),
                    html.Span(str(portfolio_data['active_orders']), style={'fontWeight': '700', 'color': 'white'})
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'gap': '1rem'})
        ])
    ], className='trading-card', id='portfolio-card')

def create_positions_card():
    """Create active positions card"""
    positions = data_provider.get_active_positions()
    
    position_rows = []
    for pos in positions:
        position_rows.append(
            html.Div([
                html.Div([
                    html.Span(pos['symbol'], style={'fontWeight': '700', 'color': 'white'}),
                    html.Span(
                        pos['type'],
                        className=f"position-{pos['type'].lower()}",
                        style={'marginLeft': '0.75rem'}
                    )
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    html.Span(pos['size'], style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    html.Span(
                        f"${pos['pnl']:+.2f}",
                        className='value-positive' if pos['pnl'] >= 0 else 'value-negative',
                        style={'marginLeft': '1rem', 'fontWeight': '600'}
                    )
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'padding': '0.75rem',
                'background': 'rgba(255, 255, 255, 0.05)',
                'borderRadius': '8px',
                'border': '1px solid #4a4a4a',
                'marginBottom': '0.75rem'
            })
        )
    
    return html.Div([
        html.Div([
            html.H3('Active Positions'),
            html.I(className="fas fa-crosshairs", style={'color': '#b0b0b0'})
        ], className='card-header'),
        html.Div(position_rows, id='positions-list')
    ], className='trading-card', id='positions-card')

def create_actions_card():
    """Create quick actions card"""
    return html.Div([
        html.Div([
            html.H3('Quick Actions'),
            html.I(className="fas fa-bolt", style={'color': '#b0b0b0'})
        ], className='card-header'),
        
        html.Div([
            html.Button([
                html.I(className="fas fa-play", style={'marginRight': '0.5rem'}),
                'Start Trading'
            ], className='action-btn-primary', id='btn-start-trading', style={'width': '48%'}),
            
            html.Button([
                html.I(className="fas fa-pause", style={'marginRight': '0.5rem'}),
                'Pause Trading'
            ], className='action-btn-secondary', id='btn-pause-trading', style={'width': '48%'}),
            
            html.Button([
                html.I(className="fas fa-shield-alt", style={'marginRight': '0.5rem'}),
                'Emergency Stop'
            ], className='action-btn-danger', id='btn-emergency-stop', style={'width': '48%'}),
            
            html.Button([
                html.I(className="fas fa-brain", style={'marginRight': '0.5rem'}),
                'Train Models'
            ], className='action-btn-accent', id='btn-train-models', style={'width': '48%'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '0.75rem'})
    ], className='trading-card', id='actions-card')

def create_market_data_card():
    """Create live market data card"""
    market_data = data_provider.get_market_data()
    
    asset_rows = []
    for asset in market_data:
        asset_rows.append(
            html.Div([
                html.Div([
                    html.Span(asset['symbol'], style={'fontWeight': '700', 'color': 'white', 'display': 'block'}),
                    html.Span(f"${asset['price']:,.2f}", style={'color': '#b0b0b0', 'fontSize': '0.875rem'})
                ]),
                html.Div([
                    html.Span(
                        f"{asset['change']:+.2f}%",
                        className='value-positive' if asset['change'] >= 0 else 'value-negative',
                        style={'display': 'block', 'fontWeight': '600'}
                    ),
                    html.Span(asset['volume'], style={'fontSize': '0.75rem', 'color': '#888'})
                ], style={'textAlign': 'right'})
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'padding': '0.75rem',
                'background': 'rgba(255, 255, 255, 0.05)',
                'borderRadius': '8px',
                'border': '1px solid #4a4a4a',
                'marginBottom': '0.75rem'
            })
        )
    
    return html.Div([
        html.Div([
            html.H3('Live Market Data'),
            html.I(className="fas fa-chart-line", style={'color': '#b0b0b0'})
        ], className='card-header'),
        html.Div(asset_rows, id='market-data-list')
    ], className='trading-card', id='market-card')

def create_system_health_card():
    """Create system health card"""
    health = data_provider.get_system_health()
    
    health_items = []
    for key, value in health.items():
        status_color = '#4ade80' if value in ['Healthy', 'Active', 'Ready', 'Monitoring'] else '#fbbf24'
        health_items.append(
            html.Div([
                html.Span(key.replace('_', ' ').title(), style={'fontSize': '0.875rem', 'color': '#b0b0b0'}),
                html.Div([
                    html.Div(style={
                        'width': '6px',
                        'height': '6px',
                        'borderRadius': '50%',
                        'background': status_color,
                        'marginRight': '0.5rem'
                    }),
                    html.Span(value, style={'color': status_color, 'fontWeight': '600'})
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '0.5rem 0'})
        )
    
    return html.Div([
        html.Div([
            html.H3('System Health'),
            html.I(className="fas fa-heartbeat", style={'color': '#b0b0b0'})
        ], className='card-header'),
        html.Div(health_items, id='health-metrics')
    ], className='trading-card', id='health-card')

def create_model_status_card():
    """Create AI models status card"""
    models = data_provider.get_model_status()
    
    return html.Div([
        html.Div([
            html.H3('AI Models Status'),
            html.I(className="fas fa-brain", style={'color': '#b0b0b0'})
        ], className='card-header'),
        
        html.Div([
            # ML Model
            html.Div([
                html.Div([
                    html.Span('ML Predictor', style={'fontWeight': '600', 'color': 'white'}),
                    html.Span(
                        models['ml_model']['status'],
                        style={
                            'background': 'rgba(74, 222, 128, 0.2)',
                            'color': '#4ade80',
                            'border': '1px solid #4ade80',
                            'padding': '0.25rem 0.75rem',
                            'borderRadius': '12px',
                            'fontSize': '0.75rem',
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'marginLeft': 'auto'
                        }
                    )
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '0.5rem'}),
                html.Div([
                    html.Span(f"Accuracy: {models['ml_model']['accuracy']*100:.1f}%", style={'fontSize': '0.875rem', 'color': '#b0b0b0'}),
                    html.Span(f"Updated: {models['ml_model']['last_update']}", style={'fontSize': '0.875rem', 'color': '#b0b0b0', 'marginLeft': 'auto'})
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ], style={'background': 'rgba(255, 255, 255, 0.05)', 'border': '1px solid #4a4a4a', 'borderRadius': '8px', 'padding': '1rem', 'marginBottom': '1rem'}),
            
            # RL Agent
            html.Div([
                html.Div([
                    html.Span('RL Agent', style={'fontWeight': '600', 'color': 'white'}),
                    html.Span(
                        models['rl_agent']['status'],
                        style={
                            'background': 'rgba(251, 191, 36, 0.2)',
                            'color': '#fbbf24',
                            'border': '1px solid #fbbf24',
                            'padding': '0.25rem 0.75rem',
                            'borderRadius': '12px',
                            'fontSize': '0.75rem',
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'marginLeft': 'auto'
                        }
                    )
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '0.5rem'}),
                html.Div([
                    html.Span(f"Episodes: {models['rl_agent']['episodes']}", style={'fontSize': '0.875rem', 'color': '#b0b0b0'}),
                    html.Span(f"Reward: {models['rl_agent']['reward']:.3f}", style={'fontSize': '0.875rem', 'color': '#b0b0b0', 'marginLeft': 'auto'})
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ], style={'background': 'rgba(255, 255, 255, 0.05)', 'border': '1px solid #4a4a4a', 'borderRadius': '8px', 'padding': '1rem'})
        ])
    ], className='trading-card', id='model-card')

def create_decorative_separator():
    """Create decorative separator between sections"""
    return html.Div([
        html.Div([
            html.I(className="fas fa-chart-line", style={'fontSize': '32px', 'color': '#008394', 'marginRight': '1.5rem'}),
            html.Span('ROCKETS TRADING SYSTEM', className='separator-text')
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
    ], className='decorative-separator')

def create_footer():
    """Create dashboard footer"""
    return html.Div([
        html.Div([
            # Footer left
            html.Div([
                html.I(className="fas fa-chart-line", style={'color': '#008394', 'marginRight': '0.5rem'}),
                html.Span('ROCKETS TRADING', style={'color': '#008394', 'fontWeight': '700', 'fontSize': '0.875rem'}),
                html.Span('v1.0.0', style={
                    'background': 'rgba(185, 133, 68, 0.2)',
                    'color': '#B98544',
                    'padding': '0.25rem 0.5rem',
                    'borderRadius': '4px',
                    'fontSize': '0.75rem',
                    'fontWeight': '600',
                    'marginLeft': '1rem'
                })
            ], style={'display': 'flex', 'alignItems': 'center'}),
            
            # Footer center
            html.Span('System Operational', style={'color': 'white', 'fontSize': '0.875rem', 'fontWeight': '500'}),
            
            # Footer right
            html.Div([
                html.Span('Session: ', id='session-time', style={'color': '#008394', 'fontFamily': 'JetBrains Mono, monospace', 'fontSize': '0.875rem'}),
                html.Div([
                    html.Div(style={'width': '8px', 'height': '8px', 'borderRadius': '50%', 'background': '#008394', 'boxShadow': '0 0 8px rgba(0, 131, 148, 0.6)'}),
                    html.Div(style={'width': '8px', 'height': '8px', 'borderRadius': '50%', 'background': '#008394', 'boxShadow': '0 0 8px rgba(0, 131, 148, 0.6)'}),
                    html.Div(style={'width': '8px', 'height': '8px', 'borderRadius': '50%', 'background': '#008394', 'boxShadow': '0 0 8px rgba(0, 131, 148, 0.6)'})
                ], style={'display': 'flex', 'gap': '0.5rem', 'marginLeft': '1rem'})
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'maxWidth': '1600px', 'margin': '0 auto'})
    ], className='dashboard-footer')

def create_dashboard_content():
    """Create the main dashboard content"""
    return html.Div([
        # Top section - 3 column grid
        html.Div([
            html.Div(create_portfolio_card(), style={'flex': '1'}),
            html.Div(create_positions_card(), style={'flex': '1'}),
            html.Div(create_actions_card(), style={'flex': '1'})
        ], style={'display': 'flex', 'gap': '1.5rem', 'marginBottom': '2rem'}),
        
        # Decorative separator
        create_decorative_separator(),
        
        # Bottom section - 3 column grid
        html.Div([
            html.Div(create_market_data_card(), style={'flex': '1'}),
            html.Div(create_system_health_card(), style={'flex': '1'}),
            html.Div(create_model_status_card(), style={'flex': '1'})
        ], style={'display': 'flex', 'gap': '1.5rem'})
    ], style={'maxWidth': '1600px', 'margin': '0 auto', 'padding': '2rem'})

# Placeholder functions for other pages
def create_training_content():
    """Create the AI Training Laboratory content"""
    return training_lab.create_layout()

def create_portfolio_content():
    """Create the Portfolio Command Center content"""
    return portfolio_center.create_layout()

def create_backtest_content():
    """Create the Backtesting Laboratory content"""
    return backtest_lab.create_layout()

def create_risk_content():
    """Create the Risk Management Center content"""
    return risk_center.create_layout()

# Main layout
app.layout = html.Div([
    # Store component for tracking current page
    dcc.Store(id='current-page', data='dashboard'),
    
    # Interval components for real-time updates
    dcc.Interval(id='interval-1s', interval=1000, n_intervals=0),  # 1 second
    dcc.Interval(id='interval-5s', interval=5000, n_intervals=0),  # 5 seconds
    dcc.Interval(id='interval-10s', interval=10000, n_intervals=0),  # 10 seconds
    
    # Header
    create_header(),
    
    # Navigation
    create_navigation(),
    
    # Page content container - this will change based on navigation
    html.Div(id='page-content', children=create_dashboard_content()),
    
    # Footer
    create_footer()
], style={'background': 'linear-gradient(135deg, #2a2a2a 0%, #1f1f1f 50%, #2a2a2a 100%)', 'minHeight': '100vh'})

# Navigation callback
@app.callback(
    [Output('page-content', 'children'),
     Output('current-page', 'data')] + 
    [Output(f'nav-{page}', 'className') for page in ['dashboard', 'data', 'training', 'portfolio', 'backtest', 'risk']],
    [Input(f'nav-{page}', 'n_clicks') for page in ['dashboard', 'data', 'training', 'portfolio', 'backtest', 'risk']],
    [State('current-page', 'data')],
    prevent_initial_call=True
)
def navigate_pages(*args):
    """Handle navigation between pages"""
    ctx = callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    # Identify which button was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    page_map = {
        'nav-dashboard': 'dashboard',
        'nav-data': 'data',
        'nav-training': 'training',
        'nav-portfolio': 'portfolio',
        'nav-backtest': 'backtest',
        'nav-risk': 'risk'
    }
    
    selected_page = page_map.get(button_id, 'dashboard')
    
    # Load the appropriate content
    if selected_page == 'dashboard':
        content = create_dashboard_content()
    elif selected_page == 'data':
        content = data_center.create_layout()
    elif selected_page == 'training':
        content = create_training_content()
    elif selected_page == 'portfolio':
        content = create_portfolio_content()
    elif selected_page == 'backtest':
        content = create_backtest_content()
    elif selected_page == 'risk':
        content = create_risk_content()
    else:
        content = create_dashboard_content()
    
    # Update button classes
    button_classes = []
    for page in ['dashboard', 'data', 'training', 'portfolio', 'backtest', 'risk']:
        if page == selected_page:
            button_classes.append('nav-tab active')
        else:
            button_classes.append('nav-tab')
    
    return [content, selected_page] + button_classes

# Callbacks for real-time updates
@app.callback(
    Output('live-time', 'children'),
    Input('interval-1s', 'n_intervals')
)
def update_time(n):
    """Update the live time display"""
    return datetime.now().strftime('%H:%M:%S')

@app.callback(
    Output('session-time', 'children'),
    Input('interval-1s', 'n_intervals'),
    State('session-time', 'children')
)
def update_session_time(n, current_session):
    """Update session duration"""
    # This would calculate actual session time in production
    hours = n // 3600
    minutes = (n % 3600) // 60
    return f"Session: {hours}h {minutes}m"

@app.callback(
    Output('portfolio-card', 'children'),
    Input('interval-10s', 'n_intervals')
)
def update_portfolio(n):
    """Update portfolio data every 10 seconds"""
    # In production, this would fetch real data from portfolio.py
    return create_portfolio_card().children

@app.callback(
    Output('positions-list', 'children'),
    Input('interval-10s', 'n_intervals')
)
def update_positions(n):
    """Update positions every 10 seconds"""
    # In production, this would fetch real positions
    return create_positions_card().children[1].children

@app.callback(
    Output('market-data-list', 'children'),
    Input('interval-5s', 'n_intervals')
)
def update_market_data(n):
    """Update market data every 5 seconds"""
    # In production, this would fetch from kraken_connector.py
    return create_market_data_card().children[1].children

# Action button callbacks
@app.callback(
    Output('system-status-text', 'children'),
    [Input('btn-start-trading', 'n_clicks'),
     Input('btn-pause-trading', 'n_clicks'),
     Input('btn-emergency-stop', 'n_clicks')],
    prevent_initial_call=True
)
def handle_trading_actions(start_clicks, pause_clicks, stop_clicks):
    """Handle trading action buttons"""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'btn-start-trading':
        # In production: start the trading bot
        return 'TRADING'
    elif button_id == 'btn-pause-trading':
        # In production: pause the trading bot
        return 'PAUSED'
    elif button_id == 'btn-emergency-stop':
        # In production: emergency stop all trading
        return 'STOPPED'
    
    return 'ACTIVE'

@app.callback(
    Output('btn-train-models', 'style'),
    Input('btn-train-models', 'n_clicks'),
    State('btn-train-models', 'style'),
    prevent_initial_call=True
)
def handle_train_models(n_clicks, current_style):
    """Handle model training button"""
    if n_clicks:
        # In production: trigger model training
        # This would call ml_predictor.train() and dqn_agent.train()
        print("Training models initiated...")
    return current_style

register_training_callbacks(app, training_lab)
register_portfolio_callbacks(app, portfolio_center)
register_backtest_callbacks(app, backtest_lab)
register_risk_callbacks(app, risk_center)
register_data_callbacks(app, data_center)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("ROCKETS TRADING DASHBOARD")
    print("="*50)
    print(f"Data Center loaded: Ready")
    print("Starting server at http://localhost:8050")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    app.run(debug=True, port=8050, host='127.0.0.1')