"""
dashboard/components/risk_center.py
Risk Management Center for the Rockets Trading Bot
Comprehensive risk monitoring, controls, and alert systems
Ready for backend integration with risk_manager.py, position_sizer.py, and portfolio.py
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

class RiskCenter:
    """Risk Management Center component for monitoring and controlling trading risk"""
    
    def __init__(self):
        """Initialize the Risk Management Center"""
        self.assets = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
        
        # Risk parameters
        self.risk_params = {
            'max_portfolio_risk': 0.02,  # 2% max portfolio risk
            'max_position_size': 0.10,    # 10% max per position
            'max_correlation': 0.70,       # 70% max correlation
            'max_drawdown': 0.15,          # 15% max drawdown
            'daily_loss_limit': 0.05,      # 5% daily loss limit
            'var_confidence': 0.95         # 95% VaR confidence
        }
        
        # Mock risk state
        self.risk_state = {
            'current_risk': 0.015,
            'positions_at_risk': 3,
            'var_1day': 0.02,
            'var_5day': 0.045,
            'current_drawdown': 0.08,
            'correlation_risk': 'Low'
        }
    
    def create_layout(self):
        """Create the main risk management center layout"""
        return html.Div([
            # Header
            self._create_header(),
            
            # Risk Dashboard - Top Level Metrics
            self._create_risk_dashboard(),
            
            # Main Content Grid
            html.Div([
                # Left Column: Risk Controls & Limits
                html.Div([
                    self._create_risk_controls(),
                    self._create_position_limits(),
                    self._create_circuit_breakers()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'}),
                
                # Right Column: Risk Analytics
                html.Div([
                    self._create_var_analysis(),
                    self._create_correlation_matrix(),
                    self._create_exposure_chart()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'})
            ], style={'display': 'flex', 'gap': '1.5rem', 'marginBottom': '2rem'}),
            
            # Risk History & Analysis
            html.Div([
                # Left: Historical Risk Metrics
                html.Div([
                    self._create_risk_history_chart(),
                    self._create_drawdown_analysis()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'}),
                
                # Right: Stress Testing
                html.Div([
                    self._create_stress_testing(),
                    self._create_scenario_analysis()
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '1.5rem'})
            ], style={'display': 'flex', 'gap': '1.5rem', 'marginBottom': '2rem'}),
            
            # Alert System
            self._create_alert_system(),
            
            # Risk Report
            self._create_risk_report()
            
        ], style={'padding': '2rem', 'maxWidth': '1600px', 'margin': '0 auto'})
    
    def _create_header(self):
        """Create the header section"""
        return html.Div([
            html.H1([
                html.I(className="fas fa-shield-alt", style={'marginRight': '1rem', 'color': '#00a3b8'}),
                'Risk Management Center'
            ], style={'color': 'white', 'marginBottom': '0.5rem'}),
            html.P('Monitor exposure, control risk, and protect capital', 
                  style={'color': '#b0b0b0', 'fontSize': '1.125rem'}),
            html.Hr(style={'borderColor': '#4a4a4a', 'marginBottom': '2rem'})
        ])
    
    def _create_risk_dashboard(self):
        """Create the main risk dashboard with key metrics"""
        return html.Div([
            # Risk Status Bar
            html.Div([
                html.Div([
                    html.Span('SYSTEM RISK LEVEL:', style={'color': '#b0b0b0', 'marginRight': '1rem'}),
                    html.Span('MODERATE', id='system-risk-level', style={
                        'color': '#ffd93d',
                        'fontWeight': '700',
                        'fontSize': '1.25rem'
                    })
                ]),
                html.Div([
                    html.Span('Last Update: ', style={'color': '#b0b0b0'}),
                    html.Span(datetime.now().strftime('%H:%M:%S'), id='risk-last-update', style={
                        'color': '#00a3b8',
                        'fontFamily': 'JetBrains Mono, monospace'
                    })
                ])
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '1.5rem'}),
            
            # Key Risk Metrics Grid
            html.Div([
                self._create_risk_metric_card(
                    'Portfolio Risk',
                    '1.5%',
                    '2.0% limit',
                    'fas fa-chart-pie',
                    'risk-portfolio-risk-card',  # Added 'risk-' prefix
                    'warning'
                ),
                self._create_risk_metric_card(
                    'Value at Risk (1D)',
                    '$2,047',
                    '95% confidence',
                    'fas fa-exclamation-triangle',
                    'risk-var-1d-card',  # Added 'risk-' prefix
                    'normal'
                ),
                self._create_risk_metric_card(
                    'Current Drawdown',
                    '-8.2%',
                    'From peak',
                    'fas fa-chart-line',
                    'risk-drawdown-card',  # Added 'risk-' prefix
                    'warning'
                ),
                self._create_risk_metric_card(
                    'Positions at Risk',
                    '3',
                    'Near stop-loss',
                    'fas fa-coins',
                    'risk-positions-at-risk-card',  # Added 'risk-' prefix
                    'warning'
                ),
                self._create_risk_metric_card(
                    'Correlation Risk',
                    'Low',
                    'Portfolio diversity',
                    'fas fa-project-diagram',
                    'risk-correlation-card',  # Added 'risk-' prefix
                    'good'
                ),
                self._create_risk_metric_card(
                    "Today's P&L",
                    '-$523',
                    '-5.2% of limit',
                    'fas fa-dollar-sign',
                    'risk-daily-pnl-card',  # Added 'risk-' prefix
                    'normal'
                )
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(6, 1fr)',
                'gap': '1rem',
                'marginBottom': '2rem'
            })
        ], className='risk-card')
    
    def _create_risk_metric_card(self, title, value, subtitle, icon, card_id, status='normal'):
        """Create a risk metric card"""
        status_colors = {
            'good': '#00ff88',
            'normal': '#00a3b8',
            'warning': '#ffd93d',
            'danger': '#ff6b6b'
        }
        
        return html.Div([
            html.I(className=icon, style={
                'fontSize': '1.5rem',
                'color': status_colors[status],
                'marginBottom': '0.5rem'
            }),
            html.H4(title, style={
                'color': '#b0b0b0',
                'fontSize': '0.75rem',
                'marginBottom': '0.5rem',
                'fontWeight': 'normal'
            }),
            html.H2(value, id=f'{card_id}-value', style={
                'color': 'white',
                'fontSize': '1.5rem',
                'marginBottom': '0.25rem',
                'fontWeight': '600'
            }),
            html.P(subtitle, id=f'{card_id}-subtitle', style={
                'color': status_colors[status],
                'fontSize': '0.75rem',
                'margin': '0'
            })
        ], style={
            'textAlign': 'center',
            'padding': '1rem',
            'background': 'rgba(0, 0, 0, 0.3)',
            'borderRadius': '8px',
            'border': f'1px solid {status_colors[status]}40'
        })
    
    def _create_risk_controls(self):
        """Create risk control panel"""
        return html.Div([
            html.H3('Risk Controls', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Risk Parameters
            html.Div([
                # Max Portfolio Risk
                html.Div([
                    html.Label('Max Portfolio Risk (%)', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    html.Div([
                        dcc.Slider(
                            id='max-portfolio-risk-slider',
                            min=0.5, max=5, step=0.1, value=2,
                            marks={0.5: '0.5%', 1: '1%', 2: '2%', 3: '3%', 5: '5%'},
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        )
                    ])
                ], style={'marginBottom': '1.5rem'}),
                
                # Max Position Size
                html.Div([
                    html.Label('Max Position Size (%)', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    html.Div([
                        dcc.Slider(
                            id='max-position-size-slider',
                            min=1, max=25, step=1, value=10,
                            marks={1: '1%', 5: '5%', 10: '10%', 15: '15%', 25: '25%'},
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        )
                    ])
                ], style={'marginBottom': '1.5rem'}),
                
                # Max Drawdown Limit
                html.Div([
                    html.Label('Max Drawdown Limit (%)', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    html.Div([
                        dcc.Slider(
                            id='max-drawdown-slider',
                            min=5, max=30, step=1, value=15,
                            marks={5: '5%', 10: '10%', 15: '15%', 20: '20%', 30: '30%'},
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        )
                    ])
                ], style={'marginBottom': '1.5rem'}),
                
                # Daily Loss Limit
                html.Div([
                    html.Label('Daily Loss Limit (%)', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    html.Div([
                        dcc.Input(
                            id='daily-loss-limit',
                            type='number',
                            value=5,
                            min=1, max=10, step=0.5,
                            style={'width': '100px', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white', 'padding': '0.5rem'}
                        )
                    ])
                ])
            ]),
            
            # Apply Button
            html.Button([
                html.I(className="fas fa-save", style={'marginRight': '0.5rem'}),
                'APPLY CHANGES'
            ], id='apply-risk-settings-btn', style={
                'background': 'linear-gradient(45deg, #008394, #00a3b8)',
                'border': 'none',
                'color': 'white',
                'padding': '0.75rem 1.5rem',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'fontWeight': '600',
                'width': '100%',
                'marginTop': '1rem'
            })
        ], className='risk-card')
    
    def _create_position_limits(self):
        """Create position limits panel"""
        return html.Div([
            html.H3('Position Limits', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Position Limits Table
            dash_table.DataTable(
                id='position-limits-table',
                columns=[
                    {'name': 'Asset', 'id': 'asset'},
                    {'name': 'Current', 'id': 'current'},
                    {'name': 'Max Size', 'id': 'max_size'},
                    {'name': 'Utilization', 'id': 'utilization'},
                    {'name': 'Status', 'id': 'status'}
                ],
                data=[
                    {'asset': 'BTC', 'current': '0.025', 'max_size': '0.05', 'utilization': '50%', 'status': 'OK'},
                    {'asset': 'ETH', 'current': '1.2', 'max_size': '2.0', 'utilization': '60%', 'status': 'OK'},
                    {'asset': 'SOL', 'current': '45', 'max_size': '50', 'utilization': '90%', 'status': 'WARNING'},
                    {'asset': 'ADA', 'current': '0', 'max_size': '1000', 'utilization': '0%', 'status': 'OK'},
                    {'asset': 'DOT', 'current': '15', 'max_size': '100', 'utilization': '15%', 'status': 'OK'}
                ],
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
                        'if': {'column_id': 'status', 'filter_query': '{status} = "WARNING"'},
                        'backgroundColor': 'rgba(255, 217, 61, 0.2)',
                        'color': '#ffd93d'
                    }
                ],
                editable=True
            )
        ], className='risk-card')
    
    def _create_circuit_breakers(self):
        """Create circuit breakers panel"""
        return html.Div([
            html.H3('Circuit Breakers', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Circuit Breaker Toggles
            html.Div([
                # Drawdown Circuit Breaker
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id='drawdown-breaker',
                            options=[{'label': ' Drawdown Breaker', 'value': 'enabled'}],
                            value=['enabled'],
                            style={'color': '#b0b0b0'}
                        ),
                        html.Span('Stops trading at 15% drawdown', style={
                            'color': '#888',
                            'fontSize': '0.75rem',
                            'marginLeft': '2rem'
                        })
                    ], style={'marginBottom': '1rem'}),
                    
                    html.Div('Status: ARMED', id='drawdown-breaker-status', style={
                        'color': '#00ff88',
                        'fontSize': '0.875rem',
                        'padding': '0.5rem',
                        'background': 'rgba(0, 255, 136, 0.1)',
                        'borderRadius': '4px',
                        'border': '1px solid rgba(0, 255, 136, 0.3)'
                    })
                ], style={'marginBottom': '1.5rem'}),
                
                # Daily Loss Circuit Breaker
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id='daily-loss-breaker',
                            options=[{'label': ' Daily Loss Breaker', 'value': 'enabled'}],
                            value=['enabled'],
                            style={'color': '#b0b0b0'}
                        ),
                        html.Span('Stops trading at 5% daily loss', style={
                            'color': '#888',
                            'fontSize': '0.75rem',
                            'marginLeft': '2rem'
                        })
                    ], style={'marginBottom': '1rem'}),
                    
                    html.Div('Status: ARMED', id='daily-loss-breaker-status', style={
                        'color': '#00ff88',
                        'fontSize': '0.875rem',
                        'padding': '0.5rem',
                        'background': 'rgba(0, 255, 136, 0.1)',
                        'borderRadius': '4px',
                        'border': '1px solid rgba(0, 255, 136, 0.3)'
                    })
                ], style={'marginBottom': '1.5rem'}),
                
                # Correlation Circuit Breaker
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id='correlation-breaker',
                            options=[{'label': ' Correlation Breaker', 'value': 'enabled'}],
                            value=['enabled'],
                            style={'color': '#b0b0b0'}
                        ),
                        html.Span('Stops trading at 70% correlation', style={
                            'color': '#888',
                            'fontSize': '0.75rem',
                            'marginLeft': '2rem'
                        })
                    ], style={'marginBottom': '1rem'}),
                    
                    html.Div('Status: ARMED', id='correlation-breaker-status', style={
                        'color': '#00ff88',
                        'fontSize': '0.875rem',
                        'padding': '0.5rem',
                        'background': 'rgba(0, 255, 136, 0.1)',
                        'borderRadius': '4px',
                        'border': '1px solid rgba(0, 255, 136, 0.3)'
                    })
                ])
            ]),
            
            # Emergency Stop Button
            html.Button([
                html.I(className="fas fa-stop-circle", style={'marginRight': '0.5rem'}),
                'EMERGENCY STOP ALL TRADING'
            ], id='emergency-stop-btn', style={
                'background': 'linear-gradient(45deg, #ff6b6b, #ff4757)',
                'border': 'none',
                'color': 'white',
                'padding': '1rem',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'fontWeight': '700',
                'width': '100%',
                'marginTop': '1.5rem',
                'fontSize': '1rem'
            })
        ], className='risk-card')
    
    def _create_var_analysis(self):
        """Create Value at Risk analysis with proper spacing"""
        # Generate sample VaR data
        confidence_levels = [0.90, 0.95, 0.99]
        periods = ['1 Day', '5 Days', '10 Days']
        
        var_data = []
        for period in periods:
            row = {'period': period}
            for conf in confidence_levels:
                var_value = np.random.uniform(1000, 5000) * (1 + conf - 0.9)
                row[f'{int(conf*100)}%'] = f'${var_value:,.0f}'
            var_data.append(row)
        
        return html.Div([
            # Main container with proper spacing
            html.Div([
                # VaR Header
                html.H3('Value at Risk (VaR) Analysis', 
                    style={
                        'color': 'white', 
                        'marginBottom': '1.5rem',
                        'fontSize': '1.25rem',
                        'fontWeight': '600'
                    }),
                
                # VaR Table Section
                html.Div([
                    html.H4('VaR Estimates by Confidence Level', 
                        style={
                            'color': '#b0b0b0', 
                            'marginBottom': '1rem',
                            'fontSize': '1rem'
                        }),
                    dash_table.DataTable(
                        id='var-table',
                        columns=[
                            {'name': 'Period', 'id': 'period'},
                            {'name': '90% Conf', 'id': '90%'},
                            {'name': '95% Conf', 'id': '95%'},
                            {'name': '99% Conf', 'id': '99%'}
                        ],
                        data=var_data,
                        style_cell={
                            'backgroundColor': 'rgba(0, 0, 0, 0.2)',
                            'color': 'white',
                            'border': '1px solid rgba(255, 255, 255, 0.1)',
                            'textAlign': 'center',
                            'padding': '10px'
                        },
                        style_header={
                            'backgroundColor': 'rgba(0, 131, 148, 0.2)',
                            'fontWeight': 'bold',
                            'borderBottom': '2px solid rgba(0, 163, 184, 0.5)'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgba(0, 0, 0, 0.1)'
                            }
                        ]
                    )
                ], style={
                    'marginBottom': '2rem',  # Add space between table and chart
                    'padding': '1rem',
                    'backgroundColor': 'rgba(0, 0, 0, 0.2)',
                    'borderRadius': '8px'
                }),
                
                # Returns Distribution Chart Section - Separate container
                html.Div([
                    html.H4('Returns Distribution & VaR Visualization', 
                        style={
                            'color': '#b0b0b0', 
                            'marginBottom': '1rem',
                            'fontSize': '1rem'
                        }),
                    html.Div([
                        dcc.Graph(
                            id='var-distribution-chart',  # Changed ID to avoid conflicts
                            figure=self._create_var_distribution_chart(),  # Renamed method
                            config={'displayModeBar': False},
                            style={
                                'height': '300px',  # Increased height for better visibility
                                'width': '100%'
                            }
                        )
                    ], style={
                        'backgroundColor': 'rgba(0, 0, 0, 0.2)',
                        'borderRadius': '8px',
                        'padding': '1rem',
                        'position': 'relative',
                        'overflow': 'hidden'
                    })
                ], style={
                    'marginTop': '0',  # Remove any top margin that might cause overlap
                    'position': 'relative'
                })
            ], style={
                'display': 'flex',
                'flexDirection': 'column',
                'gap': '0'  # Control spacing between elements
            })
        ], className='risk-card', style={
            'padding': '1.5rem',
            'height': 'auto',  # Allow natural height expansion
            'minHeight': '600px'  # Ensure minimum height for content
        })
    
    def _create_var_distribution_chart(self):
        """Create VaR distribution visualization chart - renamed to avoid confusion"""
        # Generate sample returns distribution
        returns = np.random.normal(0, 0.02, 1000)
        
        fig = go.Figure()
        
        # Histogram with improved styling
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns Distribution',
            marker=dict(
                color='#00a3b8',
                line=dict(color='#008394', width=1)
            ),
            opacity=0.7,
            hovertemplate='Return: %{x:.2%}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add VaR lines with annotations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # VaR 95% line
        fig.add_vline(
            x=var_95, 
            line=dict(color='#ffd93d', width=2, dash='dash'),
            annotation=dict(
                text=f'VaR 95%<br>{var_95:.2%}',
                font=dict(color='#ffd93d', size=11),
                yanchor='bottom',
                y=0.95,
                yref='paper'
            )
        )
        
        # VaR 99% line
        fig.add_vline(
            x=var_99, 
            line=dict(color='#ff6b6b', width=2, dash='dash'),
            annotation=dict(
                text=f'VaR 99%<br>{var_99:.2%}',
                font=dict(color='#ff6b6b', size=11),
                yanchor='bottom',
                y=0.85,
                yref='paper'
            )
        )
        
        # Add mean line for reference
        mean_return = np.mean(returns)
        fig.add_vline(
            x=mean_return,
            line=dict(color='#4CAF50', width=1, dash='dot'),
            annotation=dict(
                text=f'Mean<br>{mean_return:.2%}',
                font=dict(color='#4CAF50', size=10),
                yanchor='bottom',
                y=0.75,
                yref='paper'
            )
        )
        
        fig.update_layout(
            title=dict(
                text='Historical Returns Distribution',
                font=dict(size=14, color='white'),
                x=0.5,
                xanchor='center'
            ),
            height=300,
            margin=dict(l=50, r=30, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white'),
            xaxis=dict(
                title='Returns',
                tickformat='.1%',
                gridcolor='rgba(255,255,255,0.05)',
                showgrid=True,
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.2)'
            ),
            yaxis=dict(
                title='Frequency',
                gridcolor='rgba(255,255,255,0.05)',
                showgrid=True
            ),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(0,0,0,0.2)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def _create_correlation_matrix(self):
        """Create correlation matrix heatmap"""
        # Generate sample correlation matrix
        assets = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
        n = len(assets)
        
        # Create random correlation matrix
        corr_matrix = np.random.uniform(0.3, 0.9, (n, n))
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=assets,
            y=assets,
            colorscale='RdYlGn',
            zmid=0.5,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={'size': 10},
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            height=350,
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white')
        )
        
        return html.Div([
            html.H3('Correlation Analysis', style={'color': 'white', 'marginBottom': '1rem'}),
            html.Div([
                dcc.Graph(
                    id='correlation-matrix',
                    figure=fig,
                    config={'displayModeBar': False},
                    style={'height': '350px'}
                )
            ], style={'width': '100%', 'height': '350px', 'position': 'relative'})
        ], className='risk-card')
    
    def _create_exposure_chart(self):
        """Create exposure breakdown chart"""
        # Sample exposure data
        categories = ['Long Exposure', 'Short Exposure', 'Cash']
        values = [45, 25, 30]
        colors = ['#00ff88', '#ff6b6b', '#888888']
        
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='#000', width=2)),
            textfont=dict(color='white'),
            textposition='outside'
        )])
        
        # Add center text
        fig.add_annotation(
            text='EXPOSURE',
            x=0.5, y=0.5,
            font=dict(size=14, color='white'),
            showarrow=False
        )
        
        fig.update_layout(
            title='Portfolio Exposure',
            height=350,
            margin=dict(l=20, r=100, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True
        )
        
        return html.Div([
            html.H3('Exposure Analysis', style={'color': 'white', 'marginBottom': '1rem'}),
            html.Div([
                dcc.Graph(
                    id='exposure-chart',
                    figure=fig,
                    config={'displayModeBar': False},
                    style={'height': '350px'}
                )
            ], style={'width': '100%', 'height': '350px', 'position': 'relative'})
        ], className='risk-card')
    
    def _create_risk_history_chart(self):
        """Create historical risk metrics chart"""
        # Generate sample historical data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Risk Over Time', 'Risk Utilization'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # Portfolio risk line
        risk_values = np.random.uniform(0.005, 0.025, len(dates))
        fig.add_trace(
            go.Scatter(
                x=dates, y=risk_values,
                mode='lines',
                name='Portfolio Risk',
                line=dict(color='#00a3b8', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 163, 184, 0.1)'
            ),
            row=1, col=1
        )
        
        # Add risk limit line
        fig.add_trace(
            go.Scatter(
                x=dates, y=[0.02] * len(dates),
                mode='lines',
                name='Risk Limit',
                line=dict(color='#ff6b6b', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Risk utilization bars
        utilization = (risk_values / 0.02) * 100
        colors = ['#00ff88' if u < 70 else '#ffd93d' if u < 90 else '#ff6b6b' for u in utilization]
        
        fig.add_trace(
            go.Bar(
                x=dates[::7],  # Weekly bars
                y=utilization[::7],
                name='Utilization %',
                marker=dict(color=colors[::7])
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=50, r=30, t=50, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white'),
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(tickformat='.2%', row=1, col=1)
        fig.update_yaxes(tickformat='.0f', ticksuffix='%', row=2, col=1)
        
        return html.Div([
            html.H3('Risk History', style={'color': 'white', 'marginBottom': '1rem'}),
            html.Div([
                dcc.Graph(
                    id='risk-history-chart',
                    figure=fig,
                    config={'displayModeBar': False},
                    style={'height': '500px'}
                )
            ], style={'width': '100%', 'height': '500px', 'position': 'relative'})
        ], className='risk-card')
    
    def _create_drawdown_analysis(self):
        """Create drawdown analysis panel"""
        return html.Div([
            html.H3('Drawdown Analysis', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Current Drawdown Stats
            html.Div([
                html.Div([
                    html.Div('Current DD', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    html.Div('-8.2%', id='current-dd', style={'color': '#ffd93d', 'fontSize': '1.5rem', 'fontWeight': '600'})
                ], style={'flex': '1', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div('Max DD', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    html.Div('-12.7%', id='max-dd', style={'color': '#ff6b6b', 'fontSize': '1.5rem', 'fontWeight': '600'})
                ], style={'flex': '1', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div('Avg DD', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    html.Div('-5.3%', id='avg-dd', style={'color': '#00a3b8', 'fontSize': '1.5rem', 'fontWeight': '600'})
                ], style={'flex': '1', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div('Recovery', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    html.Div('4 days', id='recovery-time', style={'color': '#00ff88', 'fontSize': '1.5rem', 'fontWeight': '600'})
                ], style={'flex': '1', 'textAlign': 'center'})
            ], style={'display': 'flex', 'marginBottom': '1rem'}),
            
            # Drawdown periods table
            dash_table.DataTable(
                id='drawdown-periods',
                columns=[
                    {'name': 'Start', 'id': 'start'},
                    {'name': 'End', 'id': 'end'},
                    {'name': 'Depth', 'id': 'depth'},
                    {'name': 'Duration', 'id': 'duration'},
                    {'name': 'Recovery', 'id': 'recovery'}
                ],
                data=[
                    {'start': '2024-11-15', 'end': '2024-11-20', 'depth': '-12.7%', 'duration': '5 days', 'recovery': '3 days'},
                    {'start': '2024-10-02', 'end': '2024-10-08', 'depth': '-9.3%', 'duration': '6 days', 'recovery': '4 days'},
                    {'start': '2024-08-20', 'end': '2024-08-23', 'depth': '-7.1%', 'duration': '3 days', 'recovery': '2 days'}
                ],
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
                page_size=5
            )
        ], className='risk-card')
    
    def _create_stress_testing(self):
        """Create stress testing panel"""
        return html.Div([
            html.H3('Stress Testing', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Stress Test Scenarios
            html.Div([
                html.Label('Select Scenario', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                dcc.Dropdown(
                    id='stress-scenario',
                    options=[
                        {'label': 'Market Crash (-20%)', 'value': 'crash'},
                        {'label': 'Flash Crash (-10% in 1hr)', 'value': 'flash'},
                        {'label': 'Volatility Spike (3x)', 'value': 'volatility'},
                        {'label': 'Liquidity Crisis', 'value': 'liquidity'},
                        {'label': 'Correlation Breakdown', 'value': 'correlation'},
                        {'label': 'Custom Scenario', 'value': 'custom'}
                    ],
                    value='crash',
                    style={'background': '#1a1a1a', 'marginBottom': '1rem'}
                )
            ]),
            
            # Run Stress Test Button
            html.Button([
                html.I(className="fas fa-bolt", style={'marginRight': '0.5rem'}),
                'RUN STRESS TEST'
            ], id='run-stress-test-btn', style={
                'background': 'linear-gradient(45deg, #B98544, #E13A3E)',
                'border': 'none',
                'color': 'white',
                'padding': '0.75rem 1.5rem',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'fontWeight': '600',
                'width': '100%',
                'marginBottom': '1rem'
            }),
            
            # Stress Test Results
            html.Div(id='stress-test-results', children=[
                html.H4('Results', style={'color': '#00a3b8', 'marginBottom': '0.5rem'}),
                html.Div([
                    html.Div([
                        html.Span('Portfolio Impact: ', style={'color': '#b0b0b0'}),
                        html.Span('-$3,247', style={'color': '#ff6b6b', 'fontWeight': '600'})
                    ], style={'marginBottom': '0.5rem'}),
                    html.Div([
                        html.Span('Max Drawdown: ', style={'color': '#b0b0b0'}),
                        html.Span('-18.5%', style={'color': '#ff6b6b', 'fontWeight': '600'})
                    ], style={'marginBottom': '0.5rem'}),
                    html.Div([
                        html.Span('Recovery Time: ', style={'color': '#b0b0b0'}),
                        html.Span('12 days', style={'color': '#ffd93d', 'fontWeight': '600'})
                    ])
                ])
            ], style={
                'padding': '1rem',
                'background': 'rgba(0, 0, 0, 0.3)',
                'borderRadius': '8px',
                'border': '1px solid rgba(0, 131, 148, 0.3)'
            })
        ], className='risk-card')
    
    def _create_scenario_analysis(self):
        """Create scenario analysis panel"""
        return html.Div([
            html.H3('Scenario Analysis', style={'color': 'white', 'marginBottom': '1rem'}),
            
            # Monte Carlo Simulation
            html.Div([
                html.H4('Monte Carlo Simulation', style={'color': '#00a3b8', 'fontSize': '1rem', 'marginBottom': '0.5rem'}),
                
                html.Div([
                    html.Div([
                        html.Label('Simulations', style={'color': '#b0b0b0', 'fontSize': '0.75rem'}),
                        dcc.Input(
                            id='monte-carlo-sims',
                            type='number',
                            value=1000,
                            style={'width': '100px', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white'}
                        )
                    ], style={'marginRight': '1rem'}),
                    
                    html.Div([
                        html.Label('Time Horizon', style={'color': '#b0b0b0', 'fontSize': '0.75rem'}),
                        dcc.Input(
                            id='monte-carlo-days',
                            type='number',
                            value=30,
                            style={'width': '100px', 'background': '#1a1a1a', 'border': '1px solid #4a4a4a', 'color': 'white'}
                        )
                    ])
                ], style={'display': 'flex', 'marginBottom': '1rem'}),
                
                html.Button([
                    html.I(className="fas fa-dice", style={'marginRight': '0.5rem'}),
                    'RUN SIMULATION'
                ], id='run-monte-carlo-btn', style={
                    'background': 'rgba(0, 131, 148, 0.2)',
                    'border': '1px solid #008394',
                    'color': '#00a3b8',
                    'padding': '0.5rem 1rem',
                    'borderRadius': '4px',
                    'cursor': 'pointer'
                })
            ], style={'marginBottom': '1rem'}),
            
            # Results Display
            html.Div([
                html.Div('Expected Outcomes', style={'color': '#b0b0b0', 'marginBottom': '0.5rem'}),
                html.Div([
                    html.Div([
                        html.Div('Best Case', style={'color': '#00ff88', 'fontSize': '0.75rem'}),
                        html.Div('+15.3%', style={'color': '#00ff88', 'fontWeight': '600'})
                    ], style={'flex': '1', 'textAlign': 'center'}),
                    html.Div([
                        html.Div('Expected', style={'color': '#00a3b8', 'fontSize': '0.75rem'}),
                        html.Div('+3.2%', style={'color': '#00a3b8', 'fontWeight': '600'})
                    ], style={'flex': '1', 'textAlign': 'center'}),
                    html.Div([
                        html.Div('Worst Case', style={'color': '#ff6b6b', 'fontSize': '0.75rem'}),
                        html.Div('-8.7%', style={'color': '#ff6b6b', 'fontWeight': '600'})
                    ], style={'flex': '1', 'textAlign': 'center'})
                ], style={'display': 'flex'})
            ])
        ], className='risk-card')
    
    def _create_alert_system(self):
        """Create alert system panel"""
        return html.Div([
            html.H3('Risk Alerts', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                # Active Alerts
                html.Div([
                    html.H4('Active Alerts', style={'color': '#ff6b6b', 'fontSize': '1rem', 'marginBottom': '0.5rem'}),
                    html.Div(id='active-alerts', children=[
                        self._create_alert('HIGH', 'Position SOL approaching stop-loss (-4.8%)', 'danger'),
                        self._create_alert('MEDIUM', 'Daily loss at 60% of limit', 'warning'),
                        self._create_alert('LOW', 'Correlation increasing between BTC and ETH', 'info')
                    ])
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                # Alert Settings
                html.Div([
                    html.H4('Alert Settings', style={'color': '#00a3b8', 'fontSize': '1rem', 'marginBottom': '0.5rem'}),
                    dcc.Checklist(
                        id='alert-settings',
                        options=[
                            {'label': ' Position Risk Alerts', 'value': 'position'},
                            {'label': ' Drawdown Alerts', 'value': 'drawdown'},
                            {'label': ' Correlation Alerts', 'value': 'correlation'},
                            {'label': ' VaR Breach Alerts', 'value': 'var'},
                            {'label': ' Circuit Breaker Alerts', 'value': 'breaker'}
                        ],
                        value=['position', 'drawdown', 'var', 'breaker'],
                        style={'color': '#b0b0b0'}
                    )
                ], style={'flex': '1'})
            ], style={'display': 'flex'})
        ], className='risk-card')
    
    def _create_alert(self, severity, message, alert_type):
        """Create an alert message"""
        colors = {
            'danger': '#ff6b6b',
            'warning': '#ffd93d',
            'info': '#00a3b8'
        }
        
        return html.Div([
            html.Span(severity, style={
                'background': colors[alert_type],
                'color': '#000',
                'padding': '0.25rem 0.5rem',
                'borderRadius': '4px',
                'fontSize': '0.75rem',
                'fontWeight': '600',
                'marginRight': '0.5rem'
            }),
            html.Span(message, style={'color': colors[alert_type]})
        ], style={
            'padding': '0.75rem',
            'background': f'{colors[alert_type]}15',
            'border': f'1px solid {colors[alert_type]}40',
            'borderRadius': '4px',
            'marginBottom': '0.5rem'
        })
    
    def _create_risk_report(self):
        """Create risk report section"""
        return html.Div([
            html.H3('Risk Report', style={'color': 'white', 'marginBottom': '1rem'}),
            
            html.Div([
                # Report Generation
                html.Div([
                    html.Label('Report Type', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.Dropdown(
                        id='report-type',
                        options=[
                            {'label': 'Daily Risk Report', 'value': 'daily'},
                            {'label': 'Weekly Risk Summary', 'value': 'weekly'},
                            {'label': 'Monthly Risk Analysis', 'value': 'monthly'},
                            {'label': 'Custom Report', 'value': 'custom'}
                        ],
                        value='daily',
                        style={'background': '#1a1a1a'}
                    )
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                html.Div([
                    html.Label('Format', style={'color': '#b0b0b0', 'fontSize': '0.875rem'}),
                    dcc.Dropdown(
                        id='report-format',
                        options=[
                            {'label': 'PDF', 'value': 'pdf'},
                            {'label': 'CSV', 'value': 'csv'},
                            {'label': 'JSON', 'value': 'json'}
                        ],
                        value='pdf',
                        style={'background': '#1a1a1a'}
                    )
                ], style={'flex': '1', 'marginRight': '1rem'}),
                
                html.Button([
                    html.I(className="fas fa-file-download", style={'marginRight': '0.5rem'}),
                    'GENERATE REPORT'
                ], id='generate-report-btn', style={
                    'background': 'linear-gradient(45deg, #008394, #00a3b8)',
                    'border': 'none',
                    'color': 'white',
                    'padding': '0.75rem 1.5rem',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontWeight': '600',
                    'alignSelf': 'flex-end'
                })
            ], style={'display': 'flex', 'alignItems': 'flex-end', 'marginBottom': '1rem'}),
            
            # Download component
            dcc.Download(id='download-risk-report')
        ], className='risk-card')