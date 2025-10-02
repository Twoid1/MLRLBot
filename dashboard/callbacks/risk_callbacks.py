"""
dashboard/callbacks/risk_callbacks.py
Callback functions for the Risk Management Center - FIXED VERSION
Handles risk monitoring, alerts, and control systems
"""

from dash import html, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def register_risk_callbacks(app, risk_center):
    """Register all risk management callbacks"""
    
    # ==================== Real-time Risk Updates ====================
    @app.callback(
        [Output('system-risk-level', 'children'),
         Output('system-risk-level', 'style'),
         Output('risk-last-update', 'children'),
         Output('risk-portfolio-risk-card-value', 'children'),  # FIXED: Added 'risk-' prefix
         Output('risk-var-1d-card-value', 'children'),  # FIXED: Added 'risk-' prefix
         Output('risk-drawdown-card-value', 'children'),  # FIXED: Added 'risk-' prefix
         Output('risk-positions-at-risk-card-value', 'children'),  # FIXED: Added 'risk-' prefix
         Output('risk-correlation-card-value', 'children'),  # FIXED: Added 'risk-' prefix
         Output('risk-daily-pnl-card-value', 'children')],  # FIXED: Added 'risk-' prefix
        [Input('interval-5s', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_risk_metrics(n_intervals):
        """Update real-time risk metrics"""
        # In production, this will connect to:
        # - risk_manager.calculate_portfolio_risk()
        # - risk_manager.calculate_var()
        # - portfolio.get_drawdown()
        
        # Simulate risk calculations
        portfolio_risk = np.random.uniform(0.01, 0.025)
        var_1d = np.random.uniform(1500, 3000)
        current_dd = np.random.uniform(-0.12, -0.05)
        positions_at_risk = np.random.randint(0, 5)
        daily_pnl = np.random.uniform(-1000, 500)
        
        # Determine system risk level
        if portfolio_risk > 0.02:
            risk_level = 'HIGH'
            risk_color = '#ff6b6b'
        elif portfolio_risk > 0.015:
            risk_level = 'MODERATE'
            risk_color = '#ffd93d'
        else:
            risk_level = 'LOW'
            risk_color = '#00ff88'
        
        # Correlation risk
        corr_levels = ['Low', 'Medium', 'High']
        corr_risk = np.random.choice(corr_levels, p=[0.6, 0.3, 0.1])
        
        return (
            risk_level,
            {'color': risk_color, 'fontWeight': '700', 'fontSize': '1.25rem'},
            datetime.now().strftime('%H:%M:%S'),
            f'{portfolio_risk:.1%}',
            f'${var_1d:,.0f}',
            f'{current_dd:.1%}',
            str(positions_at_risk),
            corr_risk,
            f'{"+" if daily_pnl >= 0 else ""}${abs(daily_pnl):,.0f}'
        )
    
    # ==================== Apply Risk Settings ====================
    @app.callback(
        Output('apply-risk-settings-btn', 'children'),
        [Input('apply-risk-settings-btn', 'n_clicks')],
        [State('max-portfolio-risk-slider', 'value'),
         State('max-position-size-slider', 'value'),
         State('max-drawdown-slider', 'value'),
         State('daily-loss-limit', 'value')],
        prevent_initial_call=True
    )
    def apply_risk_settings(n_clicks, max_risk, max_position, max_dd, daily_limit):
        """Apply risk management settings"""
        if not n_clicks:
            raise PreventUpdate
        
        # In production, this will connect to:
        # - risk_manager.update_parameters()
        
        # Simulate applying settings
        time.sleep(0.5)  # Simulate processing
        
        return [
            html.I(className="fas fa-check", style={'marginRight': '0.5rem'}),
            'SETTINGS APPLIED'
        ]
    
    # ==================== Circuit Breakers ====================
    @app.callback(
        [Output('drawdown-breaker-status', 'children'),
         Output('drawdown-breaker-status', 'style'),
         Output('daily-loss-breaker-status', 'children'),
         Output('daily-loss-breaker-status', 'style'),
         Output('correlation-breaker-status', 'children'),
         Output('correlation-breaker-status', 'style')],
        [Input('drawdown-breaker', 'value'),
         Input('daily-loss-breaker', 'value'),
         Input('correlation-breaker', 'value'),
         Input('interval-10s', 'n_intervals')],
        prevent_initial_call=False
    )
    def update_circuit_breakers(dd_enabled, daily_enabled, corr_enabled, n_intervals):
        """Update circuit breaker status"""
        # In production, this will connect to:
        # - risk_manager.get_circuit_breaker_status()
        
        # Simulate breaker status
        current_dd = np.random.uniform(-0.12, -0.02)
        daily_loss = np.random.uniform(-0.04, 0.02)
        max_corr = np.random.uniform(0.4, 0.75)
        
        # Drawdown breaker
        if 'enabled' in (dd_enabled or []):
            if abs(current_dd) > 0.15:
                dd_status = 'TRIGGERED!'
                dd_style = {
                    'color': '#ff6b6b',
                    'background': 'rgba(255, 107, 107, 0.1)',
                    'border': '1px solid rgba(255, 107, 107, 0.3)',
                    'padding': '0.5rem',
                    'borderRadius': '4px',
                    'fontSize': '0.875rem'
                }
            else:
                dd_status = f'Status: ARMED ({current_dd:.1%} of -15%)'
                dd_style = {
                    'color': '#00ff88',
                    'background': 'rgba(0, 255, 136, 0.1)',
                    'border': '1px solid rgba(0, 255, 136, 0.3)',
                    'padding': '0.5rem',
                    'borderRadius': '4px',
                    'fontSize': '0.875rem'
                }
        else:
            dd_status = 'Status: DISABLED'
            dd_style = {
                'color': '#888',
                'background': 'rgba(136, 136, 136, 0.1)',
                'border': '1px solid rgba(136, 136, 136, 0.3)',
                'padding': '0.5rem',
                'borderRadius': '4px',
                'fontSize': '0.875rem'
            }
        
        # Daily loss breaker
        if 'enabled' in (daily_enabled or []):
            if daily_loss < -0.05:
                daily_status = 'TRIGGERED!'
                daily_style = dd_style  # Same red style
            else:
                daily_status = f'Status: ARMED ({daily_loss:.1%} of -5%)'
                daily_style = dd_style  # Same green style
        else:
            daily_status = 'Status: DISABLED'
            daily_style = dd_style  # Same gray style
        
        # Correlation breaker
        if 'enabled' in (corr_enabled or []):
            if max_corr > 0.70:
                corr_status = 'TRIGGERED!'
                corr_style = dd_style  # Same red style
            else:
                corr_status = f'Status: ARMED ({max_corr:.0%} of 70%)'
                corr_style = dd_style  # Same green style
        else:
            corr_status = 'Status: DISABLED'
            corr_style = dd_style  # Same gray style
        
        return (dd_status, dd_style, daily_status, daily_style, corr_status, corr_style)
    
    # ==================== Emergency Stop ====================
    @app.callback(
        [Output('emergency-stop-btn', 'children'),
         Output('emergency-stop-btn', 'style')],
        [Input('emergency-stop-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def emergency_stop(n_clicks):
        """Handle emergency stop button"""
        if not n_clicks:
            raise PreventUpdate
        
        # In production, this will connect to:
        # - executor.cancel_all_orders()
        # - portfolio.close_all_positions()
        # - risk_manager.trigger_emergency_stop()
        
        return (
            [
                html.I(className="fas fa-check-circle", style={'marginRight': '0.5rem'}),
                'TRADING STOPPED - ALL POSITIONS CLOSED'
            ],
            {
                'background': 'rgba(255, 107, 107, 0.2)',
                'border': '2px solid #ff6b6b',
                'color': '#ff6b6b',
                'padding': '1rem',
                'borderRadius': '4px',
                'cursor': 'not-allowed',
                'fontWeight': '700',
                'width': '100%',
                'fontSize': '1rem'
            }
        )
    
    # ==================== Position Limits Update ====================
    @app.callback(
        Output('position-limits-table', 'data'),
        [Input('interval-10s', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_position_limits(n_intervals):
        """Update position limits table"""
        # In production, this will connect to:
        # - portfolio.get_positions()
        # - risk_manager.get_position_limits()
        
        # Generate mock data
        assets = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
        data = []
        
        for asset in assets:
            current = np.random.uniform(0, 0.05)
            max_size = 0.05
            utilization = (current / max_size) * 100
            
            data.append({
                'asset': asset,
                'current': f'{current:.3f}',
                'max_size': f'{max_size:.3f}',
                'utilization': f'{utilization:.1f}%',
                'status': 'WARNING' if utilization > 80 else 'OK'
            })
        
        return data
    
    # ==================== Stress Testing ====================
    @app.callback(
        Output('stress-test-results', 'children'),
        [Input('run-stress-test-btn', 'n_clicks')],
        [State('stress-scenario', 'value')],
        prevent_initial_call=True
    )
    def run_stress_test(n_clicks, scenario):
        """Run stress test simulation"""
        if not n_clicks:
            raise PreventUpdate
        
        # In production, this will connect to:
        # - risk_manager.run_stress_test(scenario)
        
        # Simulate stress test results
        scenario_impacts = {
            'crash': {'impact': -5000, 'dd': -22, 'recovery': 18},
            'flash': {'impact': -3000, 'dd': -15, 'recovery': 8},
            'volatility': {'impact': -2500, 'dd': -18, 'recovery': 12},
            'liquidity': {'impact': -4000, 'dd': -20, 'recovery': 15},
            'correlation': {'impact': -2000, 'dd': -12, 'recovery': 10},
            'custom': {'impact': -3500, 'dd': -17, 'recovery': 14}
        }
        
        result = scenario_impacts.get(scenario, scenario_impacts['crash'])
        
        return [
            html.H4('Results', style={'color': '#00a3b8', 'marginBottom': '0.5rem'}),
            html.Div([
                html.Div([
                    html.Span('Portfolio Impact: ', style={'color': '#b0b0b0'}),
                    html.Span(f'${result["impact"]:,.0f}', style={'color': '#ff6b6b', 'fontWeight': '600'})
                ], style={'marginBottom': '0.5rem'}),
                html.Div([
                    html.Span('Max Drawdown: ', style={'color': '#b0b0b0'}),
                    html.Span(f'{result["dd"]}%', style={'color': '#ff6b6b', 'fontWeight': '600'})
                ], style={'marginBottom': '0.5rem'}),
                html.Div([
                    html.Span('Recovery Time: ', style={'color': '#b0b0b0'}),
                    html.Span(f'{result["recovery"]} days', style={'color': '#ffd93d', 'fontWeight': '600'})
                ])
            ])
        ]
    
    # ==================== Monte Carlo Simulation ====================
    # Note: This callback doesn't exist in the component, commenting out
    # If you need it, add id='monte-carlo-results' to the component
    
    # ==================== Risk History Chart Update ====================
    @app.callback(
        Output('risk-history-chart', 'figure'),
        [Input('interval-10s', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_risk_history(n_intervals):
        """Update risk history chart"""
        # Generate sample data
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
        
        return fig
    
    # ==================== Active Alerts Update ====================
    @app.callback(
        Output('active-alerts', 'children'),
        [Input('interval-5s', 'n_intervals')],
        [State('alert-settings', 'value')],
        prevent_initial_call=True
    )
    def update_alerts(n_intervals, alert_settings):
        """Update active alerts"""
        # In production, this will connect to:
        # - risk_manager.get_active_alerts()
        
        alerts = []
        
        if 'position' in (alert_settings or []):
            # Check for position alerts
            if np.random.random() > 0.5:
                alerts.append(
                    risk_center._create_alert(
                        'HIGH',
                        f'Position {np.random.choice(["BTC", "ETH", "SOL"])} approaching stop-loss ({np.random.uniform(-5, -2):.1f}%)',
                        'danger'
                    )
                )
        
        if 'drawdown' in (alert_settings or []):
            # Check for drawdown alerts
            current_dd = np.random.uniform(-0.12, -0.05)
            if abs(current_dd) > 0.10:
                alerts.append(
                    risk_center._create_alert(
                        'MEDIUM',
                        f'Drawdown at {current_dd:.1%} (limit: -15%)',
                        'warning'
                    )
                )
        
        if 'correlation' in (alert_settings or []):
            # Check for correlation alerts
            if np.random.random() > 0.7:
                alerts.append(
                    risk_center._create_alert(
                        'LOW',
                        'Correlation increasing between BTC and ETH',
                        'info'
                    )
                )
        
        if not alerts:
            alerts = [
                html.Div('No active alerts', style={
                    'color': '#00ff88',
                    'padding': '0.75rem',
                    'background': 'rgba(0, 255, 136, 0.1)',
                    'border': '1px solid rgba(0, 255, 136, 0.3)',
                    'borderRadius': '4px'
                })
            ]
        
        return alerts
    
    # ==================== Generate Risk Report ====================
    @app.callback(
        Output('download-risk-report', 'data'),
        [Input('generate-report-btn', 'n_clicks')],
        [State('report-type', 'value'),
         State('report-format', 'value')],
        prevent_initial_call=True
    )
    def generate_risk_report(n_clicks, report_type, report_format):
        """Generate risk report"""
        if not n_clicks:
            raise PreventUpdate
        
        # In production, this will connect to:
        # - risk_manager.generate_report(report_type, report_format)
        
        # Create sample report data
        report_data = {
            'Report Type': report_type,
            'Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Portfolio Risk': '1.5%',
            'VaR (95%)': '$2,047',
            'Current Drawdown': '-8.2%',
            'Positions at Risk': '3',
            'Daily P&L': '-$523',
            'Risk Level': 'MODERATE'
        }
        
        # Convert to appropriate format
        if report_format == 'csv':
            df = pd.DataFrame([report_data])
            return dict(
                content=df.to_csv(index=False),
                filename=f"risk_report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        elif report_format == 'json':
            return dict(
                content=json.dumps(report_data, indent=2),
                filename=f"risk_report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        else:  # PDF would require additional library
            df = pd.DataFrame([report_data])
            return dict(
                content=df.to_csv(index=False),
                filename=f"risk_report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )