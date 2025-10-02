"""
dashboard/callbacks/portfolio_callbacks.py
Callback functions for the Portfolio Command Center - FIXED VERSION
Handles trading interactions, position management, and performance updates
"""

from dash import Input, Output, State, callback_context, no_update, ALL
from dash.exceptions import PreventUpdate
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def register_portfolio_callbacks(app, portfolio_center):
    """Register all portfolio-related callbacks"""
    
    # ==================== Mode Switching ====================
    @app.callback(
        [Output('paper-mode-btn', 'style'),
         Output('live-mode-btn', 'style'),
         Output('live-warning', 'style')],
        [Input('paper-mode-btn', 'n_clicks'),
         Input('live-mode-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def switch_trading_mode(paper_clicks, live_clicks):
        """Switch between paper and live trading modes"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        paper_active_style = {
            'background': 'linear-gradient(90deg, #008394 0%, #00a3b8 100%)',
            'border': 'none',
            'color': 'white',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'marginRight': '1rem'
        }
        
        paper_inactive_style = {
            'background': 'rgba(0, 131, 148, 0.2)',
            'border': '1px solid #008394',
            'color': '#008394',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'marginRight': '1rem'
        }
        
        live_active_style = {
            'background': 'linear-gradient(90deg, #ff6b6b 0%, #ff4757 100%)',
            'border': 'none',
            'color': 'white',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '4px',
            'cursor': 'pointer'
        }
        
        live_inactive_style = {
            'background': 'rgba(255, 107, 107, 0.2)',
            'border': '1px solid #ff6b6b',
            'color': '#ff6b6b',
            'padding': '0.75rem 1.5rem',
            'borderRadius': '4px',
            'cursor': 'pointer'
        }
        
        warning_hidden = {'display': 'none'}
        warning_visible = {
            'color': '#ffd93d',
            'fontSize': '0.85rem',
            'marginLeft': '1rem',
            'display': 'inline'
        }
        
        if button_id == 'paper-mode-btn':
            return paper_active_style, live_inactive_style, warning_hidden
        else:
            return paper_inactive_style, live_active_style, warning_visible
    
    # ==================== Trading Panel Controls ====================
    @app.callback(
        [Output('limit-price', 'disabled'),
         Output('limit-price', 'style')],
        [Input('order-type-select', 'value')],
        prevent_initial_call=True
    )
    def toggle_limit_price(order_type):
        """Enable/disable limit price based on order type"""
        if order_type == 'LIMIT':
            return False, {'width': '100%'}
        return True, {'width': '100%', 'opacity': '0.5'}
    
    @app.callback(
        [Output('stop-loss-pct', 'style'),
         Output('take-profit-pct', 'style')],
        [Input('use-stop-loss', 'value'),
         Input('use-take-profit', 'value')],
        prevent_initial_call=True
    )
    def toggle_risk_inputs(sl_value, tp_value):
        """Show/hide risk management input fields"""
        sl_style = {'width': '60px', 'marginLeft': '0.5rem', 'display': 'inline' if sl_value else 'none'}
        tp_style = {'width': '60px', 'marginLeft': '0.5rem', 'display': 'inline' if tp_value else 'none'}
        return sl_style, tp_style
    
    # ==================== Combined Table Updates ====================
    @app.callback(
        [Output('positions-table', 'data'),
         Output('pending-orders-table', 'data'),
         Output('trade-history-table', 'data'),
         Output('balance-card-value', 'children'),
         Output('positions-card-value', 'children'),
         Output('risk-alerts-container', 'children')],
        [Input('execute-trade-btn', 'n_clicks'),
         Input('close-position-btn', 'n_clicks'),
         Input('cancel-order-btn', 'n_clicks')],
        [State('trading-pair-select', 'value'),
         State('trade-amount', 'value'),
         State('limit-price', 'value'),
         State('use-stop-loss', 'value'),
         State('stop-loss-pct', 'value'),
         State('use-take-profit', 'value'),
         State('take-profit-pct', 'value'),
         State('positions-table', 'data'),
         State('positions-table', 'selected_rows'),
         State('pending-orders-table', 'data'),
         State('pending-orders-table', 'selected_rows'),
         State('trade-history-table', 'data')],
        prevent_initial_call=True
    )
    def manage_trades_and_positions(execute_clicks, close_clicks, cancel_clicks,
                                   pair, amount, limit_price, use_sl, sl_pct, 
                                   use_tp, tp_pct, positions_data, positions_selected,
                                   pending_data, pending_selected, history_data):
        """Combined callback for all trade and position management"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Initialize data if None
        positions_data = positions_data or []
        pending_data = pending_data or []
        history_data = history_data or []
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # Execute Trade
        if button_id == 'execute-trade-btn' and pair and amount:
            # Add to pending orders (for limit orders)
            if limit_price:
                new_order = {
                    'time': current_time,
                    'symbol': pair.split('/')[0],
                    'type': 'LIMIT',
                    'side': 'BUY',
                    'price': f'${limit_price:,.2f}',
                    'amount': f'{amount}',
                    'status': 'PENDING'
                }
                pending_data.append(new_order)
            else:
                # Market order - add directly to positions
                new_position = {
                    'symbol': pair.split('/')[0],
                    'side': 'LONG',
                    'entry': f'${43000:,.2f}',  # Mock price
                    'current': f'${43100:,.2f}',
                    'size': f'{amount}',
                    'pnl': f'+${100:,.2f}',
                    'pnl_pct': '+0.23%',
                    'sl': f'${sl_pct}%' if use_sl else '-',
                    'tp': f'${tp_pct}%' if use_tp else '-'
                }
                positions_data.append(new_position)
                
                # Add to history
                new_trade = {
                    'time': current_time,
                    'symbol': pair.split('/')[0],
                    'side': 'BUY',
                    'price': f'${43000:,.2f}',
                    'amount': f'{amount}',
                    'total': f'${amount * 43000:,.2f}',
                    'fee': f'${amount * 43000 * 0.0026:,.2f}',
                    'pnl': '-'
                }
                history_data.append(new_trade)
        
        # Close Position
        elif button_id == 'close-position-btn' and positions_selected:
            for idx in sorted(positions_selected, reverse=True):
                if idx < len(positions_data):
                    closed_position = positions_data.pop(idx)
                    # Add closing trade to history
                    new_trade = {
                        'time': current_time,
                        'symbol': closed_position['symbol'],
                        'side': 'SELL',
                        'price': closed_position['current'],
                        'amount': closed_position['size'],
                        'total': '-',
                        'fee': '-',
                        'pnl': closed_position['pnl']
                    }
                    history_data.append(new_trade)
        
        # Cancel Order
        elif button_id == 'cancel-order-btn' and pending_selected:
            for idx in sorted(pending_selected, reverse=True):
                if idx < len(pending_data):
                    pending_data.pop(idx)
        
        # Update metrics
        balance = f'${10000 + len(positions_data) * 100:,.2f}'
        positions_count = str(len(positions_data))
        
        # Risk alerts
        alerts = []
        if len(positions_data) > 3:
            alerts.append(
                f'⚠️ High exposure: {len(positions_data)} open positions'
            )
        
        return positions_data, pending_data, history_data, balance, positions_count, alerts
    
    # ==================== Real-time Updates ====================
    @app.callback(
        [Output('daily-pnl-card-value', 'children'),
         Output('daily-pnl-card-subtitle', 'children'),
         Output('winrate-card-value', 'children'),
         Output('risk-card-value', 'children'),
         Output('risk-card-subtitle', 'children')],
        [Input('interval-5s', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_real_time_metrics(n_intervals):
        """Update real-time portfolio metrics"""
        # Mock real-time updates
        daily_pnl = np.random.uniform(-500, 500)
        daily_pnl_str = f'{"+" if daily_pnl >= 0 else ""}${abs(daily_pnl):.2f}'
        daily_pnl_pct = f'{daily_pnl/10000*100:.2f}%'
        
        win_rate = np.random.uniform(40, 60)
        win_rate_str = f'{win_rate:.1f}%'
        
        risk_levels = ['Low', 'Medium', 'High']
        risk_level = np.random.choice(risk_levels, p=[0.6, 0.3, 0.1])
        
        risk_status = {
            'Low': 'All systems normal',
            'Medium': 'Elevated volatility detected',
            'High': 'Risk limits approaching'
        }
        
        return (
            daily_pnl_str,
            daily_pnl_pct,
            win_rate_str,
            risk_level,
            risk_status[risk_level]
        )
    
    # ==================== ML/RL Signal Updates ====================
    @app.callback(
        [Output('ml-prediction', 'children'),
         Output('ml-prediction', 'style'),
         Output('rl-action', 'children'),
         Output('rl-action', 'style')],
        [Input('trading-pair-select', 'value'),
         Input('interval-10s', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_ai_signals(selected_pair, n_intervals):
        """Update ML and RL predictions for selected pair"""
        # Mock predictions
        ml_predictions = ['BULLISH', 'BEARISH', 'NEUTRAL']
        ml_colors = {'BULLISH': '#00ff88', 'BEARISH': '#ff6b6b', 'NEUTRAL': '#ffd93d'}
        
        rl_actions = ['BUY', 'SELL', 'HOLD']
        rl_colors = {'BUY': '#00ff88', 'SELL': '#ff6b6b', 'HOLD': '#ffd93d'}
        
        ml_pred = np.random.choice(ml_predictions)
        rl_act = np.random.choice(rl_actions)
        
        return (
            ml_pred,
            {'color': ml_colors[ml_pred], 'fontWeight': 'bold'},
            rl_act,
            {'color': rl_colors[rl_act], 'fontWeight': 'bold'}
        )
    
    # ==================== Performance Charts Update ====================
    @app.callback(
        Output('performance-chart', 'figure'),
        [Input('interval-10s', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_performance_chart(n_intervals):
        """Update the performance chart"""
        import plotly.graph_objs as go
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        values = 10000 + np.cumsum(np.random.randn(len(dates)) * 50)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00a3b8', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 163, 184, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=[10000] * len(dates),
            mode='lines',
            name='Initial Value',
            line=dict(color='#888', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='Portfolio Performance',
            height=350,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat='$,.0f'),
            hovermode='x unified'
        )
        
        return fig
    
    # ==================== Asset Allocation Update ====================
    @app.callback(
        Output('allocation-chart', 'figure'),
        [Input('interval-10s', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_allocation_chart(n_intervals):
        """Update the asset allocation chart"""
        import plotly.graph_objs as go
        
        # Sample allocation data
        labels = ['BTC', 'ETH', 'SOL', 'Cash']
        values = [30 + np.random.randint(-5, 5), 
                 25 + np.random.randint(-5, 5), 
                 20 + np.random.randint(-5, 5), 
                 25]
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
            height=400,  # Default height
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
                x=1.05
            ),
            autosize=True  # Allow responsive sizing
        )
        
        return fig

    # ==================== Export Functionality ====================
    @app.callback(
        Output('download-trades', 'data'),
        [Input('export-trades-btn', 'n_clicks')],
        [State('trade-history-table', 'data')],
        prevent_initial_call=True
    )
    def export_trades(n_clicks, trade_data):
        """Export trade history to CSV"""
        if not n_clicks or not trade_data:
            raise PreventUpdate
        
        df = pd.DataFrame(trade_data)
        return dict(content=df.to_csv(index=False), filename=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")