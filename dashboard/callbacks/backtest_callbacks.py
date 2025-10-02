"""
dashboard/callbacks/backtest_callbacks.py
Callback functions for the Backtesting Laboratory
Handles strategy testing, walk-forward analysis, and performance updates
"""

from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objs as go

def register_backtest_callbacks(app, backtest_lab):
    """Register all backtesting-related callbacks"""
    
    # ==================== Run Backtest ====================
    @app.callback(
        [Output('run-backtest-btn', 'disabled'),
         Output('pause-backtest-btn', 'disabled'),
         Output('stop-backtest-btn', 'disabled'),
         Output('backtest-progress-bar', 'style'),
         Output('backtest-progress-text', 'children'),
         Output('backtest-status', 'children'),
         Output('backtest-status', 'style')],
        [Input('run-backtest-btn', 'n_clicks'),
         Input('pause-backtest-btn', 'n_clicks'),
         Input('stop-backtest-btn', 'n_clicks'),
         Input('reset-backtest-btn', 'n_clicks')],
        [State('strategy-select', 'value'),
         State('backtest-assets', 'value'),
         State('backtest-timeframe', 'value'),
         State('backtest-start-date', 'date'),
         State('backtest-end-date', 'date'),
         State('initial-capital', 'value')],
        prevent_initial_call=True
    )
    def control_backtest(run_clicks, pause_clicks, stop_clicks, reset_clicks,
                        strategy, assets, timeframe, start_date, end_date, capital):
        """Control backtest execution"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Default styles
        progress_style = {
            'width': '0%',
            'height': '100%',
            'background': 'linear-gradient(90deg, #008394, #00a3b8)',
            'borderRadius': '4px',
            'transition': 'width 0.3s ease'
        }
        
        status_style = {
            'marginTop': '1rem',
            'padding': '0.75rem',
            'background': 'rgba(0, 131, 148, 0.1)',
            'border': '1px solid rgba(0, 131, 148, 0.3)',
            'borderRadius': '4px',
            'color': '#00a3b8',
            'fontSize': '0.875rem'
        }
        
        if button_id == 'run-backtest-btn':
            # In production, this will connect to:
            # - backtester.run_backtest()
            # - strategy.execute()
            
            # Simulate running
            progress_style['width'] = '25%'
            
            return (
                True,   # Disable run button
                False,  # Enable pause button
                False,  # Enable stop button
                progress_style,
                '25%',
                f'Running backtest: {strategy} on {", ".join(assets)} ({timeframe})',
                status_style
            )
        
        elif button_id == 'pause-backtest-btn':
            return (
                False,  # Enable run button
                True,   # Disable pause button
                False,  # Enable stop button
                progress_style,
                '25%',
                'Backtest paused',
                {**status_style, 'color': '#ffd93d', 'border': '1px solid #ffd93d'}
            )
        
        elif button_id == 'stop-backtest-btn':
            progress_style['width'] = '0%'
            return (
                False,  # Enable run button
                True,   # Disable pause button
                True,   # Disable stop button
                progress_style,
                '0%',
                'Backtest stopped',
                {**status_style, 'color': '#ff6b6b', 'border': '1px solid #ff6b6b'}
            )
        
        else:  # Reset
            progress_style['width'] = '0%'
            return (
                False,  # Enable run button
                True,   # Disable pause button
                True,   # Disable stop button
                progress_style,
                '0%',
                'Ready to start backtesting',
                status_style
            )
    
    # ==================== Update Metrics ====================
    @app.callback(
        [Output('total-return-metric', 'children'),
         Output('sharpe-ratio-metric', 'children'),
         Output('max-drawdown-metric', 'children'),
         Output('win-rate-metric', 'children'),
         Output('profit-factor-metric', 'children'),
         Output('total-trades-metric', 'children'),
         Output('sortino-metric', 'children'),
         Output('calmar-metric', 'children'),
         Output('avg-win-loss-metric', 'children'),
         Output('recovery-metric', 'children')],
        [Input('backtest-progress-bar', 'style')],
        prevent_initial_call=True
    )
    def update_metrics(progress_style):
        """Update performance metrics during backtest"""
        # In production, this will connect to:
        # - metrics.calculate_performance()
        # - backtester.get_results()
        
        # Simulate metrics
        progress = float(progress_style.get('width', '0%').replace('%', ''))
        
        if progress > 0:
            total_return = np.random.uniform(-20, 50) * (progress / 100)
            sharpe = np.random.uniform(0.5, 2.5) * (progress / 100)
            max_dd = np.random.uniform(-25, -5) * (progress / 100)
            win_rate = np.random.uniform(35, 65)
            profit_factor = np.random.uniform(0.8, 2.0)
            trades = int(np.random.uniform(50, 500) * (progress / 100))
            
            return (
                f'{total_return:.2f}%',
                f'{sharpe:.2f}',
                f'{max_dd:.2f}%',
                f'{win_rate:.2f}%',
                f'{profit_factor:.2f}',
                str(trades),
                f'{sharpe * 0.8:.2f}',  # Sortino
                f'{abs(total_return/max_dd):.2f}',  # Calmar
                f'{profit_factor * 1.2:.2f}',  # Avg Win/Loss
                f'{total_return/abs(max_dd):.2f}'  # Recovery
            )
        
        return ['0.00%'] * 6 + ['0.00'] * 4
    
    # ==================== Update Trade Analysis ====================
    @app.callback(
        [Output('winning-trades', 'children'),
         Output('losing-trades', 'children'),
         Output('breakeven-trades', 'children'),
         Output('avg-duration', 'children'),
         Output('max-duration', 'children'),
         Output('min-duration', 'children')],
        [Input('backtest-progress-bar', 'style')],
        prevent_initial_call=True
    )
    def update_trade_analysis(progress_style):
        """Update trade analysis statistics"""
        progress = float(progress_style.get('width', '0%').replace('%', ''))
        
        if progress > 0:
            total_trades = int(np.random.uniform(50, 500) * (progress / 100))
            win_rate = np.random.uniform(0.35, 0.65)
            
            winning = int(total_trades * win_rate)
            losing = int(total_trades * (1 - win_rate) * 0.9)
            breakeven = total_trades - winning - losing
            
            return (
                str(winning),
                str(losing),
                str(breakeven),
                f'{np.random.randint(1, 48)}h {np.random.randint(0, 59)}m',
                f'{np.random.randint(24, 168)}h {np.random.randint(0, 59)}m',
                f'{np.random.randint(1, 10)}m'
            )
        
        return ['0'] * 3 + ['0h 0m'] * 3
    
    # ==================== Update Charts ====================
    @app.callback(
        Output('equity-curve-chart', 'figure'),
        [Input('backtest-progress-bar', 'style')],
        [State('backtest-start-date', 'date'),
         State('backtest-end-date', 'date'),
         State('initial-capital', 'value')],
        prevent_initial_call=True
    )
    def update_equity_curve(progress_style, start_date, end_date, capital):
        """Update equity curve chart"""
        progress = float(progress_style.get('width', '0%').replace('%', ''))
        
        if progress > 0:
            # Generate progressive equity curve
            days = int(progress * 3.65)  # Progress percentage of year
            dates = pd.date_range(start=start_date, periods=days, freq='D')
            
            equity = [capital]
            for _ in range(len(dates) - 1):
                change = np.random.randn() * capital * 0.01
                equity.append(max(0, equity[-1] + change))
            
            fig = go.Figure()
            
            # Portfolio equity
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00a3b8', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 163, 184, 0.1)'
            ))
            
            # Benchmark
            benchmark = [capital * (1 + 0.0003) ** i for i in range(len(dates))]
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
            
            return fig
        
        raise PreventUpdate
    
    # ==================== Walk-Forward Analysis ====================
    @app.callback(
        Output('walk-forward-results', 'data'),
        [Input('run-walk-forward-btn', 'n_clicks')],
        [State('training-period', 'value'),
         State('test-period', 'value'),
         State('num-folds', 'value')],
        prevent_initial_call=True
    )
    def run_walk_forward(n_clicks, train_days, test_days, num_folds):
        """Run walk-forward analysis"""
        if not n_clicks:
            raise PreventUpdate
        
        # In production, this will connect to:
        # - walk_forward.run_analysis()
        
        # Generate mock results
        results = []
        start_date = datetime(2024, 1, 1)
        
        for i in range(num_folds):
            train_start = start_date + timedelta(days=i * test_days)
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)
            
            results.append({
                'fold': f'Fold {i+1}',
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'train_return': f'{np.random.uniform(-10, 30):.2f}%',
                'test_return': f'{np.random.uniform(-5, 20):.2f}%',
                'sharpe': f'{np.random.uniform(0.5, 2.0):.2f}'
            })
        
        return results
    
    # ==================== Trade Log Update ====================
    @app.callback(
        Output('trade-log-table', 'data'),
        [Input('backtest-progress-bar', 'style')],
        prevent_initial_call=True
    )
    def update_trade_log(progress_style):
        """Update trade log with executed trades"""
        progress = float(progress_style.get('width', '0%').replace('%', ''))
        
        if progress > 0:
            # Generate mock trades
            num_trades = int(progress / 10)  # Add trades progressively
            trades = []
            
            for i in range(num_trades):
                entry_price = np.random.uniform(40000, 45000)
                exit_price = entry_price * (1 + np.random.uniform(-0.03, 0.03))
                pnl = (exit_price - entry_price) * 0.01
                
                trades.append({
                    'time': (datetime.now() - timedelta(hours=i*24)).strftime('%Y-%m-%d %H:%M'),
                    'symbol': np.random.choice(['BTC', 'ETH', 'SOL']),
                    'side': np.random.choice(['LONG', 'SHORT']),
                    'entry': f'${entry_price:.2f}',
                    'exit': f'${exit_price:.2f}',
                    'quantity': f'{np.random.uniform(0.01, 0.1):.3f}',
                    'pnl': f'${pnl:.2f}',
                    'pnl_pct': f'{(exit_price/entry_price - 1)*100:.2f}%',
                    'duration': f'{np.random.randint(1, 48)}h',
                    'signal': np.random.choice(['ML', 'RL', 'Both'])
                })
            
            return trades
        
        return []
    
    # ==================== Strategy Comparison ====================
    @app.callback(
        Output('strategy-comparison-table', 'data'),
        [Input('run-backtest-btn', 'n_clicks')],
        [State('strategy-select', 'value')],
        prevent_initial_call=True
    )
    def update_strategy_comparison(n_clicks, current_strategy):
        """Update strategy comparison table"""
        if not n_clicks:
            raise PreventUpdate
        
        # Generate comparison data
        strategies = ['ml_dqn', 'ml_only', 'dqn_only', 'ma_cross', 'rsi_reversion']
        comparison_data = []
        
        for strategy in strategies:
            is_current = strategy == current_strategy
            
            comparison_data.append({
                'strategy': strategy.replace('_', ' ').title(),
                'total_return': f'{np.random.uniform(-20, 50):.2f}%',
                'sharpe_ratio': f'{np.random.uniform(0.5, 2.5):.2f}',
                'max_dd': f'{np.random.uniform(-25, -5):.2f}%',
                'win_rate': f'{np.random.uniform(35, 65):.1f}%',
                'trades': str(np.random.randint(50, 500)),
                'profit_factor': f'{np.random.uniform(0.8, 2.0):.2f}'
            })
        
        return comparison_data
    
    # ==================== Export Results ====================
    @app.callback(
        Output('download-backtest-results', 'data'),
        [Input('export-backtest-btn', 'n_clicks')],
        [State('strategy-comparison-table', 'data'),
         State('trade-log-table', 'data')],
        prevent_initial_call=True
    )
    def export_results(n_clicks, comparison_data, trade_log):
        """Export backtest results to CSV"""
        if not n_clicks:
            raise PreventUpdate
        
        # Create comprehensive results file
        results = {
            'Strategy Comparison': pd.DataFrame(comparison_data),
            'Trade Log': pd.DataFrame(trade_log)
        }
        
        # In production, combine multiple sheets or create zip
        df = pd.DataFrame(comparison_data)
        
        return dict(
            content=df.to_csv(index=False),
            filename=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )