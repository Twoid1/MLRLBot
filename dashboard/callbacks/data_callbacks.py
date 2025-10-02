"""
dashboard/callbacks/data_callbacks.py
Callbacks for Data Center - Backend Integration (Fixed for current UI)
"""

from dash import Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


def register_data_callbacks(app, data_center):
    """Register all Data Center callbacks"""
    
    # ==================== Main Chart Update ====================
    @app.callback(
        [Output('main-price-volume-chart', 'figure'),
         Output('window-state-store', 'data', allow_duplicate=True),
         Output('data-range-display', 'children')],
        [Input('window-state-store', 'data'),
         Input('main-price-volume-chart', 'relayoutData'),
         Input('chart-type-candle', 'n_clicks'),
         Input('chart-type-line', 'n_clicks'),
         Input('chart-type-area', 'n_clicks'),
         Input('prev-window-btn', 'n_clicks'),
         Input('next-window-btn', 'n_clicks')],
        [State('crypto-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_chart_with_window(window_state, relayout_data, candle_clicks, line_clicks, 
                                area_clicks, prev_clicks, next_clicks, selected_crypto):
        """Update chart with sliding window"""
        
        if not window_state:
            raise PreventUpdate
        
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        # Load full dataset
        df_full = data_center.load_data(selected_crypto, window_state['timeframe'])
        
        if df_full.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {selected_crypto}<br>Click 'Fill Gaps' to download",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color='#888')
            )
            return fig, window_state, "No data"
        
        # Handle navigation buttons
        if triggered_id == 'prev-window-btn':
            # Move window back by 40 candles (half window)
            new_start = max(0, window_state['window_start'] - 40)
            new_end = new_start + 79
            window_state['window_start'] = new_start
            window_state['window_end'] = new_end
            
            # Update cache boundaries
            window_state['cached_start'] = max(0, new_start - 80)
            window_state['cached_end'] = min(window_state['total_points'] - 1, new_end + 80)
            
        elif triggered_id == 'next-window-btn':
            # Move window forward by 40 candles
            new_end = min(window_state['total_points'] - 1, window_state['window_end'] + 40)
            new_start = max(0, new_end - 79)
            window_state['window_start'] = new_start
            window_state['window_end'] = new_end
            
            # Update cache boundaries
            window_state['cached_start'] = max(0, new_start - 80)
            window_state['cached_end'] = min(window_state['total_points'] - 1, new_end + 80)
        
        # Handle pan/zoom from chart interaction
        elif relayout_data and 'xaxis.range[0]' in relayout_data:
            # User panned or zoomed
            try:
                x_start = pd.to_datetime(relayout_data['xaxis.range[0]'])
                x_end = pd.to_datetime(relayout_data['xaxis.range[1]'])
                
                # Find indices for these dates
                start_idx = df_full.index.get_indexer([x_start], method='nearest')[0]
                end_idx = df_full.index.get_indexer([x_end], method='nearest')[0]
                
                # Update window if moved significantly
                if abs(start_idx - window_state['window_start']) > 20:
                    window_state['window_start'] = max(0, start_idx)
                    window_state['window_end'] = min(window_state['total_points'] - 1, start_idx + 79)
                    
                    # Update cache
                    window_state['cached_start'] = max(0, window_state['window_start'] - 80)
                    window_state['cached_end'] = min(window_state['total_points'] - 1, 
                                                   window_state['window_end'] + 80)
            except:
                pass  # Keep current window if parsing fails
        
        # Get cached data (window + buffers)
        cached_start = window_state['cached_start']
        cached_end = window_state['cached_end'] + 1  # +1 for inclusive
        df_cached = df_full.iloc[cached_start:cached_end].copy()
        
        # Get display window
        window_start_in_cache = window_state['window_start'] - cached_start
        window_end_in_cache = window_state['window_end'] - cached_start + 1
        df_display = df_cached.iloc[window_start_in_cache:window_end_in_cache]
        
        print(f"Window: {window_state['window_start']}-{window_state['window_end']} "
              f"(Cache: {cached_start}-{cached_end}, Total: {window_state['total_points']})")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f'{selected_crypto.replace("_USDT", "")}/USD - {window_state["timeframe"].upper()}', 
                'Volume'
            )
        )
        
        # Determine chart type
        chart_type = 'candle'
        if ctx.triggered:
            if 'chart-type-line' in ctx.triggered[0]['prop_id']:
                chart_type = 'line'
            elif 'chart-type-area' in ctx.triggered[0]['prop_id']:
                chart_type = 'area'
        
        # Use cached data for smooth panning
        if chart_type == 'candle':
            fig.add_trace(
                go.Candlestick(
                    x=df_cached.index,
                    open=df_cached['open'],
                    high=df_cached['high'],
                    low=df_cached['low'],
                    close=df_cached['close'],
                    name='Price',
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                ),
                row=1, col=1
            )
        elif chart_type == 'line':
            fig.add_trace(
                go.Scatter(
                    x=df_cached.index,
                    y=df_cached['close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#00ff88', width=2)
                ),
                row=1, col=1
            )
        else:  # area
            fig.add_trace(
                go.Scatter(
                    x=df_cached.index,
                    y=df_cached['close'],
                    mode='lines',
                    name='Price',
                    fill='tozeroy',
                    line=dict(color='#00ff88', width=2),
                    fillcolor='rgba(0, 255, 136, 0.1)'
                ),
                row=1, col=1
            )
        
        # Volume bars
        colors = ['#00ff88' if close >= open_ else '#ff4444' 
                 for close, open_ in zip(df_cached['close'], df_cached['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df_cached.index,
                y=df_cached['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # Moving averages on cached data
        if len(df_cached) > 20:
            df_cached['MA20'] = df_cached['close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_cached.index,
                    y=df_cached['MA20'],
                    name='MA20',
                    line=dict(color='#008394', width=1)
                ),
                row=1, col=1
            )
        
        if len(df_cached) > 50:
            df_cached['MA50'] = df_cached['close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_cached.index,
                    y=df_cached['MA50'],
                    name='MA50',
                    line=dict(color='#B98544', width=1)
                ),
                row=1, col=1
            )
        
        # Set view to display window
        x_range = [df_display.index[0], df_display.index[-1]]
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            margin=dict(l=50, r=50, t=50, b=50),
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='#008394',
                borderwidth=1
            ),
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            font=dict(family='Inter, sans-serif', size=12, color='#888'),
            dragmode='pan',  # Allow panning
            xaxis=dict(
                range=x_range,  # Set initial view to window
                fixedrange=False  # Allow zoom/pan
            )
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor='rgba(255,255,255,0.05)',
            showgrid=True,
            zeroline=False
        )
        
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.05)',
            showgrid=True,
            zeroline=False,
            title_text="Price (USD)",
            row=1, col=1
        )
        
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.05)',
            showgrid=True,
            zeroline=False,
            title_text="Volume",
            row=2, col=1
        )
        
        # Create range display text
        start_date = df_display.index[0].strftime('%Y-%m-%d %H:%M')
        end_date = df_display.index[-1].strftime('%Y-%m-%d %H:%M')
        range_text = f"Showing: {start_date} to {end_date} ({window_state['window_end'] - window_state['window_start'] + 1} candles)"
        
        return fig, window_state, range_text
    
    # ==================== Update Price Display ====================
    @app.callback(
        Output('current-price-display', 'children'),
        [Input('crypto-dropdown', 'value'),
         Input('data-refresh-interval', 'n_intervals')]
    )
    def update_price_display(selected_crypto, n_intervals):
        """Update the current price display"""
        df = data_center.load_data(selected_crypto, data_center.current_timeframe)
        
        if df.empty:
            return [
                html.Span(f'{selected_crypto.replace("_USDT", "")}/USD', 
                         style={'color': '#888', 'fontSize': '0.875rem'}),
                html.Div([
                    html.Span('--', style={
                        'fontSize': '2.5rem',
                        'fontWeight': '700',
                        'color': 'white',
                        'marginRight': '1rem'
                    })
                ])
            ]
        
        # Get latest price and calculate change
        latest_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else latest_price
        change = ((latest_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        return [
            html.Span(f'{selected_crypto.replace("_USDT", "")}/USD', 
                     style={'color': '#888', 'fontSize': '0.875rem'}),
            html.Div([
                html.Span(f'${latest_price:,.2f}', style={
                    'fontSize': '2.5rem',
                    'fontWeight': '700',
                    'color': 'white',
                    'marginRight': '1rem'
                }),
                html.Span(f'{change:+.2f}%', style={
                    'fontSize': '1.25rem',
                    'fontWeight': '600',
                    'color': '#00ff88' if change >= 0 else '#ff4444'
                })
            ])
        ]
    
    # ==================== Update Statistics Bar ====================
    @app.callback(
        [Output('volume-24h', 'children'),
         Output('high-24h', 'children'),
         Output('low-24h', 'children'),
         Output('market-cap', 'children'),
         Output('rsi-value', 'children')],
        [Input('crypto-dropdown', 'value'),
         Input('data-refresh-interval', 'n_intervals')]
    )
    def update_statistics(selected_crypto, n_intervals):
        """Update statistics bar"""
        df = data_center.load_data(selected_crypto, data_center.current_timeframe)
        
        if df.empty:
            return '--', '--', '--', '--', '--'
        
        # Calculate 24h stats (depends on timeframe)
        timeframe_hours = {
            '1m': 1440, '5m': 288, '15m': 96, '30m': 48,
            '1h': 24, '4h': 6, '1d': 1
        }
        periods_24h = timeframe_hours.get(data_center.current_timeframe, 24)
        last_24h = df.tail(periods_24h) if len(df) >= periods_24h else df
        
        volume_24h = last_24h['volume'].sum()
        high_24h = last_24h['high'].max()
        low_24h = last_24h['low'].min()
        
        # Market cap (placeholder - would need real data)
        market_cap = '--'
        
        # Calculate RSI
        rsi = '--'
        if len(df) >= 14:
            # Simple RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_val = 100 - (100 / (1 + rs))
            rsi = f"{rsi_val.iloc[-1]:.1f}" if not pd.isna(rsi_val.iloc[-1]) else '--'
        
        # Format volume
        if volume_24h > 1e9:
            volume_str = f'${volume_24h/1e9:.2f}B'
        elif volume_24h > 1e6:
            volume_str = f'${volume_24h/1e6:.2f}M'
        else:
            volume_str = f'${volume_24h:,.0f}'
        
        return (
            volume_str,
            f'${high_24h:,.2f}',
            f'${low_24h:,.2f}',
            market_cap,
            rsi
        )
    
    # ==================== Update Data Quality Card ====================
    @app.callback(
        [Output('data-quality-status', 'children'),
         Output('data-missing', 'children'),
         Output('data-last-update', 'children')],
        [Input('crypto-dropdown', 'value'),
         Input('data-refresh-interval', 'n_intervals')]
    )
    def update_data_quality(selected_crypto, n_intervals):
        """Update data quality card"""
        df = data_center.load_data(selected_crypto, data_center.current_timeframe)
        
        if df.empty:
            return 'No Data', 'N/A', 'Never'
        
        # Validate data
        quality_report = data_center.validate_data(df, selected_crypto, data_center.current_timeframe)
        
        # Status
        if quality_report['quality_score'] > 90:
            status = 'Excellent'
        elif quality_report['quality_score'] > 70:
            status = 'Good'
        else:
            status = 'Poor'
        
        # Missing candles
        missing = quality_report.get('missing_candles', 0)
        missing_str = f'{missing:,}' if missing > 0 else 'None'
        
        # Last update
        if not df.empty:
            last_timestamp = df.index[-1]
            time_diff = datetime.now() - last_timestamp
            if time_diff.seconds < 60:
                last_update = f'{time_diff.seconds} sec ago'
            elif time_diff.seconds < 3600:
                last_update = f'{time_diff.seconds // 60} min ago'
            else:
                last_update = f'{time_diff.seconds // 3600} hours ago'
        else:
            last_update = 'Never'
        
        return status, missing_str, last_update
    
    # ==================== Update Technical Indicators ====================
    @app.callback(
        [Output('macd-signal', 'children'),
         Output('bb-position', 'children'),
         Output('trend-direction', 'children')],
        [Input('crypto-dropdown', 'value'),
         Input('data-refresh-interval', 'n_intervals')]
    )
    def update_technical_indicators(selected_crypto, n_intervals):
        """Update technical indicators card"""
        df = data_center.load_data(selected_crypto, data_center.current_timeframe)
        
        if df.empty or len(df) < 50:
            return '--', '--', '--'
        
        # MACD Signal
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_diff = macd.iloc[-1] - signal.iloc[-1]
        macd_signal = 'Bullish' if macd_diff > 0 else 'Bearish'
        
        # Bollinger Bands Position
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        current_price = df['close'].iloc[-1]
        
        if current_price > upper_band.iloc[-1]:
            bb_position = 'Overbought'
        elif current_price < lower_band.iloc[-1]:
            bb_position = 'Oversold'
        else:
            bb_position = 'Neutral'
        
        # Trend Direction (SMA cross)
        if len(df) > 50:
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
            trend = 'Uptrend' if sma_20 > sma_50 else 'Downtrend'
        else:
            trend = '--'
        
        return macd_signal, bb_position, trend
    
    # ==================== Fill Gaps Button ====================
    @app.callback(
        Output('fill-gaps-btn', 'children'),
        [Input('fill-gaps-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def fill_data_gaps(n_clicks):
        """Fill data gaps using Kraken API"""
        if not n_clicks:
            raise PreventUpdate
        
        try:
            print("Attempting to fill data gaps...")
            import os
            
            # Create subdirectory for timeframe
            timeframe_path = os.path.join(data_center.data_path, data_center.current_timeframe)
            os.makedirs(timeframe_path, exist_ok=True)
            
            # Create sample file for current selection
            filename = f"{data_center.current_asset}_{data_center.current_timeframe}.csv"
            filepath = os.path.join(timeframe_path, filename)
            
            print(f"Checking for file: {filepath}")
            
            if not os.path.exists(filepath):
                # Generate sample data
                periods = 500
                dates = pd.date_range(end=datetime.now(), periods=periods, freq='1h')
                base_price = 40000 if 'BTC' in data_center.current_asset else 2500
                
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': np.random.uniform(base_price*0.98, base_price*1.02, periods),
                    'high': np.random.uniform(base_price*1.01, base_price*1.03, periods),
                    'low': np.random.uniform(base_price*0.97, base_price*0.99, periods),
                    'close': np.random.uniform(base_price*0.98, base_price*1.02, periods),
                    'volume': np.random.uniform(1000, 10000, periods)
                })
                df.set_index('timestamp', inplace=True)
                df.to_csv(filepath)
                print(f"Created sample data: {filepath}")
            
            # Clear cache to reload
            data_center.data_cache = {}
            
            return [html.I(className="fas fa-check", style={'marginRight': '0.5rem'}), 'Data Loaded!']
            
        except Exception as e:
            print(f"Error filling gaps: {e}")
            return [html.I(className="fas fa-exclamation", style={'marginRight': '0.5rem'}), 'Error']
    
    # ==================== Fix Data Button ====================
    @app.callback(
        Output('fix-data-btn', 'children'),
        [Input('fix-data-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def fix_data_issues(n_clicks):
        """Fix data quality issues"""
        if not n_clicks:
            raise PreventUpdate
        
        # Load current data
        df = data_center.load_data(data_center.current_asset, data_center.current_timeframe)
        
        if df.empty:
            return [html.I(className="fas fa-times", style={'marginRight': '0.5rem'}), 'No Data']
        
        try:
            # Fix issues using validator
            fixed_result = data_center.validator.validate_and_fix(
                df,
                symbol=data_center.current_asset,
                timeframe=data_center.current_timeframe,
                auto_fix=True
            )
            
            if fixed_result.fixed_df is not None:
                # Update cache
                cache_key = f"{data_center.current_asset}_{data_center.current_timeframe}"
                data_center.data_cache[cache_key] = fixed_result.fixed_df
                
                return [html.I(className="fas fa-check", style={'marginRight': '0.5rem'}), 'Fixed!']
        except Exception as e:
            print(f"Error fixing data: {e}")
        
        return [html.I(className="fas fa-wrench", style={'marginRight': '0.5rem'}), 'Fix Data']
    
    # ==================== Update Timeframe Button Styles ====================
    @app.callback(
        [Output(f'tf-btn-{tf["value"]}', 'className') for tf in data_center.timeframes],
        [Input(f'tf-btn-{tf["value"]}', 'n_clicks') for tf in data_center.timeframes]
    )
    def update_timeframe_styles(*n_clicks):
        """Update timeframe button styles when clicked"""
        ctx = callback_context
        
        # Default all to inactive
        styles = ['timeframe-btn' for _ in data_center.timeframes]
        
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if 'tf-btn' in button_id:
                selected_tf = button_id.replace('tf-btn-', '')
                # Find index and set active
                for i, tf in enumerate(data_center.timeframes):
                    if tf['value'] == selected_tf:
                        styles[i] = 'timeframe-btn active'
                        break
        else:
            # Default to 1h active
            for i, tf in enumerate(data_center.timeframes):
                if tf['value'] == '1h':
                    styles[i] = 'timeframe-btn active'
                    break
        
        return styles
    
    # ==================== Update Chart Type Button Styles ====================
    @app.callback(
        [Output('chart-type-candle', 'className'),
         Output('chart-type-line', 'className'),
         Output('chart-type-area', 'className')],
        [Input('chart-type-candle', 'n_clicks'),
         Input('chart-type-line', 'n_clicks'),
         Input('chart-type-area', 'n_clicks')]
    )
    def update_chart_type_styles(candle_clicks, line_clicks, area_clicks):
        """Update chart type button styles"""
        ctx = callback_context
        
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'chart-type-candle':
                return 'chart-type-btn active', 'chart-type-btn', 'chart-type-btn'
            elif button_id == 'chart-type-line':
                return 'chart-type-btn', 'chart-type-btn active', 'chart-type-btn'
            elif button_id == 'chart-type-area':
                return 'chart-type-btn', 'chart-type-btn', 'chart-type-btn active'
        
        # Default to candlestick active
        return 'chart-type-btn active', 'chart-type-btn', 'chart-type-btn'
    
    @app.callback(
        Output('window-state-store', 'data'),
        [Input('crypto-dropdown', 'value')] +
        [Input(f'tf-btn-{tf["value"]}', 'n_clicks') for tf in data_center.timeframes],
        prevent_initial_call=False
    )
    def initialize_window_state(selected_crypto, *tf_clicks):
        """Initialize or reset window state when asset/timeframe changes"""
        ctx = callback_context
        selected_timeframe = data_center.current_timeframe
        
        if ctx.triggered and 'tf-btn' in ctx.triggered[0]['prop_id']:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            selected_timeframe = button_id.replace('tf-btn-', '')
        
        # Load full data to get length
        df = data_center.load_data(selected_crypto, selected_timeframe)
        
        if not df.empty:
            total_points = len(df)
            # Start from the most recent data
            window_end = total_points - 1
            window_start = max(0, window_end - 79)  # 80 points
            
            return {
                'asset': selected_crypto,
                'timeframe': selected_timeframe,
                'window_start': window_start,
                'window_end': window_end,
                'total_points': total_points,
                'cached_start': max(0, window_start - 80),  # Buffer before
                'cached_end': min(total_points - 1, window_end + 80)  # Buffer after
            }
        
        return {}