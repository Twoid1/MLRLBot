"""
dashboard/components/shared_components.py
Reusable UI components for the trading dashboard
"""

from dash import html, dcc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import numpy as np

class UIComponents:
    """Shared UI components for consistent design across the dashboard"""
    
    @staticmethod
    def create_status_badge(status, type='info'):
        """Create a status badge with consistent styling"""
        color_map = {
            'success': {'bg': 'rgba(74, 222, 128, 0.2)', 'border': '#4ade80', 'text': '#4ade80'},
            'warning': {'bg': 'rgba(251, 191, 36, 0.2)', 'border': '#fbbf24', 'text': '#fbbf24'},
            'danger': {'bg': 'rgba(248, 113, 113, 0.2)', 'border': '#f87171', 'text': '#f87171'},
            'info': {'bg': 'rgba(59, 130, 246, 0.2)', 'border': '#3b82f6', 'text': '#3b82f6'},
            'primary': {'bg': 'rgba(0, 131, 148, 0.2)', 'border': '#008394', 'text': '#008394'}
        }
        
        colors = color_map.get(type, color_map['info'])
        
        return html.Span(
            status,
            style={
                'background': colors['bg'],
                'border': f"1px solid {colors['border']}",
                'color': colors['text'],
                'padding': '0.25rem 0.75rem',
                'borderRadius': '12px',
                'fontSize': '0.75rem',
                'fontWeight': '600',
                'textTransform': 'uppercase',
                'letterSpacing': '0.5px'
            }
        )
    
    @staticmethod
    def create_metric_card(title, value, change=None, icon=None):
        """Create a metric display card"""
        return html.Div([
            html.Div([
                html.Span(title, style={'fontSize': '0.875rem', 'color': '#b0b0b0'}),
                icon if icon else None
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '0.5rem'}),
            
            html.Div(value, style={
                'fontSize': '1.5rem',
                'fontWeight': '700',
                'color': '#ffffff'
            }),
            
            change if change else None
        ], style={
            'background': 'rgba(255, 255, 255, 0.05)',
            'border': '1px solid #4a4a4a',
            'borderRadius': '8px',
            'padding': '1rem'
        })
    
    @staticmethod
    def create_progress_bar(value, max_value=100, color='#008394'):
        """Create a progress bar"""
        percentage = (value / max_value) * 100
        
        return html.Div([
            html.Div(style={
                'width': f"{percentage}%",
                'height': '100%',
                'background': f"linear-gradient(90deg, {color}, {color}dd)",
                'borderRadius': '4px',
                'transition': 'width 0.3s ease',
                'boxShadow': f"0 0 10px {color}66"
            })
        ], style={
            'width': '100%',
            'height': '8px',
            'background': 'rgba(255, 255, 255, 0.1)',
            'borderRadius': '4px',
            'overflow': 'hidden'
        })
    
    @staticmethod
    def create_icon_button(icon_class, text, button_id, variant='primary'):
        """Create a button with icon"""
        style_map = {
            'primary': {
                'background': 'linear-gradient(45deg, #008394, #00a3b8)',
                'color': '#ffffff',
                'border': '1px solid #008394'
            },
            'secondary': {
                'background': 'rgba(107, 114, 128, 0.2)',
                'color': '#ffffff',
                'border': '1px solid #6b7280'
            },
            'danger': {
                'background': 'rgba(239, 68, 68, 0.2)',
                'color': '#ef4444',
                'border': '1px solid #ef4444'
            },
            'success': {
                'background': 'rgba(16, 185, 129, 0.2)',
                'color': '#10b981',
                'border': '1px solid #10b981'
            }
        }
        
        button_style = style_map.get(variant, style_map['primary'])
        button_style.update({
            'padding': '0.75rem 1.5rem',
            'borderRadius': '8px',
            'fontWeight': '600',
            'fontSize': '0.875rem',
            'cursor': 'pointer',
            'transition': 'all 0.3s ease',
            'display': 'inline-flex',
            'alignItems': 'center',
            'gap': '0.5rem'
        })
        
        return html.Button([
            html.I(className=icon_class),
            text
        ], id=button_id, style=button_style)


class ChartComponents:
    """Chart components for data visualization"""
    
    @staticmethod
    def create_candlestick_chart(df, height=400):
        """Create a candlestick chart"""
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='#4ade80',
            decreasing_line_color='#f87171'
        )])
        
        fig.update_layout(
            template='plotly_dark',
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_rangeslider_visible=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            )
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})
    
    @staticmethod
    def create_line_chart(data, x, y, title='', height=300):
        """Create a line chart"""
        fig = go.Figure(data=[go.Scatter(
            x=data[x],
            y=data[y],
            mode='lines',
            line=dict(color='#008394', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 131, 148, 0.1)'
        )])
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            )
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})
    
    @staticmethod
    def create_gauge_chart(value, max_value=100, title=''):
        """Create a gauge chart for metrics"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': "#008394"},
                'steps': [
                    {'range': [0, max_value * 0.5], 'color': "rgba(255,255,255,0.05)"},
                    {'range': [max_value * 0.5, max_value * 0.8], 'color': "rgba(255,255,255,0.1)"},
                    {'range': [max_value * 0.8, max_value], 'color': "rgba(255,255,255,0.15)"}
                ],
                'threshold': {
                    'line': {'color': "#f87171", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=200,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})
    
    @staticmethod
    def create_heatmap(correlation_matrix, height=400):
        """Create a correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})


class TableComponents:
    """Table components for data display"""
    
    @staticmethod
    def create_data_table(data, columns, table_id):
        """Create a styled data table"""
        from dash import dash_table
        
        return dash_table.DataTable(
            id=table_id,
            columns=[{"name": col, "id": col} for col in columns],
            data=data,
            style_cell={
                'backgroundColor': 'rgba(0,0,0,0)',
                'color': 'white',
                'border': '1px solid #4a4a4a',
                'textAlign': 'left',
                'fontSize': '0.875rem',
                'padding': '10px'
            },
            style_data={
                'backgroundColor': 'rgba(255,255,255,0.05)',
                'color': 'white'
            },
            style_header={
                'backgroundColor': 'rgba(0, 131, 148, 0.2)',
                'color': 'white',
                'fontWeight': 'bold',
                'border': '1px solid #008394'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgba(255,255,255,0.02)',
                }
            ],
            page_size=10
        )
    
    @staticmethod
    def create_position_row(symbol, type, size, pnl, entry_price=None):
        """Create a position row display"""
        return html.Div([
            # Left side - Symbol and type
            html.Div([
                html.Span(symbol, style={
                    'fontWeight': '700',
                    'fontSize': '1rem',
                    'color': '#ffffff'
                }),
                UIComponents.create_status_badge(
                    type,
                    'success' if type == 'LONG' else 'danger'
                )
            ], style={'display': 'flex', 'gap': '0.75rem', 'alignItems': 'center'}),
            
            # Right side - Size and P&L
            html.Div([
                html.Div([
                    html.Span('Size:', style={'color': '#888', 'fontSize': '0.75rem'}),
                    html.Span(size, style={
                        'fontFamily': 'JetBrains Mono, monospace',
                        'color': '#b0b0b0',
                        'fontSize': '0.875rem',
                        'marginLeft': '0.25rem'
                    })
                ]),
                html.Div([
                    html.Span('P&L:', style={'color': '#888', 'fontSize': '0.75rem'}),
                    html.Span(
                        f"${pnl:+.2f}",
                        style={
                            'color': '#4ade80' if pnl >= 0 else '#f87171',
                            'fontWeight': '600',
                            'fontSize': '0.875rem',
                            'marginLeft': '0.25rem'
                        }
                    )
                ])
            ], style={'display': 'flex', 'gap': '1rem', 'alignItems': 'center'})
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'padding': '1rem',
            'background': 'rgba(255, 255, 255, 0.05)',
            'border': '1px solid #4a4a4a',
            'borderRadius': '8px',
            'marginBottom': '0.75rem',
            'transition': 'all 0.3s ease'
        })


class AlertComponents:
    """Alert and notification components"""
    
    @staticmethod
    def create_alert(message, type='info', dismissible=True):
        """Create an alert notification"""
        color_map = {
            'success': {'bg': 'rgba(74, 222, 128, 0.1)', 'border': '#4ade80', 'icon': 'fa-check-circle'},
            'warning': {'bg': 'rgba(251, 191, 36, 0.1)', 'border': '#fbbf24', 'icon': 'fa-exclamation-triangle'},
            'danger': {'bg': 'rgba(248, 113, 113, 0.1)', 'border': '#f87171', 'icon': 'fa-times-circle'},
            'info': {'bg': 'rgba(59, 130, 246, 0.1)', 'border': '#3b82f6', 'icon': 'fa-info-circle'}
        }
        
        alert_style = color_map.get(type, color_map['info'])
        
        return html.Div([
            html.I(className=f"fas {alert_style['icon']}", style={
                'color': alert_style['border'],
                'marginRight': '0.75rem',
                'fontSize': '1.25rem'
            }),
            html.Span(message, style={'flex': '1'}),
            html.Button('×', style={
                'background': 'none',
                'border': 'none',
                'color': alert_style['border'],
                'fontSize': '1.5rem',
                'cursor': 'pointer',
                'padding': '0',
                'marginLeft': '1rem'
            }) if dismissible else None
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '1rem',
            'background': alert_style['bg'],
            'border': f"1px solid {alert_style['border']}",
            'borderRadius': '8px',
            'marginBottom': '1rem'
        })
    
    @staticmethod
    def create_toast(message, type='info'):
        """Create a toast notification"""
        return html.Div([
            AlertComponents.create_alert(message, type, dismissible=False)
        ], style={
            'position': 'fixed',
            'top': '20px',
            'right': '20px',
            'zIndex': '9999',
            'minWidth': '300px',
            'animation': 'slideInRight 0.3s ease'
        })


class LoadingComponents:
    """Loading and skeleton components"""
    
    @staticmethod
    def create_spinner(size='default'):
        """Create a loading spinner"""
        size_map = {
            'small': '20px',
            'default': '40px',
            'large': '60px'
        }
        
        spinner_size = size_map.get(size, size_map['default'])
        
        return html.Div([
            html.Div(style={
                'width': spinner_size,
                'height': spinner_size,
                'border': '3px solid rgba(0, 131, 148, 0.2)',
                'borderTop': '3px solid #008394',
                'borderRadius': '50%',
                'animation': 'spin 1s linear infinite'
            })
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'padding': '2rem'
        })
    
    @staticmethod
    def create_skeleton_card():
        """Create a skeleton loading card"""
        return html.Div([
            html.Div(style={
                'height': '20px',
                'background': 'linear-gradient(90deg, #3a3a3a 25%, #4a4a4a 50%, #3a3a3a 75%)',
                'backgroundSize': '200% 100%',
                'animation': 'loading 1.5s infinite',
                'borderRadius': '4px',
                'marginBottom': '1rem'
            }),
            html.Div(style={
                'height': '60px',
                'background': 'linear-gradient(90deg, #3a3a3a 25%, #4a4a4a 50%, #3a3a3a 75%)',
                'backgroundSize': '200% 100%',
                'animation': 'loading 1.5s infinite',
                'borderRadius': '4px'
            })
        ], className='trading-card')