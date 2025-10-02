"""
dashboard/callbacks/training_callbacks.py
Callback functions for the AI Training Laboratory
"""

from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import json
import time
from datetime import datetime
import numpy as np

def register_training_callbacks(app, training_lab):
    """Register all training-related callbacks"""
    
    # Mode switching callback
    @app.callback(
        [Output('training-content', 'children'),
         Output('ml-mode-btn', 'className'),
         Output('rl-mode-btn', 'className')],
        [Input('ml-mode-btn', 'n_clicks'),
         Input('rl-mode-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def switch_training_mode(ml_clicks, rl_clicks):
        """Switch between ML and RL training modes"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'ml-mode-btn':
            return (
                training_lab._create_ml_section(),
                'training-mode-btn active',
                'training-mode-btn'
            )
        else:
            return (
                training_lab._create_rl_section(),
                'training-mode-btn',
                'training-mode-btn active'
            )
    
    # ML Training Control
    @app.callback(
        [Output('ml-start-btn', 'disabled'),
         Output('ml-pause-btn', 'disabled'),
         Output('ml-stop-btn', 'disabled'),
         Output('ml-progress-bar', 'style'),
         Output('ml-progress-text', 'children'),
         Output('training-log', 'children'),
         Output('epoch-metric', 'children'),
         Output('time-metric', 'children'),
         Output('eta-metric', 'children')],
        [Input('ml-start-btn', 'n_clicks'),
         Input('ml-pause-btn', 'n_clicks'),
         Input('ml-stop-btn', 'n_clicks'),
         Input('ml-reset-btn', 'n_clicks')],
        [State('ml-asset-checklist', 'value'),
         State('ml-timeframe-checklist', 'value'),
         State('ml-model-type', 'value'),
         State('training-log', 'children')],
        prevent_initial_call=True
    )
    def control_ml_training(start_clicks, pause_clicks, stop_clicks, reset_clicks, 
                           assets, timeframes, model_type, current_log):
        """Control ML model training"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Get current timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Initialize log if it's not a list
        if not isinstance(current_log, list):
            current_log = []
        
        if button_id == 'ml-start-btn':
            # Start training
            new_log_entry = training_lab._create_log_entry(
                'ML Training',
                f'Starting {model_type} training on {", ".join(assets)}',
                'success'
            )
            current_log.append(new_log_entry)
            
            # Simulate progress (in production, this would be real training progress)
            progress = 25  # Example progress
            
            return (
                True,   # Disable start
                False,  # Enable pause
                False,  # Enable stop
                {
                    'width': f'{progress}%',
                    'height': '100%',
                    'background': 'linear-gradient(90deg, #008394 0%, #00a3b8 100%)',
                    'borderRadius': '4px',
                    'transition': 'width 0.3s ease'
                },
                f'{progress}%',
                current_log,
                '25/100',
                '00:15:32',
                '00:45:00'
            )
        
        elif button_id == 'ml-pause-btn':
            # Pause training
            new_log_entry = training_lab._create_log_entry(
                'ML Training',
                'Training paused',
                'warning'
            )
            current_log.append(new_log_entry)
            
            return (
                False,  # Enable start
                True,   # Disable pause
                False,  # Enable stop
                no_update,  # Keep progress
                no_update,  # Keep progress text
                current_log,
                no_update,  # Keep epoch
                no_update,  # Keep time
                '--:--:--'  # Clear ETA
            )
        
        elif button_id == 'ml-stop-btn':
            # Stop training
            new_log_entry = training_lab._create_log_entry(
                'ML Training',
                'Training stopped',
                'error'
            )
            current_log.append(new_log_entry)
            
            return (
                False,  # Enable start
                True,   # Disable pause
                True,   # Disable stop
                {
                    'width': '0%',
                    'height': '100%',
                    'background': 'linear-gradient(90deg, #008394 0%, #00a3b8 100%)',
                    'borderRadius': '4px',
                    'transition': 'width 0.3s ease'
                },
                '0%',
                current_log,
                '0/100',
                '00:00:00',
                '--:--:--'
            )
        
        elif button_id == 'ml-reset-btn':
            # Reset training
            initial_log = [
                training_lab._create_log_entry('System', 'Training environment reset', 'info'),
                training_lab._create_log_entry('Ready', 'Waiting for training to start...', 'warning')
            ]
            
            return (
                False,  # Enable start
                True,   # Disable pause
                True,   # Disable stop
                {
                    'width': '0%',
                    'height': '100%',
                    'background': 'linear-gradient(90deg, #008394 0%, #00a3b8 100%)',
                    'borderRadius': '4px',
                    'transition': 'width 0.3s ease'
                },
                '0%',
                initial_log,
                '0/100',
                '00:00:00',
                '--:--:--'
            )
        
        raise PreventUpdate
    
    # RL Training Control (similar structure)
    @app.callback(
        [Output('rl-start-btn', 'disabled'),
         Output('rl-pause-btn', 'disabled'),
         Output('rl-stop-btn', 'disabled'),
         Output('rl-progress-bar', 'style'),
         Output('rl-progress-text', 'children')],
        [Input('rl-start-btn', 'n_clicks'),
         Input('rl-pause-btn', 'n_clicks'),
         Input('rl-stop-btn', 'n_clicks'),
         Input('rl-reset-btn', 'n_clicks')],
        [State('rl-agent-type', 'value'),
         State('training-episodes', 'value')],
        prevent_initial_call=True
    )
    def control_rl_training(start_clicks, pause_clicks, stop_clicks, reset_clicks,
                           agent_type, episodes):
        """Control RL agent training"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'rl-start-btn':
            # Start training
            progress = 15  # Example progress
            
            return (
                True,   # Disable start
                False,  # Enable pause
                False,  # Enable stop
                {
                    'width': f'{progress}%',
                    'height': '100%',
                    'background': 'linear-gradient(90deg, #008394 0%, #00a3b8 100%)',
                    'borderRadius': '4px',
                    'transition': 'width 0.3s ease'
                },
                f'{progress}%'
            )
        
        elif button_id == 'rl-pause-btn':
            return (
                False,  # Enable start
                True,   # Disable pause
                False,  # Enable stop
                no_update,
                no_update
            )
        
        elif button_id == 'rl-stop-btn' or button_id == 'rl-reset-btn':
            return (
                False,  # Enable start
                True,   # Disable pause
                True,   # Disable stop
                {
                    'width': '0%',
                    'height': '100%',
                    'background': 'linear-gradient(90deg, #008394 0%, #00a3b8 100%)',
                    'borderRadius': '4px',
                    'transition': 'width 0.3s ease'
                },
                '0%'
            )
        
        raise PreventUpdate
    
    # Feature selection update
    @app.callback(
        Output('n-features-slider', 'value'),
        [Input('feature-selection-method', 'value')],
        prevent_initial_call=True
    )
    def update_feature_count(method):
        """Update feature count based on selection method"""
        if method == 'manual':
            return 100  # Allow all features for manual selection
        elif method == 'kbest':
            return 50
        elif method == 'rfe':
            return 30
        else:  # importance
            return 40
    
    # Update live metrics during training (simulated)
    @app.callback(
        [Output('gpu-metric', 'children'),
         Output('lr-metric', 'children'),
         Output('best-metric', 'children')],
        [Input('ml-start-btn', 'n_clicks'),
         Input('rl-start-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def update_live_metrics(ml_clicks, rl_clicks):
        """Update live training metrics"""
        # Simulate GPU usage
        gpu_used = np.random.uniform(2, 6)
        gpu_total = 8
        
        # Simulate learning rate
        lr = 0.001
        
        # Simulate best score
        best_score = np.random.uniform(0.75, 0.85)
        
        return (
            f'{gpu_used:.1f}/{gpu_total} GB',
            f'{lr:.4f}',
            f'{best_score:.3f}'
        )
    
    # Update charts based on training progress
    @app.callback(
        [Output('ml-accuracy-chart', 'figure'),
         Output('ml-loss-chart', 'figure'),
         Output('ml-f1-chart', 'figure')],
        [Input('ml-progress-bar', 'style')],
        [State('ml-accuracy-chart', 'figure'),
         State('ml-loss-chart', 'figure'),
         State('ml-f1-chart', 'figure')],
        prevent_initial_call=True
    )
    def update_ml_charts(progress_style, acc_fig, loss_fig, f1_fig):
        """Update ML training charts based on progress"""
        if not progress_style:
            raise PreventUpdate
        
        # Extract progress percentage
        progress_str = progress_style.get('width', '0%')
        progress = int(progress_str.replace('%', ''))
        
        if progress == 0:
            # Reset charts
            return (
                training_lab._create_metric_chart('Accuracy', 'Training Accuracy'),
                training_lab._create_metric_chart('Loss', 'Training Loss', ascending=False),
                training_lab._create_metric_chart('F1 Score', 'F1 Score')
            )
        
        # Add new data points to existing charts (simplified for demo)
        # In production, you would append real training metrics
        
        return acc_fig, loss_fig, f1_fig
    
    # Model save functionality
    @app.callback(
        Output('saved-models-table', 'data'),
        [Input('save-model-btn', 'n_clicks')],
        [State('saved-models-table', 'data'),
         State('ml-model-type', 'value'),
         State('ml-progress-text', 'children')],
        prevent_initial_call=True
    )
    def save_model(n_clicks, current_data, model_type, progress):
        """Save current model to the table"""
        if not n_clicks:
            raise PreventUpdate
        
        # Create new model entry
        new_model = {
            'name': f'{model_type}_v{len(current_data) + 1}',
            'type': model_type,
            'accuracy': f'{np.random.uniform(75, 85):.1f}%',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'status': 'Testing'
        }
        
        # Add to beginning of list
        return [new_model] + current_data

# Additional helper functions for backend integration
def prepare_training_data(assets, timeframes, date_range):
    """
    Prepare training data based on selected parameters
    This will connect to data_manager.py in production
    """
    # Placeholder for data preparation
    # In production: 
    # - Call data_manager.load_existing_data()
    # - Filter by assets and timeframes
    # - Apply date range
    return {
        'samples': 147320,
        'features': 100,
        'assets': assets,
        'timeframes': timeframes
    }

def start_ml_training(config):
    """
    Start ML model training
    This will connect to ml_predictor.py in production
    """
    # Placeholder for ML training
    # In production:
    # - Initialize MLPredictor with config
    # - Start training in background thread
    # - Return training job ID
    return {
        'job_id': f'ml_training_{int(time.time())}',
        'status': 'started',
        'estimated_time': '45 minutes'
    }

def start_rl_training(config):
    """
    Start RL agent training
    This will connect to dqn_agent.py in production
    """
    # Placeholder for RL training
    # In production:
    # - Initialize DQNAgent with config
    # - Create TradingEnvironment
    # - Start training in background thread
    # - Return training job ID
    return {
        'job_id': f'rl_training_{int(time.time())}',
        'status': 'started',
        'estimated_episodes': config['episodes']
    }

def get_training_status(job_id):
    """
    Get current training status
    This will check actual training progress in production
    """
    # Placeholder for status check
    # In production:
    # - Query training job status
    # - Get current metrics
    # - Return progress info
    return {
        'progress': np.random.uniform(0, 100),
        'current_epoch': np.random.randint(0, 100),
        'current_loss': np.random.uniform(0.1, 0.5),
        'current_accuracy': np.random.uniform(0.6, 0.9),
        'eta': '00:30:00'
    }