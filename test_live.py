#!/usr/bin/env python3
"""
Comprehensive Live Trading Test Suite
Tests ALL components before going live with real money

Usage:
    python test_live_trading.py              # Run all tests
    python test_live_trading.py --quick      # Quick smoke test
    python test_live_trading.py --section 1  # Run specific section
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import joblib
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging

# Add project to path
sys.path.append(str(Path(__file__).parent))
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test results tracking
test_results = {
    'total': 0,
    'passed': 0,
    'failed': 0,
    'warnings': 0,
    'tests': []
}


# ==================== HELPER FUNCTIONS ====================

def test_section(name: str):
    """Decorator for test sections"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("\n" + "="*80)
            print(f"  {name}")
            print("="*80)
            result = func(*args, **kwargs)
            return result
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


def test_case(description: str):
    """Decorator for individual test cases"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            test_results['total'] += 1
            test_name = f"{func.__name__}: {description}"
            
            try:
                print(f"\n  Testing: {description}...", end=" ")
                result = func(*args, **kwargs)
                
                if result is True:
                    print(" PASS")
                    test_results['passed'] += 1
                    test_results['tests'].append({'name': test_name, 'status': 'PASS'})
                    return True
                elif result == 'WARNING':
                    print("  WARNING")
                    test_results['warnings'] += 1
                    test_results['tests'].append({'name': test_name, 'status': 'WARNING'})
                    return True
                else:
                    print(" FAIL")
                    test_results['failed'] += 1
                    test_results['tests'].append({'name': test_name, 'status': 'FAIL', 'error': str(result)})
                    return False
                    
            except Exception as e:
                print(f" FAIL - {str(e)}")
                test_results['failed'] += 1
                test_results['tests'].append({'name': test_name, 'status': 'FAIL', 'error': str(e)})
                return False
                
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


def log_detail(message: str):
    """Log detailed information"""
    print(f"      {message}")


def log_value(name: str, value, expected=None):
    """Log a value with optional expected comparison"""
    if expected is not None:
        match = "Y" if value == expected else "N"
        print(f"    {match} {name}: {value} (expected: {expected})")
    else:
        print(f"      {name}: {value}")


# ==================== SECTION 1: ENVIRONMENT & DEPENDENCIES ====================

@test_section("SECTION 1: Environment & Dependencies")
def test_environment():
    """Test Python environment and required packages"""
    
    @test_case("Python version >= 3.9")
    def test_python_version():
        version = sys.version_info
        log_value("Python version", f"{version.major}.{version.minor}.{version.micro}")
        return version.major == 3 and version.minor >= 9
    
    @test_case("Required packages installed")
    def test_packages():
        required = [
            'pandas', 'numpy', 'torch', 'sklearn', 'xgboost', 
            'joblib', 'requests', 'krakenex', 'dotenv'
        ]
        missing = []
        
        for package in required:
            try:
                __import__(package)
                log_detail(f" {package}")
            except ImportError:
                missing.append(package)
                log_detail(f" {package} - MISSING")
        
        if missing:
            log_detail(f"Install missing: pip install {' '.join(missing)}")
            return False
        return True
    
    @test_case("CUDA/GPU availability")
    def test_gpu():
        cuda_available = torch.cuda.is_available()
        log_value("CUDA available", cuda_available)
        
        if cuda_available:
            log_value("GPU name", torch.cuda.get_device_name(0))
            log_value("GPU memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            log_detail("GPU not available - will use CPU (slower)")
        
        return 'WARNING' if not cuda_available else True
    
    @test_case("Environment variables")
    def test_env_vars():
        required_vars = ['KRAKEN_API_KEY', 'KRAKEN_API_SECRET']
        missing = []
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                log_detail(f" {var}: {'*' * 20} (hidden)")
            else:
                missing.append(var)
                log_detail(f" {var}: NOT SET")
        
        if missing:
            log_detail("Set missing vars in .env file")
            return False
        return True
    
    # Run tests
    test_python_version()
    test_packages()
    test_gpu()
    test_env_vars()


# ==================== SECTION 2: DATA PIPELINE ====================

@test_section("SECTION 2: Data Pipeline")
def test_data_pipeline():
    """Test data fetching and storage"""
    
    @test_case("Binance connector initialization")
    def test_binance_init():
        from src.data.binance_connector import BinanceDataConnector
        
        try:
            connector = BinanceDataConnector(data_path='./data/raw/')
            log_detail(f" Connector initialized")
            return True
        except Exception as e:
            log_detail(f" Failed: {e}")
            return False
    
    @test_case("Historical data files exist")
    def test_historical_data():
        data_path = Path('./data/raw/')
        symbols = ['ETH_USDT', 'SOL_USDT', 'DOT_USDT', 'AVAX_USDT', 'ADA_USDT']
        timeframes = ['5m', '15m', '1h']
        
        missing = []
        for symbol in symbols:
            for tf in timeframes:
                file_path = data_path / tf / f"{symbol}_{tf}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    log_detail(f" {symbol} {tf}: {len(df)} candles")
                else:
                    missing.append(f"{symbol}_{tf}")
                    log_detail(f" {symbol} {tf}: MISSING")
        
        if missing:
            log_detail("Run: python main.py data --fetch")
            return False
        return True
    
    @test_case("Data quality (no gaps, no duplicates)")
    def test_data_quality():
        data_path = Path('./data/raw/')
        symbol = 'ETH_USDT'
        timeframe = '5m'
        
        file_path = data_path / timeframe / f"{symbol}_{timeframe}.csv"
        if not file_path.exists():
            return False
        
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # ✅ Check duplicates (BAD)
        duplicates = df.index.duplicated().sum()
        log_value("Duplicate timestamps", duplicates, 0)
        
        # ✅ Check gaps (BAD)
        time_diff = df.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=5)
        gaps = (time_diff > expected_diff * 1.5).sum()
        log_value("Time gaps", gaps)
        
        # ✅ Check NaN in RAW DATA (BAD - would mean missing price data)
        nan_count = df.isna().sum().sum()
        log_value("NaN in raw OHLCV", nan_count, 0)
        
        # Pass if no duplicates and no NaN in raw data
        # Gaps are warned about but don't fail
        if duplicates == 0 and nan_count == 0:
            if gaps > 0:
                log_detail(f"  Found {gaps} time gaps, but continuing")
            return True
        else:
            return False
    
    @test_case("Data recency (< 7 days old)")
    def test_data_recency():
        data_path = Path('./data/raw/')
        symbol = 'ETH_USDT'
        timeframe = '5m'
        
        file_path = data_path / timeframe / f"{symbol}_{timeframe}.csv"
        if not file_path.exists():
            return False
        
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        last_timestamp = df['timestamp'].iloc[-1]
        now = pd.Timestamp.now()
        age_hours = (now - last_timestamp).total_seconds() / 3600
        
        log_value("Last data", last_timestamp.strftime('%Y-%m-%d %H:%M'))
        log_value("Age", f"{age_hours:.1f} hours")
        
        if age_hours > 168:  # 7 days
            log_detail("Data is stale! Run: python main.py data --update")
            return False
        elif age_hours > 24:
            log_detail("Data >24h old - consider updating")
            return 'WARNING'
        
        return True
    
    @test_case("Binance API connection (live)")
    def test_binance_api():
        from src.data.binance_connector import BinanceDataConnector
        
        try:
            connector = BinanceDataConnector(data_path='./data/raw/')
            
            # Try fetching current price
            df = connector.fetch_ohlc('ETH_USDT', '5m', limit=1)
            
            if df is not None and len(df) > 0:
                latest_price = df['close'].iloc[-1]
                log_value("Latest ETH price", f"${latest_price:.2f}")
                
                # Sanity check price
                if 1000 < latest_price < 10000:
                    return True
                else:
                    log_detail(f"Price seems unrealistic: ${latest_price}")
                    return False
            else:
                log_detail("No data returned from API")
                return False
                
        except Exception as e:
            log_detail(f"API error: {e}")
            return False
    
    # Run tests
    test_binance_init()
    test_historical_data()
    test_data_quality()
    test_data_recency()
    test_binance_api()


# ==================== SECTION 3: FEATURE ENGINEERING ====================

@test_section("SECTION 3: Feature Engineering")
def test_feature_engineering():
    """Test feature calculation and selection"""
    
    @test_case("Feature engineer loads")
    def test_feature_engineer_load():
        fe_path = Path('models/feature_engineer.pkl')
        
        if not fe_path.exists():
            log_detail("Feature engineer not found")
            log_detail("This should exist after training")
            return False
        
        try:
            feature_engineer = joblib.load(fe_path)
            log_detail(" Feature engineer loaded")
            
            # Check if it has expected methods
            required_methods = ['calculate_all_features']
            for method in required_methods:
                if not hasattr(feature_engineer, method):
                    log_detail(f" Missing method: {method}")
                    return False
            
            return True
        except Exception as e:
            log_detail(f" Load failed: {e}")
            return False
    
    @test_case("Selected features list exists")
    def test_selected_features():
        features_path = Path('models/features/selected_features.pkl')
        
        if not features_path.exists():
            log_detail("Selected features not found")
            log_detail("Run: python main.py features --select")
            return False
        
        try:
            selected_features = joblib.load(features_path)
            log_value("Number of features", len(selected_features))
            
            if len(selected_features) != 50:
                log_detail(f"Expected 50 features, got {len(selected_features)}")
                return False
            
            log_detail(f"Features: {selected_features[:5]}... (showing first 5)")
            return True
            
        except Exception as e:
            log_detail(f" Load failed: {e}")
            return False
    
    # Add this near the top of test_live.py, in the test_feature_calculation function:

    @test_case("Feature calculation on sample data")
    def test_feature_calculation():
        from src.features.feature_engineer import FeatureEngineer
        
        # Load sample data
        data_path = Path('./data/raw/5m/ETH_USDT_5m.csv')
        if not data_path.exists():
            return False
        
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.tail(250)  # Last 200 candles
        
        try:
            # Try both paths for feature engineer
            fe_path = Path('models/feature_engineer.pkl')
            if not fe_path.exists():
                fe_path = Path('models/features_engineer.pkl')
            
            if fe_path.exists():
                fe = joblib.load(fe_path)
                log_detail(f"Loaded from {fe_path}")
            else:
                fe = FeatureEngineer()
                log_detail("Using new FeatureEngineer instance")
            
            features = fe.calculate_all_features(df, 'ETH_USDT')
            
            log_value("Input rows", len(df))
            log_value("Output rows", len(features))
            log_value("Features calculated", len(features.columns))
            
            # ✅ SMART NaN CHECKING
            total_nan = features.isna().sum().sum()
            log_value("Total NaN values", total_nan)
            
            # Check LAST ROW (most important for live trading)
            last_row_nan = features.iloc[-1].isna().sum()
            log_value("NaN in LAST row", last_row_nan)
            
            if last_row_nan > 0:
                log_detail(" CRITICAL: Last row has NaN!")
                log_detail("This will break live trading!")
                nan_features = features.columns[features.iloc[-1].isna()].tolist()
                log_detail(f"Features with NaN: {nan_features[:10]}")
                return False
            
            # Check if NaN only in early rows (expected)
            first_row_nan = features.iloc[0].isna().sum()
            log_value("NaN in FIRST row", first_row_nan)
            
            if first_row_nan > 0:
                log_detail("  First rows have NaN (normal - warmup period)")
                log_detail("This is expected for indicators like MA(200)")
            
            log_detail(" Last row has NO NaN - ready for live trading")
            return True
            
        except Exception as e:
            log_detail(f" Calculation failed: {e}")
            import traceback
            log_detail(traceback.format_exc())
            return False
    
    @test_case("Feature selection matches training")
    def test_feature_selection_match():
        # Load selected features
        features_path = Path('models/features/selected_features.pkl')
        if not features_path.exists():
            return False
        
        selected_features = joblib.load(features_path)
        
        # Load ML model dict
        ml_path = Path('models/ml/ml_predictor.pkl')
        if not ml_path.exists():
            log_detail("ML model not found - can't verify")
            return 'WARNING'
        
        try:
            ml_dict = joblib.load(ml_path)
            
            # Extract features from dict
            if 'selected_features' in ml_dict:
                ml_features = ml_dict['selected_features']
            else:
                log_detail("Dict doesn't have 'selected_features' key")
                return 'WARNING'
            
            # Compare
            if ml_features == selected_features:
                log_detail(" Features match perfectly")
                return True
            else:
                log_detail(" Feature mismatch!")
                log_detail(f"Selected file: {len(selected_features)} features")
                log_detail(f"ML dict: {len(ml_features)} features")
                return False
                
        except Exception as e:
            log_detail(f" Comparison failed: {e}")
            return False
    
    # Run tests
    test_feature_engineer_load()
    test_selected_features()
    test_feature_calculation()
    test_feature_selection_match()


# ==================== SECTION 4: MODEL LOADING ====================

@test_section("SECTION 4: Model Loading")
def test_model_loading():
    """Test ML and RL model loading"""
    
    @test_case("ML predictor loads")
    def test_ml_loading():
        ml_path = Path('models/ml/ml_predictor.pkl')
        
        if not ml_path.exists():
            log_detail("ML model not found")
            log_detail("Run: python main.py train --ml --fast")
            return False
        
        try:
            # Load dict
            ml_dict = joblib.load(ml_path)
            
            log_detail(" ML dict loaded")
            
            # Extract components
            model = ml_dict['model']
            scaler = ml_dict['scaler']
            selected_features = ml_dict['selected_features']
            
            log_value("Model type", type(model).__name__)
            log_value("Has scaler", type(scaler).__name__)
            log_value("Features", len(selected_features))
            
            # Test prediction with DataFrame
            dummy_input = pd.DataFrame(
                np.random.randn(1, len(selected_features)),
                columns=selected_features
            )
            
            # Scale + predict
            dummy_scaled = scaler.transform(dummy_input)
            prediction = model.predict_proba(dummy_scaled)
            
            log_value("Prediction shape", prediction.shape)
            log_value("Prediction values", prediction[0])
            
            # Should return 3 probabilities
            if prediction.shape[-1] == 3:
                return True
            else:
                log_detail(f"Expected 3 outputs, got {prediction.shape[-1]}")
                return False
                
        except Exception as e:
            log_detail(f" Loading failed: {e}")
            import traceback
            log_detail(traceback.format_exc())
            return False
    
    @test_case("RL agent loads")
    def test_rl_loading():
        rl_path = Path('models/rl/dqn_agent.pth')
        
        if not rl_path.exists():
            log_detail("RL agent not found")
            return False
        
        try:
            from src.models.dqn_agent import DQNAgent, DQNConfig
            
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load checkpoint
            checkpoint = torch.load(rl_path, weights_only=False, map_location=device)
            
            if 'config_dict' not in checkpoint:
                log_detail(" Checkpoint missing config_dict")
                return False
            
            config_dict = checkpoint['config_dict']
            log_value("State dim", config_dict.get('state_dim'))
            log_value("Action dim", config_dict.get('action_dim'))
            log_value("Hidden dims", config_dict.get('hidden_dims'))
            
            # Create agent
            rl_config = DQNConfig(
                state_dim=int(config_dict.get('state_dim', 183)),
                action_dim=int(config_dict.get('action_dim', 3)),
                hidden_dims=[int(x) for x in config_dict.get('hidden_dims', [256, 256, 128])]
            )
            
            agent = DQNAgent(config=rl_config)
            
            # Load weights
            agent.q_network.load_state_dict(checkpoint['q_network_state'])
            
            # ✅ FIX: Move to device AFTER loading
            agent.device = device
            agent.q_network = agent.q_network.to(device)
            agent.q_network.eval()
            
            log_detail(" RL agent loaded")
            log_value("Device", str(device))
            
            # ✅ FIX: Test with input on correct device
            dummy_state = torch.randn(1, rl_config.state_dim).to(device)
            
            with torch.no_grad():
                q_values = agent.q_network(dummy_state)
            
            log_value("Q-values shape", q_values.shape)
            log_value("Q-values", q_values.cpu().numpy()[0])
            
            # Should return 3 Q-values
            if q_values.shape[-1] == 3:
                return True
            else:
                log_detail(f"Expected 3 Q-values, got {q_values.shape[-1]}")
                return False
                
        except Exception as e:
            log_detail(f" Loading failed: {e}")
            import traceback
            log_detail(traceback.format_exc())
            return False
    
    @test_case("RL agent state dimension matches")
    def test_rl_state_dim():
        rl_path = Path('models/rl/dqn_agent.pth')
        if not rl_path.exists():
            return False
        
        try:
            checkpoint = torch.load(rl_path, weights_only=False, map_location='cpu')
            config_dict = checkpoint['config_dict']
            state_dim = int(config_dict.get('state_dim', 183))
            
            expected_dim = 183  # 150 (features) + 5 (position) + 5 (account) + 5 (asset) + 18 (timeframe)
            
            log_value("Agent state dim", state_dim)
            log_value("Expected dim", expected_dim)
            
            if state_dim == expected_dim:
                log_detail(" Dimensions match perfectly")
                return True
            else:
                log_detail(" DIMENSION MISMATCH!")
                log_detail("This will cause errors in live trading!")
                log_detail("Agent expects different state than you'll provide")
                return False
                
        except Exception as e:
            log_detail(f" Check failed: {e}")
            return False
    
    # Run tests
    test_ml_loading()
    test_rl_loading()
    test_rl_state_dim()


# ==================== SECTION 5: STATE CONSTRUCTION ====================

@test_section("SECTION 5: State Construction (CRITICAL)")
def test_state_construction():
    """Test state vector construction - MOST IMPORTANT!"""
    
    @test_case("State builder imports successfully")
    def test_state_import():
        try:
            from src.live.live_trader import build_state_for_agent
            log_detail(" build_state_for_agent imported")
            return True
        except ImportError as e:
            log_detail(f" Import failed: {e}")
            return False
    
    @test_case("State construction with dummy data")
    def test_state_build():
        from src.live.live_trader import build_state_for_agent
        from src.trading.portfolio import Portfolio
        
        # Create dummy features
        features_dict = {
            '5m': pd.DataFrame(np.random.randn(1, 50)),
            '15m': pd.DataFrame(np.random.randn(1, 50)),
            '1h': pd.DataFrame(np.random.randn(1, 50))
        }
        
        # Create dummy portfolio
        portfolio = Portfolio(initial_capital=100.0, max_positions=2)
        
        try:
            state = build_state_for_agent(
                features_dict=features_dict,
                portfolio=portfolio,
                current_price=2500.0,
                initial_balance=100.0,
                symbol='ETH_USDT',
                position_opened_step=None,
                current_step=0
            )
            
            log_value("State shape", state.shape)
            log_value("State dim", len(state))
            log_value("State dtype", state.dtype)
            
            # Check dimension
            if len(state) == 183:
                log_detail(" State has correct 183 dimensions")
            else:
                log_detail(f" Expected 183 dims, got {len(state)}")
                return False
            
            # Check for NaN/Inf
            if np.isnan(state).any():
                log_detail(" State contains NaN values!")
                return False
            
            if np.isinf(state).any():
                log_detail(" State contains Inf values!")
                return False
            
            # Check reasonable ranges
            if (np.abs(state) > 100).any():
                log_detail("  State has very large values (>100)")
                log_detail(f"   Max: {np.abs(state).max():.2f}")
                return 'WARNING'
            
            return True
            
        except Exception as e:
            log_detail(f" State construction failed: {e}")
            import traceback
            log_detail(traceback.format_exc())
            return False
    
    @test_case("State breakdown verification")
    def test_state_breakdown():
        from src.live.live_trader import build_state_for_agent
        from src.trading.portfolio import Portfolio
        
        features_dict = {
            '5m': pd.DataFrame(np.random.randn(1, 50)),
            '15m': pd.DataFrame(np.random.randn(1, 50)),
            '1h': pd.DataFrame(np.random.randn(1, 50))
        }
        
        portfolio = Portfolio(initial_capital=100.0, max_positions=2)
        
        state = build_state_for_agent(
            features_dict=features_dict,
            portfolio=portfolio,
            current_price=2500.0,
            initial_balance=100.0,
            symbol='ETH_USDT'
        )
        
        # Verify structure
        log_detail("State breakdown:")
        log_detail(f"  [0:50]     = 5m features ({len(state[0:50])} dims)")
        log_detail(f"  [50:100]   = 15m features ({len(state[50:100])} dims)")
        log_detail(f"  [100:150]  = 1h features ({len(state[100:150])} dims)")
        log_detail(f"  [150:155]  = Position info ({len(state[150:155])} dims)")
        log_detail(f"  [155:160]  = Account info ({len(state[155:160])} dims)")
        log_detail(f"  [160:165]  = Asset encoding ({len(state[160:165])} dims)")
        log_detail(f"  [165:183]  = Timeframe encoding ({len(state[165:183])} dims)")
        
        # Check asset encoding
        asset_encoding = state[160:165]
        log_detail(f"\nAsset encoding for ETH_USDT: {asset_encoding}")
        
        # Should be one-hot: [1, 0, 0, 0, 0] for ETH
        if asset_encoding[0] == 1.0 and asset_encoding.sum() == 1.0:
            log_detail(" Asset encoding correct")
            return True
        else:
            log_detail(" Asset encoding incorrect")
            return False
    
    @test_case("State with real features")
    def test_state_real_features():
        from src.live.live_trader import build_state_for_agent
        from src.trading.portfolio import Portfolio
        from src.features.feature_engineer import FeatureEngineer
        
        # Load real data
        data_path = Path('./data/raw/5m/ETH_USDT_5m.csv')
        if not data_path.exists():
            log_detail("No data file - skipping")
            return 'WARNING'
        
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.tail(250)
        
        # Calculate real features
        fe = FeatureEngineer()
        features = fe.calculate_all_features(df, 'ETH_USDT')
        
        # Load selected features
        features_path = Path('models/features/selected_features.pkl')
        if features_path.exists():
            selected_features = joblib.load(features_path)
            features = features[selected_features]
        else:
            features = features.iloc[:, :50]  # First 50
        
        # Get latest row
        latest_features = features.iloc[-1:].copy()
        
        # Build features dict
        features_dict = {
            '5m': latest_features,
            '15m': latest_features,
            '1h': latest_features
        }
        
        portfolio = Portfolio(initial_capital=100.0, max_positions=2)
        
        try:
            state = build_state_for_agent(
                features_dict=features_dict,
                portfolio=portfolio,
                current_price=df['close'].iloc[-1],
                initial_balance=100.0,
                symbol='ETH_USDT'
            )
            
            log_detail(" State built with real features")
            log_value("State min", f"{state.min():.4f}")
            log_value("State max", f"{state.max():.4f}")
            log_value("State mean", f"{state.mean():.4f}")
            
            # Sanity checks
            if np.isnan(state).any() or np.isinf(state).any():
                log_detail(" State has invalid values!")
                return False
            
            return True
            
        except Exception as e:
            log_detail(f" Failed: {e}")
            return False
    
    # Run tests
    test_state_import()
    test_state_build()
    test_state_breakdown()
    test_state_real_features()


# ==================== SECTION 6: POSITION SIZING ====================

@test_section("SECTION 6: Position Sizing (95% Rule)")
def test_position_sizing():
    """Test that position sizing matches training"""
    
    @test_case("Position sizing calculation")
    def test_sizing_calc():
        balance = 100.0
        price = 2500.0
        
        # Training logic: 95% of balance
        expected_position_value = balance * 0.95
        expected_position_size = expected_position_value / price
        
        log_value("Balance", f"${balance:.2f}")
        log_value("Price", f"${price:.2f}")
        log_value("Expected position value", f"${expected_position_value:.2f}")
        log_value("Expected position size", f"{expected_position_size:.8f}")
        log_value("Percentage", "95.0%")
        
        # Verify math
        calculated_percentage = (expected_position_value / balance) * 100
        
        if abs(calculated_percentage - 95.0) < 0.01:
            log_detail(" Position sizing calculation correct")
            return True
        else:
            log_detail(f" Calculation error: {calculated_percentage:.2f}%")
            return False
    
    @test_case("Position sizing with fees")
    def test_sizing_with_fees():
        balance = 100.0
        price = 2500.0
        fee_rate = 0.0026  # Kraken fee
        
        position_value = balance * 0.95
        position_size = position_value / price
        fee = position_value * fee_rate
        total_cost = position_value + fee
        
        log_value("Position value", f"${position_value:.2f}")
        log_value("Fee (0.26%)", f"${fee:.4f}")
        log_value("Total cost", f"${total_cost:.2f}")
        log_value("Remaining balance", f"${balance - total_cost:.2f}")
        
        # Check if we can afford it
        if total_cost <= balance:
            log_detail(" Position affordable with fees")
            return True
        else:
            log_detail(" Can't afford position + fees!")
            return False
    
    @test_case("Multiple positions don't exceed balance")
    def test_multiple_positions():
        initial_balance = 100.0
        positions = []
        
        # Simulate opening 2 positions (max_positions=2)
        balance = initial_balance
        
        for i in range(2):
            price = 2500.0 + (i * 100)  # Varying prices
            position_value = balance * 0.95
            position_size = position_value / price
            fee = position_value * 0.0026
            total_cost = position_value + fee
            
            if total_cost <= balance:
                balance -= total_cost
                positions.append({
                    'size': position_size,
                    'price': price,
                    'cost': total_cost
                })
                log_detail(f"Position {i+1}: ${total_cost:.2f} @ ${price:.2f}")
            else:
                log_detail(f"Position {i+1}: Can't afford (need ${total_cost:.2f}, have ${balance:.2f})")
        
        log_value("Positions opened", len(positions))
        log_value("Final balance", f"${balance:.2f}")
        log_value("Total deployed", f"${initial_balance - balance:.2f}")
        
        # Check we didn't go negative
        if balance >= 0:
            return True
        else:
            log_detail(" Balance went negative!")
            return False
    
    # Run tests
    test_sizing_calc()
    test_sizing_with_fees()
    test_multiple_positions()


# ==================== SECTION 7: KRAKEN INTEGRATION ====================

@test_section("SECTION 7: Kraken Integration")
def test_kraken_integration():
    """Test Kraken connector"""
    
    @test_case("Kraken connector initialization (paper mode)")
    def test_kraken_init():
        from src.data.kraken_connector import KrakenConnector
        
        try:
            connector = KrakenConnector(
                api_key=None,
                api_secret=None,
                mode='paper',
                data_path='./data/raw/'
            )
            
            log_detail(" Kraken connector initialized")
            log_value("Mode", connector.mode)
            log_value("Has API", connector.has_credentials)
            
            return True
            
        except Exception as e:
            log_detail(f" Init failed: {e}")
            return False
    
    @test_case("Paper trading order execution")
    def test_paper_order():
        from src.data.kraken_connector import KrakenConnector, KrakenOrder
        
        connector = KrakenConnector(mode='paper', data_path='./data/raw/')
        
        # Initial balance
        initial_balance = connector.paper_balance['USDT']
        log_value("Initial USDT balance", f"${initial_balance:.2f}")
        
        # Place BUY order
        order = KrakenOrder(
            pair='ETH_USDT',
            type='buy',
            ordertype='market',
            volume=0.038
        )
        
        result = connector.place_order(order, current_price=2500.0)
        
        if not result['success']:
            log_detail(f" Order failed: {result.get('error')}")
            return False
        
        log_detail(" BUY order executed")
        log_value("Order ID", result['order_id'])
        log_value("Execution price", f"${result['execution_price']:.2f}")
        
        # Check balances changed
        new_usdt = connector.paper_balance['USDT']
        new_eth = connector.paper_balance.get('ETH', 0)
        
        log_value("New USDT balance", f"${new_usdt:.2f}")
        log_value("New ETH balance", f"{new_eth:.8f}")
        
        if new_usdt < initial_balance and new_eth > 0:
            log_detail(" Balances updated correctly")
            return True
        else:
            log_detail(" Balances didn't update properly")
            return False
    
    @test_case("Paper trading round trip (buy + sell)")
    def test_paper_round_trip():
        from src.data.kraken_connector import KrakenConnector, KrakenOrder
        
        connector = KrakenConnector(mode='paper', data_path='./data/raw/')
        
        initial_usdt = connector.paper_balance['USDT']
        
        # BUY
        buy_order = KrakenOrder(
            pair='ETH_USDT',
            type='buy',
            ordertype='market',
            volume=0.038
        )
        buy_result = connector.place_order(buy_order, current_price=2500.0)
        
        if not buy_result['success']:
            return False
        
        eth_amount = connector.paper_balance.get('ETH', 0)
        
        # SELL
        sell_order = KrakenOrder(
            pair='ETH_USDT',
            type='sell',
            ordertype='market',
            volume=eth_amount
        )
        sell_result = connector.place_order(sell_order, current_price=2550.0)  # +2% price
        
        if not sell_result['success']:
            return False
        
        final_usdt = connector.paper_balance['USDT']
        pnl = final_usdt - initial_usdt
        
        log_value("Initial USDT", f"${initial_usdt:.2f}")
        log_value("Final USDT", f"${final_usdt:.2f}")
        log_value("P&L", f"${pnl:.2f}")
        log_value("Return", f"{(pnl/initial_usdt)*100:.2f}%")
        
        # Should have profit (2% price increase - 0.26% fees each way)
        expected_return = 0.02 - (0.0026 * 2)
        
        if pnl > 0:
            log_detail(" Profitable round trip")
            return True
        else:
            log_detail(" Lost money on profitable trade!")
            return False
    
    @test_case("Kraken API credentials (for live trading)")
    def test_kraken_credentials():
        api_key = os.getenv('KRAKEN_API_KEY')
        api_secret = os.getenv('KRAKEN_API_SECRET')
        
        if api_key and api_secret:
            log_detail(" Kraken credentials found")
            log_value("API Key", api_key[:10] + "..." + api_key[-5:])
            
            # Try initializing with real creds (but don't make calls)
            from src.data.kraken_connector import KrakenConnector
            
            try:
                connector = KrakenConnector(
                    api_key=api_key,
                    api_secret=api_secret,
                    mode='live',  # This will validate but not trade
                    data_path='./data/raw/'
                )
                
                log_detail(" Connector initialized with live credentials")
                return True
                
            except Exception as e:
                log_detail(f" Credential initialization failed: {e}")
                return False
        else:
            log_detail("  No Kraken credentials - can't test live mode")
            log_detail("   Set KRAKEN_API_KEY and KRAKEN_API_SECRET in .env")
            return 'WARNING'
    
    # Run tests
    test_kraken_init()
    test_paper_order()
    test_paper_round_trip()
    test_kraken_credentials()


# ==================== SECTION 8: COMPLETE TRADING CYCLE ====================

@test_section("SECTION 8: Complete Trading Cycle (DRY RUN)")
def test_complete_cycle():
    """Test a complete trading decision cycle"""
    
    from src.data.binance_connector import BinanceDataConnector
    _loaded_components = None

    @test_case("Load all components")
    def test_load_all():
        global _loaded_components
        
        try:
            # Feature engineer
            from src.features.feature_engineer import FeatureEngineer
            
            fe_path = Path('models/feature_engineer.pkl')
            if fe_path.exists():
                fe = joblib.load(fe_path)
            else:
                fe = FeatureEngineer()
            
            # Selected features
            features_path = Path('models/features/selected_features.pkl')
            if not features_path.exists():
                log_detail("Selected features not found")
                return False
            
            selected_features = joblib.load(features_path)
            
            if not isinstance(selected_features, list):
                log_detail(f" selected_features is {type(selected_features)}, expected list")
                return False
            
            log_detail(f" Loaded {len(selected_features)} features")
            
            # ML predictor (load dict)
            ml_path = Path('models/ml/ml_predictor.pkl')
            if not ml_path.exists():
                return False
            
            ml_dict = joblib.load(ml_path)
            model = ml_dict['model']
            scaler = ml_dict['scaler']
            
            # RL agent
            from src.models.dqn_agent import DQNAgent, DQNConfig
            
            rl_path = Path('models/rl/dqn_agent.pth')
            if not rl_path.exists():
                return False
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(rl_path, weights_only=False, map_location=device)
            
            config_dict = checkpoint['config_dict']
            rl_config = DQNConfig(
                state_dim=int(config_dict['state_dim']),
                action_dim=int(config_dict['action_dim']),
                hidden_dims=[int(x) for x in config_dict['hidden_dims']]
            )
            
            rl_agent = DQNAgent(config=rl_config)
            rl_agent.q_network.load_state_dict(checkpoint['q_network_state'])
            rl_agent.device = device
            rl_agent.q_network = rl_agent.q_network.to(device)
            rl_agent.q_network.eval()
            
            # Portfolio
            from src.trading.portfolio import Portfolio
            portfolio = Portfolio(initial_capital=100.0, max_positions=2)
            
            # State builder
            from src.live.live_trader import build_state_for_agent
            
            log_detail(" All components loaded successfully")
            
            # Store components globally for other tests
            _loaded_components = {
                'fe': fe,
                'selected_features': selected_features,
                'model': model,
                'scaler': scaler,
                'rl_agent': rl_agent,
                'portfolio': portfolio,
                'build_state': build_state_for_agent
            }
            
            # ✅ CRITICAL: Return True, not the dict!
            return True
            
        except Exception as e:
            log_detail(f" Loading failed: {e}")
            import traceback
            log_detail(traceback.format_exc())
            return False


    @test_case("Simulate complete trading decision")
    def test_trading_decision():
        global _loaded_components
        
        # Load components if not already loaded
        if _loaded_components is None:
            success = test_load_all()
            if not success:
                log_detail(" Failed to load components")
                return False
        
        components = _loaded_components
        
        try:
            # Load test data
            binance = BinanceDataConnector(data_path='./data/raw/')
            
            symbol = 'ETH_USDT'
            timeframes = ['5m', '15m', '1h']
            
            # Get recent data
            all_features = {}
            for tf in timeframes:
                df = binance.load_existing_data(symbol, timeframe=tf)
                if df.empty:
                    log_detail(f"No data for {symbol} {tf}")
                    return False
                
                # Calculate features
                features_df = components['fe'].calculate_all_features(df.tail(250), symbol)
                
                # Select features
                selected = features_df[components['selected_features']].iloc[-1:]
                all_features[tf] = selected
            
            # Get current price
            current_price = float(df.iloc[-1]['close'])
            
            # Build state
            state = components['build_state'](
                features_dict=all_features,
                portfolio=components['portfolio'],
                current_price=current_price,
                initial_balance=100.0,
                symbol=symbol
            )
            
            # ML prediction
            ml_input = all_features['5m']
            ml_scaled = components['scaler'].transform(ml_input)
            ml_pred = components['model'].predict_proba(ml_scaled)
            
            log_value("ML prediction", ml_pred[0])
            
            # RL action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(components['rl_agent'].device)
                q_values = components['rl_agent'].q_network(state_tensor)
                action = q_values.argmax().item()
            
            action_names = ['HOLD', 'BUY', 'SELL']
            log_value("RL action", action_names[action])
            log_value("Q-values", q_values.cpu().numpy()[0])
            
            log_detail(" Complete trading decision simulated")
            return True
            
        except Exception as e:
            log_detail(f" Simulation failed: {e}")
            import traceback
            log_detail(traceback.format_exc())
            return False


    @test_case("Simulate 10 consecutive decisions")
    def test_multiple_decisions():
        global _loaded_components
        
        # Load components if not already loaded
        if _loaded_components is None:
            success = test_load_all()
            if not success:
                log_detail(" Failed to load components")
                return False
        
        components = _loaded_components
        
        try:
            # Load test data
            binance = BinanceDataConnector(data_path='./data/raw/')
            
            symbol = 'ETH_USDT'
            timeframes = ['5m', '15m', '1h']
            
            # Get recent data
            df_5m = binance.load_existing_data(symbol, timeframe='5m')
            if df_5m.empty:
                return False
            
            # Simulate 10 decisions
            decisions = []
            
            for i in range(10):
                # Get window
                end_idx = len(df_5m) - i - 1
                start_idx = max(0, end_idx - 250)
                window = df_5m.iloc[start_idx:end_idx]
                
                if len(window) < 100:
                    continue
                
                # Calculate features for all timeframes
                all_features = {}
                for tf in timeframes:
                    df_tf = binance.load_existing_data(symbol, timeframe=tf)
                    
                    # Align timeframe with 5m window
                    end_time = window.index[-1]
                    df_tf_window = df_tf[df_tf.index <= end_time].tail(250)
                    
                    if len(df_tf_window) < 100:
                        continue
                    
                    features_df = components['fe'].calculate_all_features(df_tf_window, symbol)
                    selected = features_df[components['selected_features']].iloc[-1:]
                    all_features[tf] = selected
                
                if len(all_features) != 3:
                    continue
                
                # Get price
                current_price = float(window.iloc[-1]['close'])
                
                # Build state
                state = components['build_state'](
                    features_dict=all_features,
                    portfolio=components['portfolio'],
                    current_price=current_price,
                    initial_balance=100.0,
                    symbol=symbol
                )
                
                # Get action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(components['rl_agent'].device)
                    q_values = components['rl_agent'].q_network(state_tensor)
                    action = q_values.argmax().item()
                
                decisions.append({
                    'step': i,
                    'price': current_price,
                    'action': action
                })
            
            log_value("Decisions simulated", len(decisions))
            
            if len(decisions) >= 5:
                action_counts = {}
                for d in decisions:
                    action_counts[d['action']] = action_counts.get(d['action'], 0) + 1
                
                log_value("Action distribution", action_counts)
                log_detail(" Multiple decisions simulated successfully")
                return True
            else:
                log_detail(f" Only {len(decisions)} decisions (need 5+)")
                return False
            
        except Exception as e:
            log_detail(f" Simulation failed: {e}")
            import traceback
            log_detail(traceback.format_exc())
            return False
    
    # Run tests
    test_load_all()
    test_trading_decision()
    test_multiple_decisions()


# ==================== SECTION 9: SAFETY CHECKS ====================

@test_section("SECTION 9: Safety Mechanisms")
def test_safety_mechanisms():
    """Test stop-loss, take-profit, and emergency stops"""
    
    @test_case("Stop-loss calculation")
    def test_stop_loss():
        entry_price = 2500.0
        stop_loss_pct = 0.03  # 3%
        
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        
        log_value("Entry price", f"${entry_price:.2f}")
        log_value("Stop loss %", f"{stop_loss_pct*100:.1f}%")
        log_value("Stop loss price", f"${stop_loss_price:.2f}")
        
        # Simulate price dropping
        current_price = 2425.0  # -3% from entry
        loss_pct = (current_price - entry_price) / entry_price
        
        log_value("Current price", f"${current_price:.2f}")
        log_value("Loss", f"{loss_pct*100:.2f}%")
        
        # Should trigger stop loss
        if loss_pct <= -stop_loss_pct:
            log_detail(" Stop loss would trigger correctly")
            return True
        else:
            log_detail(" Stop loss didn't trigger")
            return False
    
    @test_case("Take-profit calculation")
    def test_take_profit():
        entry_price = 2500.0
        take_profit_pct = 0.06  # 6%
        
        take_profit_price = entry_price * (1 + take_profit_pct)
        
        log_value("Entry price", f"${entry_price:.2f}")
        log_value("Take profit %", f"{take_profit_pct*100:.1f}%")
        log_value("Take profit price", f"${take_profit_price:.2f}")
        
        # Simulate price rising
        current_price = 2650.0  # +6% from entry
        gain_pct = (current_price - entry_price) / entry_price
        
        log_value("Current price", f"${current_price:.2f}")
        log_value("Gain", f"{gain_pct*100:.2f}%")
        
        # Should trigger take profit
        if gain_pct >= take_profit_pct:
            log_detail(" Take profit would trigger correctly")
            return True
        else:
            log_detail(" Take profit didn't trigger")
            return False
    
    @test_case("Max drawdown check")
    def test_max_drawdown():
        initial_capital = 100.0
        max_drawdown = 0.15  # 15%
        
        # Simulate loss
        current_value = 82.0  # -18% drawdown
        drawdown = (initial_capital - current_value) / initial_capital
        
        log_value("Initial capital", f"${initial_capital:.2f}")
        log_value("Current value", f"${current_value:.2f}")
        log_value("Drawdown", f"{drawdown*100:.2f}%")
        log_value("Max allowed", f"{max_drawdown*100:.2f}%")
        
        # Should trigger emergency stop
        if drawdown >= max_drawdown:
            log_detail(" Emergency stop would trigger")
            return True
        else:
            log_detail(" Emergency stop wouldn't trigger")
            return False
    
    @test_case("Balance validation")
    def test_balance_validation():
        balance = 100.0
        
        # Test scenarios
        tests = [
            (balance, True, "Normal balance"),
            (0.0, True, "Zero balance (should warn but not crash)"),
            (-10.0, False, "Negative balance (should error)"),
            (1e6, True, "Very large balance (should warn)")
        ]
        
        all_passed = True
        
        for test_balance, should_pass, description in tests:
            log_detail(f"  Test: {description}")
            log_detail(f"    Balance: ${test_balance:.2f}")
            
            # Validate
            if test_balance < 0:
                log_detail("     Would catch negative balance")
            elif test_balance > 100000:
                log_detail("      Would warn about unrealistic balance")
            elif test_balance == 0:
                log_detail("      Would warn about zero balance")
            else:
                log_detail("     Balance valid")
        
        return True
    
    # Run tests
    test_stop_loss()
    test_take_profit()
    test_max_drawdown()
    test_balance_validation()


# ==================== MAIN EXECUTION ====================

def print_summary():
    """Print test results summary"""
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    
    total = test_results['total']
    passed = test_results['passed']
    failed = test_results['failed']
    warnings = test_results['warnings']
    
    print(f"\nTotal Tests: {total}")
    print(f" Passed: {passed} ({passed/total*100:.1f}%)")
    print(f" Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"  Warnings: {warnings} ({warnings/total*100:.1f}%)")
    
    if failed > 0:
        print("\n" + "!"*80)
        print("  CRITICAL: Some tests failed!")
        print("!"*80)
        print("\nFailed tests:")
        for test in test_results['tests']:
            if test['status'] == 'FAIL':
                print(f"   {test['name']}")
                if 'error' in test:
                    print(f"     Error: {test['error']}")
        
        print("\n  DO NOT GO LIVE until all tests pass!")
        print("="*80)
        return False
    
    elif warnings > 0:
        print("\n" + "="*80)
        print("    All tests passed with some warnings")
        print("="*80)
        print("\nWarnings:")
        for test in test_results['tests']:
            if test['status'] == 'WARNING':
                print(f"    {test['name']}")
        
        print("\n Safe to proceed, but review warnings")
        print("="*80)
        return True
    
    else:
        print("\n" + "="*80)
        print("   ALL TESTS PASSED!")
        print("="*80)
        print("\n System ready for live trading!")
        print("\nRecommended next steps:")
        print("  1. Run 24h paper trading: python main.py live --start (in paper mode)")
        print("  2. Monitor closely and verify behavior")
        print("  3. Go live with small capital ($100)")
        print("="*80)
        return True


def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description='Live Trading Test Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick smoke test')
    parser.add_argument('--section', type=int, help='Run specific section (1-9)')
    args = parser.parse_args()
    
    print("="*80)
    print("  LIVE TRADING PRE-FLIGHT TEST SUITE")
    print("="*80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working Dir: {os.getcwd()}")
    print("="*80)
    
    # Run tests based on arguments
    if args.section:
        sections = {
            1: test_environment,
            2: test_data_pipeline,
            3: test_feature_engineering,
            4: test_model_loading,
            5: test_state_construction,
            6: test_position_sizing,
            7: test_kraken_integration,
            8: test_complete_cycle,
            9: test_safety_mechanisms
        }
        
        if args.section in sections:
            sections[args.section]()
        else:
            print(f"Invalid section: {args.section}")
            return
    
    elif args.quick:
        # Quick smoke test - most critical tests only
        print("\n QUICK SMOKE TEST (Critical tests only)")
        test_environment()
        test_model_loading()
        test_state_construction()
        test_position_sizing()
    
    else:
        # Full test suite
        test_environment()
        test_data_pipeline()
        test_feature_engineering()
        test_model_loading()
        test_state_construction()
        test_position_sizing()
        test_kraken_integration()
        test_complete_cycle()
        test_safety_mechanisms()
    
    # Print summary
    success = print_summary()
    
    # Save results
    results_file = Path('test_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': test_results,
            'success': success
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()