"""
Test Script for State Management Module
Tests state extraction, representation, and tracking
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import state management
from src.environment.state import (
    StateManager, MarketState, AccountState, TechnicalState
)

def create_sample_data(num_bars=100):
    """Create sample OHLCV data"""
    dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='1h')
    
    # Create realistic price movement
    base_price = 42000
    prices = []
    for i in range(num_bars):
        change = np.random.randn() * 100
        base_price += change
        prices.append(base_price)
    
    data = pd.DataFrame({
        'open': np.array(prices) + np.random.uniform(-50, 50, num_bars),
        'high': np.array(prices) + np.random.uniform(50, 150, num_bars),
        'low': np.array(prices) - np.random.uniform(50, 150, num_bars),
        'close': prices,
        'volume': np.random.uniform(100, 1000, num_bars)
    }, index=dates)
    
    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

def create_sample_indicators(data):
    """Create sample technical indicators"""
    indicators = pd.DataFrame(index=data.index)
    
    indicators['rsi_14'] = np.random.uniform(30, 70, len(data))
    indicators['macd'] = np.random.uniform(-100, 100, len(data))
    indicators['macd_signal'] = indicators['macd'] * 0.8
    indicators['bb_position'] = np.random.uniform(0, 1, len(data))
    indicators['support_1'] = data['close'] * 0.95
    indicators['resistance_1'] = data['close'] * 1.05
    
    return indicators

def test_initialization():
    """Test StateManager initialization"""
    print("\n" + "="*60)
    print("TEST 1: StateManager Initialization")
    print("="*60)
    
    try:
        # Default initialization
        manager = StateManager()
        print(f" Default initialization successful")
        print(f"   - Window size: {manager.window_size}")
        print(f"   - Feature version: {manager.feature_version}")
        print(f"   - Normalize: {manager.normalize}")
        
        # Custom initialization
        manager = StateManager(
            window_size=30,
            feature_version='v2',
            normalize=False
        )
        print(f" Custom initialization successful")
        print(f"   - Window size: {manager.window_size}")
        print(f"   - Feature version: {manager.feature_version}")
        print(f"   - Normalize: {manager.normalize}")
        
        return True
        
    except Exception as e:
        print(f" Initialization failed: {e}")
        return False

def test_market_state_extraction():
    """Test market state extraction"""
    print("\n" + "="*60)
    print("TEST 2: Market State Extraction")
    print("="*60)
    
    try:
        manager = StateManager(window_size=20)
        data = create_sample_data(100)
        
        # Extract market state at different positions
        current_step = 50
        market_state = manager.extract_market_state(data, current_step)
        
        print(f" Market state extracted")
        print(f"   - Current price: ${market_state.current_price:.2f}")
        print(f"   - High: ${market_state.high:.2f}")
        print(f"   - Low: ${market_state.low:.2f}")
        print(f"   - Volume: {market_state.volume:.2f}")
        print(f"   - Volatility: {market_state.volatility:.4f}")
        print(f"   - Trend: {market_state.trend:.4f}")
        print(f"   - Support: ${market_state.support_level:.2f}")
        print(f"   - Resistance: ${market_state.resistance_level:.2f}")
        print(f"   - Price history shape: {market_state.price_history.shape}")
        
        # Test at beginning (edge case)
        market_state_start = manager.extract_market_state(data, 5)
        print(f" Edge case (early position) handled")
        
        return True
        
    except Exception as e:
        print(f" Market state extraction failed: {e}")
        return False

def test_account_state_extraction():
    """Test account state extraction"""
    print("\n" + "="*60)
    print("TEST 3: Account State Extraction")
    print("="*60)
    
    try:
        # Create a mock environment
        class MockEnv:
            def __init__(self):
                self.balance = 10000
                self.equity = 10500
                self.position_size = 0.1
                self.position = 1  # Long
                self.entry_price = 42000
                self.unrealized_pnl = 100
                self.realized_pnl = 200
                self.trades = [{'pnl': 100}, {'pnl': -50}, {'pnl': 150}]
                self.max_drawdown = 0.1
                self.current_drawdown = 0.05
                
            def _get_win_rate(self):
                return 0.6
            
            def _get_sharpe_ratio(self):
                return 1.5
        
        manager = StateManager()
        mock_env = MockEnv()
        
        account_state = manager.extract_account_state(mock_env)
        
        print(f" Account state extracted")
        print(f"   - Balance: ${account_state.balance:.2f}")
        print(f"   - Equity: ${account_state.equity:.2f}")
        print(f"   - Position type: {account_state.position_type}")
        print(f"   - Position size: {account_state.position_size}")
        print(f"   - Entry price: ${account_state.entry_price:.2f}")
        print(f"   - Unrealized PnL: ${account_state.unrealized_pnl:.2f}")
        print(f"   - Realized PnL: ${account_state.realized_pnl:.2f}")
        print(f"   - Win rate: {account_state.win_rate:.2%}")
        print(f"   - Sharpe ratio: {account_state.sharpe_ratio:.2f}")
        
        return True
        
    except Exception as e:
        print(f" Account state extraction failed: {e}")
        return False

def test_technical_state_extraction():
    """Test technical state extraction"""
    print("\n" + "="*60)
    print("TEST 4: Technical State Extraction")
    print("="*60)
    
    try:
        manager = StateManager()
        data = create_sample_data(100)
        indicators = create_sample_indicators(data)
        
        current_step = 50
        
        # Test with indicators
        tech_state = manager.extract_technical_state(data, indicators, current_step)
        
        print(f" Technical state extracted")
        print(f"   - RSI: {tech_state.rsi:.4f}")
        print(f"   - MACD: {tech_state.macd:.4f}")
        print(f"   - MACD Signal: {tech_state.macd_signal:.4f}")
        print(f"   - BB Position: {tech_state.bb_position:.4f}")
        print(f"   - Volume ratio: {tech_state.volume_ratio:.4f}")
        print(f"   - Momentum: {tech_state.momentum:.4f}")
        print(f"   - Support distance: {tech_state.support_distance:.4f}")
        print(f"   - Resistance distance: {tech_state.resistance_distance:.4f}")
        
        # Test without indicators (should return defaults)
        tech_state_empty = manager.extract_technical_state(data, None, current_step)
        print(f" Default values returned when no indicators")
        
        return True
        
    except Exception as e:
        print(f" Technical state extraction failed: {e}")
        return False

def test_state_vector_creation():
    """Test state vector creation"""
    print("\n" + "="*60)
    print("TEST 5: State Vector Creation")
    print("="*60)
    
    try:
        manager = StateManager(window_size=20, feature_version='v1')
        
        # Create sample states
        market_state = MarketState(
            current_price=42000,
            high=42500,
            low=41500,
            volume=500,
            price_history=np.random.uniform(41000, 43000, 20),
            volume_history=np.random.uniform(100, 1000, 20),
            returns_history=np.random.randn(20) * 0.01,
            volatility=0.02,
            trend=0.001,
            support_level=41000,
            resistance_level=43000
        )
        
        account_state = AccountState(
            balance=10000,
            equity=10500,
            position_size=0.1,
            position_type=1,
            entry_price=42000,
            unrealized_pnl=100,
            realized_pnl=200,
            total_trades=10,
            win_rate=0.6,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            current_drawdown=0.05
        )
        
        tech_state = TechnicalState(
            rsi=0.5,
            macd=50,
            macd_signal=45,
            bb_position=0.6,
            volume_ratio=1.2,
            momentum=0.01,
            support_distance=0.02,
            resistance_distance=0.03
        )
        
        # Create state vector
        state_vector = manager.create_state_vector(
            market_state, account_state, tech_state
        )
        
        print(f" State vector created")
        print(f"   - Shape: {state_vector.shape}")
        print(f"   - Dtype: {state_vector.dtype}")
        print(f"   - Min value: {state_vector.min():.4f}")
        print(f"   - Max value: {state_vector.max():.4f}")
        print(f"   - Mean value: {state_vector.mean():.4f}")
        
        # Test v2 features
        manager_v2 = StateManager(feature_version='v2')
        state_vector_v2 = manager_v2.create_state_vector(
            market_state, account_state, tech_state
        )
        print(f" V2 features created")
        print(f"   - V2 Shape: {state_vector_v2.shape}")
        
        return True
        
    except Exception as e:
        print(f" State vector creation failed: {e}")
        return False

def test_state_discretization():
    """Test state discretization"""
    print("\n" + "="*60)
    print("TEST 6: State Discretization")
    print("="*60)
    
    try:
        manager = StateManager()
        
        # Create a continuous state vector
        state_vector = np.array([0.5, -0.3, 0.8, -0.9, 0.0, 1.0, -1.0])
        
        # Discretize with different bin sizes
        discrete_10 = manager.discretize_state(state_vector, n_bins=10)
        discrete_5 = manager.discretize_state(state_vector, n_bins=5)
        
        print(f" State discretization successful")
        print(f"   - Original: {state_vector[:3]}")
        print(f"   - 10 bins: {discrete_10[:20]}...")
        print(f"   - 5 bins: {discrete_5[:20]}...")
        
        # Test that discretization is consistent
        discrete_again = manager.discretize_state(state_vector, n_bins=10)
        assert discrete_10 == discrete_again
        print(f" Discretization is deterministic")
        
        return True
        
    except Exception as e:
        print(f" State discretization failed: {e}")
        return False

def test_state_hashing():
    """Test state hashing"""
    print("\n" + "="*60)
    print("TEST 7: State Hashing")
    print("="*60)
    
    try:
        manager = StateManager()
        
        # Create state vectors
        state1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Same as state1
        state3 = np.array([1.0, 2.0, 3.0, 4.0, 6.0])  # Different
        
        # Hash states
        hash1 = manager.hash_state(state1)
        hash2 = manager.hash_state(state2)
        hash3 = manager.hash_state(state3)
        
        print(f" State hashing successful")
        print(f"   - Hash 1: {hash1}")
        print(f"   - Hash 2: {hash2}")
        print(f"   - Hash 3: {hash3}")
        
        # Test hash properties
        assert hash1 == hash2, "Same states should have same hash"
        assert hash1 != hash3, "Different states should have different hashes"
        assert len(hash1) == 8, "Hash should be 8 characters"
        
        print(f" Hash properties verified")
        
        return True
        
    except Exception as e:
        print(f" State hashing failed: {e}")
        return False

def test_state_visitation_tracking():
    """Test state visitation tracking"""
    print("\n" + "="*60)
    print("TEST 8: State Visitation Tracking")
    print("="*60)
    
    try:
        manager = StateManager()
        
        # Create and visit states
        state1 = np.array([1.0, 2.0, 3.0])
        state2 = np.array([4.0, 5.0, 6.0])
        
        # First visits
        count1 = manager.track_state_visit(state1)
        count2 = manager.track_state_visit(state2)
        
        print(f" First visits tracked")
        print(f"   - State 1 count: {count1}")
        print(f"   - State 2 count: {count2}")
        
        assert count1 == 1
        assert count2 == 1
        
        # Revisit states
        count1_again = manager.track_state_visit(state1)
        count1_third = manager.track_state_visit(state1)
        
        print(f" Revisits tracked")
        print(f"   - State 1 after 3 visits: {count1_third}")
        
        assert count1_third == 3
        
        # Get statistics
        stats = manager.get_state_statistics()
        
        print(f" State statistics computed")
        print(f"   - Unique states: {stats['unique_states']}")
        print(f"   - Total visits: {stats['total_visits']}")
        print(f"   - Max visits: {stats['max_visits']}")
        print(f"   - Average visits: {stats['avg_visits']:.2f}")
        
        return True
        
    except Exception as e:
        print(f" State visitation tracking failed: {e}")
        return False

def test_state_augmentation():
    """Test state augmentation"""
    print("\n" + "="*60)
    print("TEST 9: State Augmentation")
    print("="*60)
    
    try:
        manager = StateManager()
        
        # Base state vector
        base_state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Augment with additional features
        additional_features = {
            'time_of_day': 0.5,
            'day_of_week': 0.2,
            'market_regime': 1.0
        }
        
        augmented = manager.augment_state(base_state, additional_features)
        
        print(f" State augmentation successful")
        print(f"   - Base state shape: {base_state.shape}")
        print(f"   - Augmented shape: {augmented.shape}")
        print(f"   - Additional features: {len(additional_features)}")
        
        assert len(augmented) == len(base_state) + len(additional_features)
        
        # Test without additional features
        no_augment = manager.augment_state(base_state, None)
        assert np.array_equal(no_augment, base_state)
        print(f" No augmentation when features are None")
        
        return True
        
    except Exception as e:
        print(f" State augmentation failed: {e}")
        return False

def test_normalization():
    """Test feature normalization"""
    print("\n" + "="*60)
    print("TEST 10: Feature Normalization")
    print("="*60)
    
    try:
        manager = StateManager(normalize=True)
        
        # Create features with different scales
        features = np.array([100.0, -50.0, 0.5, 1000.0, -200.0])
        
        # Normalize
        normalized = manager._normalize_features(features)
        
        print(f" Normalization successful")
        print(f"   - Original range: [{features.min():.2f}, {features.max():.2f}]")
        print(f"   - Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
        print(f"   - Original: {features}")
        print(f"   - Normalized: {normalized}")
        
        # Check bounds
        assert normalized.min() >= -1
        assert normalized.max() <= 1
        
        return True
        
    except Exception as e:
        print(f" Normalization test failed: {e}")
        return False

def test_advanced_market_features():
    """Test advanced market feature extraction"""
    print("\n" + "="*60)
    print("TEST 11: Advanced Market Features")
    print("="*60)
    
    try:
        manager = StateManager(feature_version='v2')
        
        # Create market state with sufficient history
        market_state = MarketState(
            current_price=42000,
            high=42500,
            low=41500,
            volume=500,
            price_history=np.random.uniform(41000, 43000, 50),
            volume_history=np.random.uniform(100, 1000, 50),
            returns_history=np.random.randn(50) * 0.01,
            volatility=0.02,
            trend=0.001,
            support_level=41000,
            resistance_level=43000
        )
        
        # Extract advanced features
        features = manager._extract_advanced_market_features(market_state)
        
        print(f" Advanced features extracted")
        print(f"   - Number of features: {len(features)}")
        print(f"   - Features: {features}")
        
        assert len(features) > 0
        assert all(isinstance(f, (int, float)) for f in features)
        
        return True
        
    except Exception as e:
        print(f" Advanced features test failed: {e}")
        return False

def test_persistence():
    """Test saving and loading normalization stats"""
    print("\n" + "="*60)
    print("TEST 12: Persistence (Save/Load Stats)")
    print("="*60)
    
    try:
        import tempfile
        import os
        
        manager = StateManager(window_size=30, feature_version='v2')
        
        # Set some stats
        manager.feature_means = {'feature1': 100, 'feature2': 50}
        manager.feature_stds = {'feature1': 10, 'feature2': 5}
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        manager.save_normalization_stats(tmp_path)
        print(f" Stats saved to {tmp_path}")
        
        # Create new manager and load
        new_manager = StateManager()
        new_manager.load_normalization_stats(tmp_path)
        
        print(f" Stats loaded successfully")
        print(f"   - Window size: {new_manager.window_size}")
        print(f"   - Feature version: {new_manager.feature_version}")
        print(f"   - Means: {new_manager.feature_means}")
        print(f"   - Stds: {new_manager.feature_stds}")
        
        # Verify loaded correctly
        assert new_manager.window_size == 30
        assert new_manager.feature_version == 'v2'
        assert new_manager.feature_means == manager.feature_means
        assert new_manager.feature_stds == manager.feature_stds
        
        # Cleanup
        os.remove(tmp_path)
        
        return True
        
    except Exception as e:
        print(f" Persistence test failed: {e}")
        return False

def test_full_pipeline():
    """Test complete state extraction pipeline"""
    print("\n" + "="*60)
    print("TEST 13: Full Pipeline Integration")
    print("="*60)
    
    try:
        # Create complete test data
        data = create_sample_data(100)
        indicators = create_sample_indicators(data)
        
        # Initialize manager
        manager = StateManager(window_size=20)
        
        # Create mock environment
        class MockEnv:
            def __init__(self):
                self.balance = 10000
                self.equity = 10500
                self.position_size = 0.1
                self.position = 1
                self.entry_price = 42000
                self.unrealized_pnl = 100
                self.realized_pnl = 200
                self.trades = []
                self.max_drawdown = 0.1
                self.current_drawdown = 0.05
                
            def _get_win_rate(self):
                return 0.6
            
            def _get_sharpe_ratio(self):
                return 1.5
        
        env = MockEnv()
        
        # Run pipeline for multiple steps
        states = []
        for step in range(30, 40):
            # Extract all state components
            market_state = manager.extract_market_state(data, step)
            account_state = manager.extract_account_state(env)
            tech_state = manager.extract_technical_state(data, indicators, step)
            
            # Create state vector
            state_vector = manager.create_state_vector(
                market_state, account_state, tech_state
            )
            
            # Track visitation
            manager.track_state_visit(state_vector)
            
            states.append(state_vector)
        
        print(f" Pipeline executed for 10 steps")
        print(f"   - States collected: {len(states)}")
        print(f"   - State shape: {states[0].shape}")
        print(f"   - Unique states visited: {manager.get_state_statistics()['unique_states']}")
        
        # Verify all states have same shape
        assert all(s.shape == states[0].shape for s in states)
        print(f" All states have consistent shape")
        
        return True
        
    except Exception as e:
        print(f" Full pipeline test failed: {e}")
        return False

def run_all_tests():
    """Run all state management tests"""
    print("\n" + "="*60)
    print("STATE MANAGEMENT TEST SUITE")
    print("="*60)
    
    tests = [
        ("Initialization", test_initialization),
        ("Market State Extraction", test_market_state_extraction),
        ("Account State Extraction", test_account_state_extraction),
        ("Technical State Extraction", test_technical_state_extraction),
        ("State Vector Creation", test_state_vector_creation),
        ("State Discretization", test_state_discretization),
        ("State Hashing", test_state_hashing),
        ("State Visitation Tracking", test_state_visitation_tracking),
        ("State Augmentation", test_state_augmentation),
        ("Normalization", test_normalization),
        ("Advanced Market Features", test_advanced_market_features),
        ("Persistence", test_persistence),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f" Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = " PASSED" if result else " FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n ALL TESTS PASSED! State management is working correctly!")
    else:
        print(f"\n  {total - passed} tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)