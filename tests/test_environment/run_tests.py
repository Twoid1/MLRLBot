"""
Simple Test Runner for Trading Environment
Run this script to test all the environment modules
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def print_menu():
    """Print test menu"""
    print("\n" + "="*60)
    print("   TRADING ENVIRONMENT TEST RUNNER")
    print("="*60)
    print("\nSelect which tests to run:")
    print("  1. Trading Environment Only")
    print("  2. Reward Functions Only")
    print("  3. State Management Only")
    print("  4. All Individual Tests")
    print("  5. Integration Tests Only")
    print("  6. Performance Tests Only")
    print("  7. Run ALL Tests (Complete Suite)")
    print("  0. Exit")
    print("-"*60)

def run_test_suite(choice):
    """Run selected test suite"""
    
    if choice == '1':
        print("\n Running Trading Environment Tests...")
        from tests.test_trading_env import run_all_tests
        return run_all_tests()
        
    elif choice == '2':
        print("\n Running Reward Functions Tests...")
        from tests.test_rewards import run_all_tests
        return run_all_tests()
        
    elif choice == '3':
        print("\n Running State Management Tests...")
        from tests.test_state import run_all_tests
        return run_all_tests()
        
    elif choice == '4':
        print("\n Running All Individual Tests...")
        results = []
        
        from tests.test_trading_env import run_all_tests as test_env
        print("\n--- Trading Environment ---")
        results.append(test_env())
        
        from tests.test_rewards import run_all_tests as test_rewards
        print("\n--- Reward Functions ---")
        results.append(test_rewards())
        
        from tests.test_state import run_all_tests as test_state
        print("\n--- State Management ---")
        results.append(test_state())
        
        return all(results)
        
    elif choice == '5':
        print("\n Running Integration Tests...")
        from tests.test_all import test_integration
        return test_integration()
        
    elif choice == '6':
        print("\n Running Performance Tests...")
        from tests.test_all import test_performance
        return test_performance()
        
    elif choice == '7':
        print("\n Running Complete Test Suite...")
        from test_all import main
        return main()
        
    else:
        print("Invalid choice!")
        return False

def quick_test():
    """Quick test to verify basic functionality"""
    print("\n Running Quick Functionality Test...")
    print("-"*60)
    
    try:
        import numpy as np
        import pandas as pd
        
        # Test data creation
        print("Creating test data...")
        data = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 100),
            'high': np.random.uniform(45000, 46000, 100),
            'low': np.random.uniform(39000, 40000, 100),
            'close': np.random.uniform(40000, 45000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        print(" Test data created")
        
        # Test environment
        print("\nTesting Trading Environment...")
        from src.environment.trading_env import TradingEnvironment
        env = TradingEnvironment(df=data, initial_balance=10000)
        obs = env.reset()
        
        for i in range(10):
            action = np.random.choice([0, 1, 2])
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        
        print(f" Environment working - {i+1} steps completed")
        print(f"   Final balance: ${env.balance:.2f}")
        print(f"   Portfolio value: ${env._get_portfolio_value():.2f}")
        
        # Test reward calculator
        print("\nTesting Reward Functions...")
        from src.environment.rewards import RewardCalculator
        calc = RewardCalculator()
        reward = calc.simple_returns(10500, 10000)
        print(f" Rewards working - Simple return: {reward:.4f}")
        
        # Test state manager
        print("\nTesting State Management...")
        from src.environment.state import StateManager
        manager = StateManager()
        market_state = manager.extract_market_state(data, 50)
        print(f" State management working")
        print(f"   Current price: ${market_state.current_price:.2f}")
        
        print("\n" + "="*60)
        print(" ALL MODULES ARE WORKING!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    print("\n" + "="*60)
    print("   TRADING ENVIRONMENT TEST SYSTEM")
    print("="*60)
    
    # First run quick test
    print("\nRunning quick functionality check...")
    if not quick_test():
        print("\n  Quick test failed! There may be import or setup issues.")
        print("Please check:")
        print("  1. All module files are in the correct directories")
        print("  2. Python path is set correctly")
        print("  3. All required packages are installed")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Interactive menu
    while True:
        print_menu()
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == '0':
            print("\nExiting test runner...")
            break
            
        elif choice in ['1', '2', '3', '4', '5', '6', '7']:
            success = run_test_suite(choice)
            
            if success:
                print("\n Tests PASSED!")
            else:
                print("\n Tests FAILED!")
            
            input("\nPress Enter to continue...")
        else:
            print("\n  Invalid choice! Please enter 0-7")
    
    print("\nThank you for testing!")
    return True

if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h']:
            print("\nUsage: python run_tests.py [option]")
            print("\nOptions:")
            print("  --quick     Run quick functionality test")
            print("  --all       Run all tests")
            print("  --env       Test Trading Environment only")
            print("  --rewards   Test Reward Functions only")
            print("  --state     Test State Management only")
            print("  --help      Show this help message")
            sys.exit(0)
            
        elif arg == '--quick':
            success = quick_test()
            sys.exit(0 if success else 1)
            
        elif arg == '--all':
            from tests.test_all import main as run_all
            success = run_all()
            sys.exit(0 if success else 1)
            
        elif arg == '--env':
            from tests.test_trading_env import run_all_tests
            success = run_all_tests()
            sys.exit(0 if success else 1)
            
        elif arg == '--rewards':
            from tests.test_rewards import run_all_tests
            success = run_all_tests()
            sys.exit(0 if success else 1)
            
        elif arg == '--state':
            from tests.test_state import run_all_tests
            success = run_all_tests()
            sys.exit(0 if success else 1)
        else:
            print(f"Unknown option: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
    else:
        # Run interactive menu
        main()