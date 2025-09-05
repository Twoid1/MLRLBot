"""
Test Script for Order Executor Module
Comprehensive testing of order execution functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import the module to test
from src.trading.executor import (
    OrderExecutor, Order, OrderType, OrderSide, OrderStatus,
    TimeInForce, ExecutionResult, MarketConditions
)


class TestOrderExecutor(unittest.TestCase):
    """Test suite for OrderExecutor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.executor = OrderExecutor(
            mode='simulation',
            commission_rate=0.0026,
            slippage_model='linear',
            max_slippage=0.01
        )
        
        # Set up market conditions
        self.market_conditions = MarketConditions(
            bid=50000,
            ask=50010,
            last_price=50005,
            spread=10,
            volume=1000000,
            volatility=0.02,
            liquidity=0.9
        )
        self.executor.update_market_conditions('BTC/USDT', self.market_conditions)
    
    def test_initialization(self):
        """Test OrderExecutor initialization"""
        self.assertEqual(self.executor.mode, 'simulation')
        self.assertEqual(self.executor.commission_rate, 0.0026)
        self.assertEqual(self.executor.slippage_model, 'linear')
        self.assertEqual(len(self.executor.orders), 0)
        print(" Initialization test passed")
    
    def test_create_order(self):
        """Test order creation"""
        order = self.executor.create_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=0.1,
            order_type='market'
        )
        
        self.assertIsInstance(order, Order)
        self.assertEqual(order.symbol, 'BTC/USDT')
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, 0.1)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.status, OrderStatus.PENDING)
        
        print(" Order creation test passed")
        print(f"   Order ID: {order.order_id}")
    
    def test_market_order_execution(self):
        """Test market order execution"""
        order = self.executor.create_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=0.1,
            order_type='market'
        )
        
        result = self.executor.submit_order(order)
        
        self.assertTrue(result.success)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertGreater(result.execution_price, 0)
        self.assertEqual(result.actual_quantity, 0.1)
        self.assertGreater(result.commission, 0)
        
        # Check that execution price is at ask for buy order
        self.assertGreaterEqual(result.execution_price, self.market_conditions.ask)
        
        print(" Market order execution test passed")
        print(f"   Executed at: ${result.execution_price:.2f}")
        print(f"   Commission: ${result.commission:.2f}")
        print(f"   Slippage: {result.slippage:.4%}")
    
    def test_limit_order_execution(self):
        """Test limit order execution"""
        # Create limit order above current ask (won't fill immediately)
        order = self.executor.create_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=0.1,
            order_type='limit',
            price=49900  # Below current market
        )
        
        result = self.executor.submit_order(order)
        
        # For simulation, this should fill if price is favorable
        if self.market_conditions.ask <= 49900:
            self.assertTrue(result.success)
            self.assertEqual(order.status, OrderStatus.FILLED)
        else:
            self.assertEqual(order.status, OrderStatus.PENDING)
        
        print(" Limit order execution test passed")
        print(f"   Order status: {order.status.value}")
    
    def test_stop_order_execution(self):
        """Test stop order execution"""
        # Create stop order
        order = self.executor.create_order(
            symbol='BTC/USDT',
            side='sell',
            quantity=0.1,
            order_type='stop',
            stop_price=49500  # Stop loss below current price
        )
        
        result = self.executor.submit_order(order)
        
        # In simulation, stop won't trigger if price hasn't reached stop level
        self.assertEqual(order.status, OrderStatus.PENDING)
        
        # Update market to trigger stop
        self.executor.update_market_conditions('BTC/USDT', MarketConditions(
            bid=49400, ask=49410, last_price=49405,
            spread=10, volume=1000000, volatility=0.02, liquidity=0.9
        ))
        
        # Re-submit should now execute
        result = self.executor.submit_order(order)
        
        print(" Stop order execution test passed")
        print(f"   Stop status: {order.status.value}")
    
    def test_slippage_calculation(self):
        """Test slippage calculation"""
        # Large order should have more slippage
        large_order = self.executor.create_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=10,  # Large order
            order_type='market'
        )
        
        small_order = self.executor.create_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=0.01,  # Small order
            order_type='market'
        )
        
        large_result = self.executor.submit_order(large_order)
        small_result = self.executor.submit_order(small_order)
        
        # Large order should have more slippage
        self.assertGreater(large_result.slippage, small_result.slippage)
        
        print(" Slippage calculation test passed")
        print(f"   Large order slippage: {large_result.slippage:.4%}")
        print(f"   Small order slippage: {small_result.slippage:.4%}")
    
    def test_commission_calculation(self):
        """Test commission calculation"""
        order = self.executor.create_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=0.1,
            order_type='market'
        )
        
        result = self.executor.submit_order(order)
        
        expected_commission = 0.1 * result.execution_price * 0.0026
        self.assertAlmostEqual(result.commission, expected_commission, places=2)
        
        print(" Commission calculation test passed")
        print(f"   Commission: ${result.commission:.2f}")
    
    def test_twap_execution(self):
        """Test TWAP execution algorithm"""
        results = self.executor.execute_twap(
            symbol='BTC/USDT',
            side='buy',
            total_quantity=1.0,
            duration_minutes=5,
            intervals=5
        )
        
        self.assertEqual(len(results), 5)
        
        # Check that each slice is approximately equal
        expected_slice = 1.0 / 5
        for result in results:
            self.assertAlmostEqual(result.actual_quantity, expected_slice, places=4)
        
        # Calculate average execution price
        total_qty = sum(r.actual_quantity for r in results)
        avg_price = sum(r.execution_price * r.actual_quantity for r in results) / total_qty
        
        print(" TWAP execution test passed")
        print(f"   Executed in {len(results)} slices")
        print(f"   Average price: ${avg_price:.2f}")
    
    def test_vwap_execution(self):
        """Test VWAP execution algorithm"""
        # Volume profile (e.g., U-shaped for typical trading day)
        volume_profile = [30, 20, 15, 10, 10, 15, 20, 30]  # Higher at open/close
        
        results = self.executor.execute_vwap(
            symbol='BTC/USDT',
            side='buy',
            total_quantity=1.0,
            volume_profile=volume_profile
        )
        
        self.assertEqual(len(results), len(volume_profile))
        
        # Check that size distribution matches volume profile
        total_volume = sum(volume_profile)
        for i, result in enumerate(results):
            expected_size = 1.0 * (volume_profile[i] / total_volume)
            self.assertAlmostEqual(result.actual_quantity, expected_size, places=4)
        
        print(" VWAP execution test passed")
        print(f"   Executed following volume profile")
    
    def test_iceberg_execution(self):
        """Test iceberg order execution"""
        results = self.executor.execute_iceberg(
            symbol='BTC/USDT',
            side='buy',
            total_quantity=2.0,
            visible_quantity=0.5
        )
        
        # Should be split into multiple slices
        self.assertGreater(len(results), 1)
        
        # Each slice (except possibly the last) should be visible_quantity
        for result in results[:-1]:
            self.assertLessEqual(result.actual_quantity, 0.5)
        
        # Total should equal requested
        total_executed = sum(r.actual_quantity for r in results)
        self.assertAlmostEqual(total_executed, 2.0, places=4)
        
        print(" Iceberg execution test passed")
        print(f"   Split into {len(results)} visible slices")
    
    def test_order_cancellation(self):
        """Test order cancellation"""
        order = self.executor.create_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=0.1,
            order_type='limit',
            price=45000  # Far from market, won't execute
        )
        
        # Submit order
        self.executor.submit_order(order)
        
        # Cancel order
        success = self.executor.cancel_order(order.order_id)
        
        self.assertTrue(success)
        self.assertEqual(order.status, OrderStatus.CANCELLED)
        self.assertNotIn(order.order_id, self.executor.active_orders)
        
        print(" Order cancellation test passed")
    
    def test_cancel_all_orders(self):
        """Test cancelling all orders"""
        # Create multiple orders
        for i in range(5):
            order = self.executor.create_order(
                symbol='BTC/USDT',
                side='buy',
                quantity=0.1,
                order_type='limit',
                price=45000 + i * 100
            )
            self.executor.submit_order(order)
        
        # Cancel all
        cancelled = self.executor.cancel_all_orders()
        
        self.assertEqual(cancelled, 5)
        self.assertEqual(len(self.executor.active_orders), 0)
        
        print(f" Cancel all orders test passed ({cancelled} orders cancelled)")
    
    def test_order_modification(self):
        """Test order modification"""
        order = self.executor.create_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=0.1,
            order_type='limit',
            price=49000
        )
        
        self.executor.submit_order(order)
        
        # Modify order
        success = self.executor.modify_order(
            order.order_id,
            new_quantity=0.2,
            new_price=49500
        )
        
        self.assertTrue(success)
        self.assertEqual(order.quantity, 0.2)
        self.assertEqual(order.price, 49500)
        
        print(" Order modification test passed")
    
    def test_order_tracking(self):
        """Test order tracking functionality"""
        # Create and submit orders
        orders = []
        for i in range(3):
            order = self.executor.create_order(
                symbol='BTC/USDT',
                side='buy',
                quantity=0.1,
                order_type='market'
            )
            self.executor.submit_order(order)
            orders.append(order)
        
        # Check order history
        history = self.executor.get_order_history('BTC/USDT')
        self.assertEqual(len(history), 3)
        
        # Check fills
        fills = self.executor.get_fills('BTC/USDT')
        self.assertEqual(len(fills), 3)
        
        print(" Order tracking test passed")
        print(f"   Orders in history: {len(history)}")
        print(f"   Fills recorded: {len(fills)}")
    
    def test_execution_metrics(self):
        """Test execution metrics calculation"""
        # Execute some orders
        for i in range(10):
            order = self.executor.create_order(
                symbol='BTC/USDT',
                side='buy' if i % 2 == 0 else 'sell',
                quantity=0.1,
                order_type='market'
            )
            self.executor.submit_order(order)
        
        metrics = self.executor.get_execution_metrics()
        
        self.assertIn('total_orders', metrics)
        self.assertIn('success_rate', metrics)
        self.assertIn('avg_slippage', metrics)
        self.assertIn('total_commission', metrics)
        
        print(" Execution metrics test passed")
        print(f"   Success rate: {metrics['success_rate']:.2%}")
        print(f"   Avg slippage: {metrics['avg_slippage']:.4%}")
        print(f"   Total commission: ${metrics['total_commission']:.2f}")
    
    def test_callbacks(self):
        """Test callback functionality"""
        fill_called = []
        
        def on_fill(order, result):
            fill_called.append((order.order_id, result.success))
        
        self.executor.add_fill_callback(on_fill)
        
        # Execute order
        order = self.executor.create_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=0.1,
            order_type='market'
        )
        self.executor.submit_order(order)
        
        # Check callback was called
        self.assertEqual(len(fill_called), 1)
        self.assertEqual(fill_called[0][0], order.order_id)
        self.assertTrue(fill_called[0][1])
        
        print(" Callbacks test passed")


def run_execution_simulation():
    """Run realistic execution simulation"""
    print("\n" + "="*60)
    print("EXECUTION SIMULATION")
    print("="*60)
    
    executor = OrderExecutor(mode='simulation')
    
    # Set up varying market conditions
    market_scenarios = [
        ('Normal', MarketConditions(50000, 50010, 50005, 10, 1000000, 0.02, 0.9)),
        ('Wide Spread', MarketConditions(50000, 50050, 50025, 50, 800000, 0.025, 0.8)),
        ('High Volatility', MarketConditions(49900, 50100, 50000, 200, 1500000, 0.05, 0.7)),
        ('Low Liquidity', MarketConditions(50000, 50020, 50010, 20, 500000, 0.02, 0.5))
    ]
    
    print("\n Testing different market conditions...")
    
    for scenario_name, conditions in market_scenarios:
        print(f"\n{scenario_name}:")
        executor.update_market_conditions('BTC/USDT', conditions)
        
        # Execute a standard order
        order = executor.create_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=0.5,
            order_type='market'
        )
        
        result = executor.submit_order(order)
        
        print(f"   Spread: ${conditions.spread:.2f}")
        print(f"   Execution: ${result.execution_price:.2f}")
        print(f"   Slippage: {result.slippage:.4%}")
        print(f"   Commission: ${result.commission:.2f}")
        print(f"   Total cost: ${result.execution_price * 0.5 + result.commission:.2f}")


def run_algo_comparison():
    """Compare different execution algorithms"""
    print("\n" + "="*60)
    print("EXECUTION ALGORITHM COMPARISON")
    print("="*60)
    
    executor = OrderExecutor(mode='simulation')
    
    # Standard market conditions
    executor.update_market_conditions('BTC/USDT', MarketConditions(
        50000, 50010, 50005, 10, 1000000, 0.02, 0.9
    ))
    
    total_quantity = 5.0
    results_by_algo = {}
    
    print("\n Comparing execution algorithms...")
    
    # 1. Single Market Order
    print("\n1. Single Market Order:")
    single_order = executor.create_order(
        'BTC/USDT', 'buy', total_quantity, 'market'
    )
    single_result = executor.submit_order(single_order)
    results_by_algo['Single'] = {
        'avg_price': single_result.execution_price,
        'total_cost': single_result.execution_price * total_quantity + single_result.commission,
        'slippage': single_result.slippage
    }
    
    # 2. TWAP
    print("\n2. TWAP (10 slices):")
    twap_results = executor.execute_twap(
        'BTC/USDT', 'buy', total_quantity, 
        duration_minutes=10, intervals=10
    )
    total_qty_twap = sum(r.actual_quantity for r in twap_results)
    avg_price_twap = sum(r.execution_price * r.actual_quantity for r in twap_results) / total_qty_twap
    total_commission_twap = sum(r.commission for r in twap_results)
    
    results_by_algo['TWAP'] = {
        'avg_price': avg_price_twap,
        'total_cost': avg_price_twap * total_quantity + total_commission_twap,
        'slippage': np.mean([r.slippage for r in twap_results])
    }
    
    # 3. Iceberg
    print("\n3. Iceberg (1.0 visible):")
    iceberg_results = executor.execute_iceberg(
        'BTC/USDT', 'buy', total_quantity, 
        visible_quantity=1.0
    )
    total_qty_ice = sum(r.actual_quantity for r in iceberg_results)
    avg_price_ice = sum(r.execution_price * r.actual_quantity for r in iceberg_results) / total_qty_ice
    total_commission_ice = sum(r.commission for r in iceberg_results)
    
    results_by_algo['Iceberg'] = {
        'avg_price': avg_price_ice,
        'total_cost': avg_price_ice * total_quantity + total_commission_ice,
        'slippage': np.mean([r.slippage for r in iceberg_results])
    }
    
    # Compare results
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*60)
    print(f"\n{'Algorithm':<15} {'Avg Price':<12} {'Total Cost':<12} {'Avg Slippage':<12}")
    print("-" * 60)
    
    for algo, metrics in results_by_algo.items():
        print(f"{algo:<15} ${metrics['avg_price']:<11.2f} "
              f"${metrics['total_cost']:<11.2f} {metrics['slippage']:<11.4%}")
    
    # Find best algorithm
    best_algo = min(results_by_algo.items(), key=lambda x: x[1]['total_cost'])
    print(f"\n Best algorithm: {best_algo[0]} (lowest total cost)")


def run_stress_test():
    """Stress test the executor with many orders"""
    print("\n" + "="*60)
    print("EXECUTOR STRESS TEST")
    print("="*60)
    
    executor = OrderExecutor(mode='simulation')
    
    # Set market conditions
    executor.update_market_conditions('BTC/USDT', MarketConditions(
        50000, 50010, 50005, 10, 1000000, 0.02, 0.9
    ))
    
    print("\n Executing 100 orders...")
    
    start_time = time.time()
    
    for i in range(100):
        order = executor.create_order(
            symbol='BTC/USDT',
            side='buy' if i % 2 == 0 else 'sell',
            quantity=np.random.uniform(0.01, 0.5),
            order_type='market'
        )
        executor.submit_order(order)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Get metrics
    metrics = executor.get_execution_metrics()
    
    print(f"\n Stress test completed!")
    print(f"   Orders executed: {metrics['total_orders']}")
    print(f"   Success rate: {metrics['success_rate']:.2%}")
    print(f"   Execution time: {execution_time:.3f} seconds")
    print(f"   Orders per second: {100/execution_time:.1f}")
    print(f"   Total commission: ${metrics['total_commission']:.2f}")
    print(f"   Total slippage cost: ${metrics['total_slippage']:.2f}")


if __name__ == "__main__":
    # Run unit tests
    print("Running Order Executor Unit Tests...")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOrderExecutor)
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("UNIT TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n All unit tests passed!")
    else:
        print("\n Some tests failed. Check output above.")
    
    # Run additional tests
    run_execution_simulation()
    run_algo_comparison()
    run_stress_test()
    
    print("\n Order Executor testing complete!")