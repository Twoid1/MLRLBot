"""
Integration Test Script - FIXED VERSION
Tests all four trading modules working together
Simulates realistic trading scenarios
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from src.trading.risk_manager import RiskManager, RiskConfig, RiskLevel
from src.trading.position_sizer import PositionSizer, MarketConditions
from src.trading.portfolio import Portfolio
from src.trading.executor import OrderExecutor, OrderType, OrderSide, MarketConditions as ExecutorMarketConditions


class IntegrationTest:
    """Complete integration testing of all trading modules"""
    
    def __init__(self, initial_capital: float = 10000):
        """Initialize all components"""
        self.initial_capital = initial_capital
        
        # Initialize all modules with test-friendly settings
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            config=RiskConfig(
                risk_level=RiskLevel.MODERATE,
                max_risk_per_trade=0.02,
                position_sizing_method='volatility_based',
                min_time_between_trades=0,  # FIXED: Set to 0 for testing
                daily_trades_limit=100,  # FIXED: Increased for testing
                hourly_trades_limit=50   # FIXED: Increased for testing
            )
        )
        
        self.position_sizer = PositionSizer(
            capital=initial_capital,
            max_risk_per_trade=0.02,
            max_portfolio_risk=0.06
        )
        
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            max_positions=5,
            enable_short=False
        )
        
        self.executor = OrderExecutor(
            mode='simulation',
            commission_rate=0.0026,
            slippage_model='linear'
        )
        
        print(f" Integration test system initialized with ${initial_capital:,.2f}")
    
    def run_complete_trade_cycle(self, symbol: str = None, 
                                price: float = None, 
                                volatility: float = None):
        """Test complete trade cycle from signal to execution"""
        print("\n" + "="*60)
        print("COMPLETE TRADE CYCLE TEST")
        print("="*60)
        
        # Use provided values or defaults
        if not symbol:
            symbol = 'BTC/USDT'
        if not price:
            price = 50000
        if not volatility:
            volatility = 0.02
        
        # Step 1: Risk Check
        print("\n RISK CHECK")
        can_trade = self.risk_manager.can_open_position(symbol)
        print(f"   Can open position: {can_trade}")
        
        if not can_trade:
            # Check why we can't trade
            if symbol in self.risk_manager.open_positions:
                print(f"    Position already exists for {symbol}")
            elif len(self.risk_manager.open_positions) >= self.risk_manager.position_limits.max_positions:
                print(f"    Maximum positions reached")
            elif self.risk_manager.consecutive_losses >= self.risk_manager.config.consecutive_losses_limit:
                print(f"    Too many consecutive losses")
            else:
                print(f"    Other risk limits triggered")
            return None
        
        # Step 2: Position Sizing
        print("\n POSITION SIZING")
        
        # Update capital in position sizer to match risk manager
        self.position_sizer.update_capital(self.risk_manager.current_capital)
        
        # Method 1: Risk Manager sizing
        rm_size = self.risk_manager.calculate_position_size(
            symbol=symbol,
            signal_strength=0.8,
            current_price=price,
            volatility=volatility,
            win_rate=0.55,
            avg_win=300,
            avg_loss=200
        )
        
        # Method 2: Kelly Criterion
        kelly_result = self.position_sizer.kelly_criterion_basic(
            win_probability=0.55,
            win_amount=300,
            loss_amount=200,
            kelly_fraction=0.25
        )
        
        # Choose the smaller size for safety
        position_value = min(
            rm_size * price,
            kelly_result.position_value,
            self.risk_manager.current_capital * 0.02  # Max 2% risk
        )
        
        final_size = position_value / price
        
        print(f"   Risk Manager size: {rm_size:.6f} units")
        print(f"   Kelly size: {kelly_result.position_value/price:.6f} units")
        print(f"   Final size: {final_size:.6f} units")
        print(f"   Position value: ${position_value:.2f}")
        
        # Step 3: Calculate Stops
        print("\n STOP LOSS & TAKE PROFIT")
        atr = price * volatility  # Simplified ATR
        
        stop_loss = self.risk_manager.calculate_stop_loss(
            price, 'long', atr
        )
        
        take_profit = self.risk_manager.calculate_take_profit(
            price, 'long', atr
        )
        
        print(f"   Entry: ${price:,.2f}")
        print(f"   Stop Loss: ${stop_loss:,.2f} ({((stop_loss/price)-1)*100:.2f}%)")
        print(f"   Take Profit: ${take_profit:,.2f} ({((take_profit/price)-1)*100:.2f}%)")
        
        # Step 4: Order Execution
        print("\n ORDER EXECUTION")
        
        # Set market conditions
        market = ExecutorMarketConditions(
            bid=price - 5,
            ask=price + 5,
            last_price=price,
            spread=10,
            volume=1000000,
            volatility=volatility,
            liquidity=0.9
        )
        self.executor.update_market_conditions(symbol, market)
        
        # Create order
        order = self.executor.create_order(
            symbol=symbol,
            side='buy',
            quantity=final_size,
            order_type='market',
            metadata={
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        )
        
        # Execute order
        exec_result = self.executor.submit_order(order)
        
        if exec_result.success:
            print(f"    Order executed")
            print(f"   Execution price: ${exec_result.execution_price:,.2f}")
            print(f"   Slippage: {exec_result.slippage:.4%}")
            print(f"   Commission: ${exec_result.commission:.2f}")
        else:
            print(f"    Order failed: {exec_result.message}")
            return None
        
        # Step 5: Portfolio Update
        print("\n PORTFOLIO UPDATE")
        
        portfolio_success = self.portfolio.open_position(
            symbol=symbol,
            quantity=exec_result.actual_quantity,
            price=exec_result.execution_price,
            position_type='long',
            stop_loss=stop_loss,
            take_profit=take_profit,
            fees=exec_result.commission
        )
        
        if portfolio_success:
            print(f"    Position added to portfolio")
        else:
            print(f"    Failed to add to portfolio")
        
        # Step 6: Risk Manager Update
        print("\n RISK TRACKING UPDATE")
        
        self.risk_manager.add_position(
            symbol=symbol,
            position_size=exec_result.actual_quantity,
            entry_price=exec_result.execution_price,
            position_type='long',
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        print(f"    Position added to risk tracking")
        
        # Update capital after trade
        trade_cost = exec_result.actual_quantity * exec_result.execution_price + exec_result.commission
        new_capital = self.risk_manager.current_capital - trade_cost
        self.risk_manager.update_capital(new_capital)
        
        # Get current status
        self.print_system_status()
        
        return exec_result
    
    def simulate_price_movement(self, symbol: str, price_change: float):
        """Simulate price movement and update all systems"""
        print(f"\n Simulating {price_change:.2%} price change for {symbol}")
        
        if symbol not in self.portfolio.positions:
            print(f"    No position in {symbol}")
            return
        
        position = self.portfolio.positions[symbol]
        new_price = position.current_price * (1 + price_change)
        
        # Update portfolio
        self.portfolio.update_position_price(symbol, new_price)
        
        # Check risk manager recommendations
        recommendations = self.risk_manager.update_position(
            symbol, new_price
        )
        
        print(f"   New price: ${new_price:,.2f}")
        print(f"   Unrealized P&L: ${position.unrealized_pnl:.2f}")
        print(f"   Recommendations: {recommendations['action']}")
        
        # Handle stop/take profit
        if recommendations['action'] == 'close':
            self.close_position(symbol, new_price, recommendations['reason'])
    
    def close_position(self, symbol: str, price: float, reason: str):
        """Close position across all systems"""
        print(f"\n Closing {symbol} position (reason: {reason})")
        
        # Close in portfolio
        portfolio_result = self.portfolio.close_position(
            symbol, price, fees=10, reason=reason
        )
        
        if portfolio_result:
            print(f"   P&L: ${portfolio_result['net_pnl']:.2f}")
            print(f"   Return: {portfolio_result['return_pct']:.2%}")
            
            # Update risk manager
            rm_result = self.risk_manager.remove_position(symbol, price)
            
            # Update capital
            new_capital = self.risk_manager.current_capital + portfolio_result['net_pnl']
            self.risk_manager.update_capital(new_capital)
            self.position_sizer.update_capital(new_capital)
    
    def print_system_status(self):
        """Print status of all systems"""
        print("\n" + "="*60)
        print(" SYSTEM STATUS")
        print("="*60)
        
        # Portfolio status
        portfolio_summary = self.portfolio.get_summary()
        print("\n PORTFOLIO:")
        print(f"   Value: ${portfolio_summary['value']['total']:,.2f}")
        print(f"   Cash: ${portfolio_summary['value']['cash']:,.2f}")
        print(f"   Positions: {portfolio_summary['positions']['count']}")
        print(f"   Unrealized P&L: ${portfolio_summary['pnl']['unrealized']:.2f}")
        print(f"   Realized P&L: ${portfolio_summary['pnl']['realized']:.2f}")
        
        # Risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        print("\n RISK:")
        print(f"   Current Drawdown: {risk_metrics.current_drawdown:.2%}")
        print(f"   Max Drawdown: {risk_metrics.max_drawdown:.2%}")
        print(f"   Win Rate: {risk_metrics.win_rate:.2%}")
        print(f"   Consecutive Losses: {self.risk_manager.consecutive_losses}")
        
        # Executor metrics
        exec_metrics = self.executor.get_execution_metrics()
        print("\n EXECUTION:")
        print(f"   Total Orders: {exec_metrics.get('total_orders', 0)}")
        print(f"   Success Rate: {exec_metrics.get('success_rate', 0):.2%}")
        print(f"   Total Commission: ${exec_metrics.get('total_commission', 0):.2f}")


def run_full_trading_session():
    """Run a complete trading session simulation"""
    print("\n" + "="*70)
    print(" FULL TRADING SESSION SIMULATION")
    print("="*70)
    
    # Initialize system
    system = IntegrationTest(initial_capital=10000)
    
    # Trading scenarios - different symbols to avoid conflicts
    trades = [
        {'symbol': 'BTC/USDT', 'price': 45000, 'volatility': 0.02, 'outcome': 'win', 'price_change': 0.03},
        {'symbol': 'ETH/USDT', 'price': 3000, 'volatility': 0.025, 'outcome': 'loss', 'price_change': -0.02},
        {'symbol': 'SOL/USDT', 'price': 100, 'volatility': 0.03, 'outcome': 'win', 'price_change': 0.05},
    ]
    
    print("\n Starting trading session...")
    
    for i, trade in enumerate(trades):
        print(f"\n" + "="*60)
        print(f" TRADE {i+1}: {trade['symbol']}")
        print("="*60)
        
        # Execute trade
        result = system.run_complete_trade_cycle(
            symbol=trade['symbol'],
            price=trade['price'],
            volatility=trade['volatility']
        )
        
        if result:
            # Wait a bit to simulate real trading
            time.sleep(0.1)
            
            # Simulate price movement
            system.simulate_price_movement(
                trade['symbol'],
                trade['price_change']
            )
    
    # Final summary
    print("\n" + "="*70)
    print(" SESSION SUMMARY")
    print("="*70)
    
    system.print_system_status()
    
    # Performance analysis
    portfolio_metrics = system.portfolio.calculate_performance_metrics()
    
    print("\n PERFORMANCE:")
    print(f"   Total Return: {portfolio_metrics.total_return:.2%}")
    print(f"   Sharpe Ratio: {portfolio_metrics.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {portfolio_metrics.sortino_ratio:.2f}")
    print(f"   Win Rate: {portfolio_metrics.win_rate:.2%}")
    print(f"   Profit Factor: {portfolio_metrics.profit_factor:.2f}")


def test_risk_cascade():
    """Test risk management cascade through all systems"""
    print("\n" + "="*70)
    print(" RISK CASCADE TEST")
    print("="*70)
    
    system = IntegrationTest(initial_capital=10000)
    
    print("\n Simulating consecutive losses to trigger risk limits...")
    
    # Simulate losing trades
    for i in range(5):
        print(f"\n Loss #{i+1}")
        
        # Record a loss in risk manager
        system.risk_manager.record_trade(
            symbol=f'TEST{i}/USDT',  # Different symbols
            pnl=-100,
            position_size=0.1,
            entry_price=50000,
            exit_price=49000,
            position_type='long'
        )
        
        # Update capital
        new_capital = system.risk_manager.current_capital - 100
        system.risk_manager.update_capital(new_capital)
        system.portfolio.cash_balance = new_capital
        system.position_sizer.update_capital(new_capital)
        
        # Check if we can still trade
        can_trade = system.risk_manager.can_open_position('BTC/USDT')
        print(f"   Can still trade: {can_trade}")
        print(f"   Consecutive losses: {system.risk_manager.consecutive_losses}")
        print(f"   Capital: ${system.risk_manager.current_capital:,.2f}")
        print(f"   Circuit breaker: {' ACTIVE' if system.risk_manager.circuit_breaker_active else ' inactive'}")
        
        if not can_trade:
            print("\n Risk limits triggered - trading suspended!")
            break


def test_multi_asset_execution():
    """Test multi-asset portfolio execution"""
    print("\n" + "="*70)
    print(" MULTI-ASSET EXECUTION TEST")
    print("="*70)
    
    system = IntegrationTest(initial_capital=50000)
    
    # Define multiple assets
    assets = [
        {'symbol': 'BTC/USDT', 'price': 45000, 'volatility': 0.02, 'allocation': 0.3},
        {'symbol': 'ETH/USDT', 'price': 3000, 'volatility': 0.025, 'allocation': 0.25},
        {'symbol': 'SOL/USDT', 'price': 100, 'volatility': 0.03, 'allocation': 0.15},
        {'symbol': 'ADA/USDT', 'price': 1.2, 'volatility': 0.028, 'allocation': 0.1},
        {'symbol': 'DOT/USDT', 'price': 30, 'volatility': 0.027, 'allocation': 0.1}
    ]
    
    print("\n Building multi-asset portfolio...")
    
    for asset in assets:
        print(f"\n {asset['symbol']}:")
        
        # Calculate position size based on allocation
        position_value = system.portfolio.cash_balance * asset['allocation'] * 0.9  # Keep 10% cash
        position_size = position_value / asset['price']
        
        # Set market conditions
        system.executor.update_market_conditions(
            asset['symbol'],
            ExecutorMarketConditions(
                bid=asset['price'] - 5,
                ask=asset['price'] + 5,
                last_price=asset['price'],
                spread=10,
                volume=1000000,
                volatility=asset['volatility'],
                liquidity=0.9
            )
        )
        
        # Create and execute order
        order = system.executor.create_order(
            symbol=asset['symbol'],
            side='buy',
            quantity=position_size,
            order_type='market'
        )
        
        result = system.executor.submit_order(order)
        
        if result.success:
            # Add to portfolio
            portfolio_success = system.portfolio.open_position(
                symbol=asset['symbol'],
                quantity=result.actual_quantity,
                price=result.execution_price,
                position_type='long',
                fees=result.commission
            )
            
            if portfolio_success:
                # Add to risk manager
                system.risk_manager.add_position(
                    symbol=asset['symbol'],
                    position_size=result.actual_quantity,
                    entry_price=result.execution_price,
                    position_type='long'
                )
                
                print(f"    Position opened: {result.actual_quantity:.4f} @ ${result.execution_price:.2f}")
            else:
                print(f"    Failed to add to portfolio")
        else:
            print(f"    Order failed: {result.message}")
    
    # Show final allocation
    allocation = system.portfolio.get_position_allocation()
    
    print("\n FINAL ALLOCATION:")
    for symbol, weight in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
        bar = '|' * int(weight * 50)
        print(f"   {symbol:<10} {bar:<25} {weight:>6.2%}")
    
    # Calculate portfolio metrics
    exposure = system.portfolio.get_exposure_summary()
    
    print("\n PORTFOLIO METRICS:")
    print(f"   Total Exposure: ${exposure['total_exposure']:,.2f}")
    print(f"   Number of Positions: {exposure['n_positions']}")
    print(f"   Gross Leverage: {exposure['gross_leverage']:.2f}x")
    print(f"   Concentration Risk: {exposure['concentration']:.2%}")


def test_stop_loss_trigger():
    """Test stop loss and take profit triggers"""
    print("\n" + "="*70)
    print(" STOP LOSS & TAKE PROFIT TEST")
    print("="*70)
    
    system = IntegrationTest(initial_capital=10000)
    
    # Open a position
    print("\n Opening test position...")
    result = system.run_complete_trade_cycle(
        symbol='BTC/USDT',
        price=50000,
        volatility=0.02
    )
    
    if result:
        # Simulate price drop to trigger stop loss
        print("\n Simulating price drop to trigger stop loss...")
        system.simulate_price_movement('BTC/USDT', -0.05)
        
        # Reset and open another position
        system = IntegrationTest(initial_capital=10000)
        print("\n Opening another test position...")
        result = system.run_complete_trade_cycle(
            symbol='ETH/USDT',
            price=3000,
            volatility=0.025
        )
        
        if result:
            # Simulate price rise to trigger take profit
            print("\n Simulating price rise to trigger take profit...")
            system.simulate_price_movement('ETH/USDT', 0.07)


def main():
    """Main test runner"""
    print("="*70)
    print(" INTEGRATION TEST SUITE")
    print("Running comprehensive tests of all trading modules")
    print("="*70)
    
    tests = [
        ("Complete Trade Cycle", run_full_trading_session),
        ("Risk Cascade", test_risk_cascade),
        ("Multi-Asset Execution", test_multi_asset_execution),
        ("Stop Loss & Take Profit", test_stop_loss_trigger)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n\n{'='*70}")
            print(f" Running: {test_name}")
            print('='*70)
            test_func()
            results.append((test_name, " PASSED"))
            print(f"\n {test_name} completed successfully")
        except Exception as e:
            results.append((test_name, f" FAILED: {str(e)}"))
            print(f"\n Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        print(f"{test_name:<30} {result}")
    
    # Overall result
    passed = sum(1 for _, r in results if "PASSED" in r)
    total = len(results)
    
    print("\n" + "="*70)
    if passed == total:
        print(f" ALL TESTS PASSED! ({passed}/{total})")
    else:
        print(f" Some tests failed: {passed}/{total} passed")
    print("="*70)


if __name__ == "__main__":
    main()