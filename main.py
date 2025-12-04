#!/usr/bin/env python3
"""
Kraken Trading Bot - Main Entry Point (OPTIMIZED + TRADE-BASED + MULTI-OBJECTIVE)

Added:
- Fast training with GPU support
- Real-time progress monitoring
- Automatic optimization
- Performance benchmarking
- ⭐ Trade-based training (1 episode = 1 trade)
- ⭐ Multi-objective rewards (5 separate learning signals)
"""

import argparse
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import subprocess
import os

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Set up logger
logger = None


def handle_train_command_trade_based(args):
    """
    Handle TRADE-BASED training commands
    
    ⭐ Each episode = 1 complete trade
    ⭐ Optional: Multi-objective rewards (5 separate signals)
    
    Agent learns trade QUALITY, not step-by-step prediction
    """
    from src.trade_based_trainer import TradeBasedTrainer, train_trade_based, train_trade_based_rl_only
    
    # Check if multi-objective mode requested
    use_mo = hasattr(args, 'multi_objective') and args.multi_objective
    
    logger.info("="*80)
    logger.info("TRADE-BASED TRAINING SYSTEM")
    logger.info("="*80)
    logger.info("  Each episode = 1 complete trade")
    logger.info("  Agent learns: WHEN to enter, WHEN to exit")
    logger.info("  Focus: Trade QUALITY over QUANTITY")
    
    if use_mo:
        logger.info("")
        logger.info("   MULTI-OBJECTIVE MODE ENABLED ")
        logger.info("  5 Separate Reward Signals:")
        logger.info("    1. pnl_quality   - Maximize wins, minimize losses")
        logger.info("    2. hold_duration - Hold trades longer")
        logger.info("    3. win_achieved  - Win more trades")
        logger.info("    4. loss_control  - Cut losers early")
        logger.info("    5. risk_reward   - Good risk/reward ratios")
    
    logger.info("="*80)
    
    # Build custom config if needed
    config_updates = {}
    
    # Apply command line overrides
    if hasattr(args, 'episodes') and args.episodes:
        config_updates['rl_episodes'] = args.episodes
    
    if hasattr(args, 'max_wait') and args.max_wait:
        if 'trade_config' not in config_updates:
            config_updates['trade_config'] = {}
        config_updates['trade_config']['max_wait_steps'] = args.max_wait
    
    if hasattr(args, 'max_hold') and args.max_hold:
        if 'trade_config' not in config_updates:
            config_updates['trade_config'] = {}
        config_updates['trade_config']['max_hold_steps'] = args.max_hold
    
    # Start progress monitor if requested
    if hasattr(args, 'monitor') and args.monitor:
        logger.info("\nStarting progress monitor in new terminal...")
        try:
            if sys.platform == 'win32':
                subprocess.Popen(['cmd', '/c', 'start', 'python', 'src/training_monitor.py'])
            else:
                subprocess.Popen(['gnome-terminal', '--', 'python', 'src/training_monitor.py'])
        except Exception as e:
            logger.warning(f"Could not start monitor: {e}")
    
    try:
        # Create trainer with optional config path
        config_path = args.config if hasattr(args, 'config') and args.config else None
        trainer = TradeBasedTrainer(config_path)
        
        # Apply any config updates
        for key, value in config_updates.items():
            if isinstance(value, dict) and key in trainer.config:
                trainer.config[key].update(value)
            else:
                trainer.config[key] = value
        
        # ═══════════════════════════════════════════════════════════════════════
        # ENABLE MULTI-OBJECTIVE MODE IF REQUESTED
        # ═══════════════════════════════════════════════════════════════════════
        if use_mo:
            trainer.config['use_multi_objective'] = True
            logger.info("\n Multi-objective rewards ENABLED")
            
            # Apply MO-specific config overrides if provided
            if hasattr(args, 'weight_pnl') and args.weight_pnl is not None:
                trainer.config['mo_reward_config']['weight_pnl_quality'] = args.weight_pnl
            if hasattr(args, 'weight_hold') and args.weight_hold is not None:
                trainer.config['mo_reward_config']['weight_hold_duration'] = args.weight_hold
            if hasattr(args, 'weight_win') and args.weight_win is not None:
                trainer.config['mo_reward_config']['weight_win_achieved'] = args.weight_win
            if hasattr(args, 'weight_loss') and args.weight_loss is not None:
                trainer.config['mo_reward_config']['weight_loss_control'] = args.weight_loss
            if hasattr(args, 'weight_rr') and args.weight_rr is not None:
                trainer.config['mo_reward_config']['weight_risk_reward'] = args.weight_rr
        
        # Determine what to train
        if args.ml:
            logger.info("\n>>> Training ML Predictor Only <<<\n")
            results = trainer.train_complete_system(train_ml=True, train_rl=False)
            logger.info(" ML training complete")
            
        elif args.rl:
            mode_str = "Multi-Objective " if use_mo else ""
            logger.info(f"\n>>> {mode_str}Trade-Based RL Training Only <<<\n")
            logger.info("  Will load existing ML model for feature selection")
            results = trainer.train_complete_system(train_ml=False, train_rl=True)
            logger.info(f" {mode_str}Trade-based RL training complete")
            
        elif args.both:
            mode_str = "Multi-Objective " if use_mo else ""
            logger.info(f"\n>>> Training Both ML and {mode_str}Trade-Based RL <<<\n")
            results = trainer.train_complete_system(train_ml=True, train_rl=True)
            logger.info(f" Complete {mode_str.lower()}trade-based system training finished")
            
        else:
            logger.error("Please specify --ml, --rl, or --both")
            sys.exit(1)
        
        # Print success message
        print("\n" + "="*80)
        print(" TRADE-BASED TRAINING COMPLETED SUCCESSFULLY")
        if use_mo:
            print(" (Multi-Objective Mode)")
        print("="*80)
        
        print("\nWhat the agent learned:")
        print("  - WHEN to enter trades (good setups)")
        print("  - WHEN to exit trades (optimal timing)")
        print("  - Trade QUALITY over quantity")
        
        if use_mo:
            print("\nMulti-objective signals used:")
            print("  - pnl_quality:   Maximize wins, minimize losses")
            print("  - hold_duration: Hold trades longer")
            print("  - win_achieved:  Win more trades")
            print("  - loss_control:  Cut losers early")
            print("  - risk_reward:   Good risk/reward ratios")
        
        print("\nNext steps:")
        print("  1. Run backtesting: python main.py backtest --walk-forward")
        print("  2. Compare with time-based: python main.py train --both --fast")
        print("  3. Start paper trading: python main.py paper --start")
        
    except Exception as e:
        logger.error(f"Trade-based training failed: {e}", exc_info=True)
        sys.exit(1)


def handle_train_command_optimized(args):
    """Handle optimized model training commands WITH EXPLAINABILITY"""
    from src.optimized_train_system import train_ml_only_fast, train_rl_only_fast, train_both_fast
    from src.gpu_optimization import setup_optimal_training
    
    # ⭐ NEW: Check if explainability requested ⭐
    if args.explain:
        logger.info("="*80)
        logger.info(" EXPLAINABILITY ENABLED")
        logger.info("="*80)
        logger.info(f"  Explain Frequency: Every {args.explain_freq} steps")
        logger.info(f"  Verbose Mode: {'ON' if args.verbose else 'OFF'}")
        logger.info(f"  Save Directory: {args.explain_dir}")
        logger.info("="*80 + "\n")
    
    logger.info("="*80)
    logger.info("STARTING OPTIMIZED MODEL TRAINING (TIME-BASED)")
    logger.info("="*80)
    logger.info("  Episode = fixed number of steps")
    logger.info("  Tip: Use --trade-based for trade-quality learning!")
    logger.info("="*80)
    
    # Setup optimal configuration
    config = setup_optimal_training(args.config if hasattr(args, 'config') else None)
    
    # ⭐ ADD EXPLAINABILITY SETTINGS TO CONFIG ⭐
    if args.explain:
        config['explainability'] = {
            'enabled': True,
            'verbose': args.verbose,
            'explain_frequency': args.explain_freq,
            'save_dir': args.explain_dir
        }
    
    config_path = 'config/optimal_training_config.json'
    
    # Start progress monitor if requested
    if args.monitor:
        logger.info("\nStarting progress monitor in new terminal...")
        try:
            if sys.platform == 'win32':
                subprocess.Popen(['cmd', '/c', 'start', 'python', 'src/training_monitor.py'])
            else:
                subprocess.Popen(['gnome-terminal', '--', 'python', 'src/training_monitor.py'])
        except Exception as e:
            logger.warning(f"Could not start monitor: {e}")
    
    try:
        if args.ml:
            logger.info("\n>>> Training ML Predictor Only (FAST) <<<\n")
            results = train_ml_only_fast(config_path)
            logger.info(" ML training complete")
            
        elif args.rl:
            logger.info("\n>>> Training RL Agent Only (FAST) <<<\n")
            # ⭐ Pass explainability args to RL training ⭐
            results = train_rl_only_fast(config_path, 
                                        explain=args.explain,
                                        explain_freq=args.explain_freq,
                                        verbose=args.verbose,
                                        explain_dir=args.explain_dir)
            logger.info(" RL training complete")
            
        elif args.both:
            logger.info("\n>>> Training Both ML and RL (FAST) <<<\n")
            # ⭐ Pass explainability args to full training ⭐
            results = train_both_fast(config_path,
                                     explain=args.explain,
                                     explain_freq=args.explain_freq,
                                     verbose=args.verbose,
                                     explain_dir=args.explain_dir)
            logger.info(" Complete system training finished")
            
        else:
            logger.error("Please specify --ml, --rl, or --both")
            sys.exit(1)
        
        # Print success message
        print("\n" + "="*80)
        print(" TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # ⭐ Show explainability report location if enabled ⭐
        if args.explain:
            print(f"\n EXPLAINABILITY REPORTS:")
            print(f"  - Decision history: {args.explain_dir}/decision_history.json")
            print(f"  - Policy report: {args.explain_dir}/final_policy_report.txt")
            print(f"  - Visualizations: {args.explain_dir}/visualizations/")
            print(f"\nTo view the policy report:")
            print(f"  cat {args.explain_dir}/final_policy_report.txt")
        
        print("\nNext steps:")
        print("  1. Run backtesting: python main.py backtest --run --explain")
        print("  2. Start paper trading: python main.py paper --start --explain")
        print("  3. View training report: python src/training_monitor.py --report")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


def handle_train_command_standard(args):
    """Handle standard (non-optimized) training"""
    from src.train_system import train_ml_only, train_rl_only, train_both
    
    logger.info("="*80)
    logger.info("STARTING STANDARD MODEL TRAINING")
    logger.info("="*80)
    logger.info("Tip: Use --fast flag for 10-20x speed improvement!")
    logger.info("Tip: Use --trade-based for trade-quality learning!")
    
    config_path = getattr(args, 'config', None)
    
    try:
        if args.ml:
            logger.info("\n>>> Training ML Predictor Only <<<\n")
            results = train_ml_only(config_path)
            logger.info(" ML training complete")
            
        elif args.rl:
            logger.info("\n>>> Training RL Agent Only <<<\n")
            results = train_rl_only(config_path)
            logger.info(" RL training complete")
            
        elif args.both:
            logger.info("\n>>> Training Both ML and RL <<<\n")
            results = train_both(config_path)
            logger.info(" Complete system training finished")
            
        else:
            logger.error("Please specify --ml, --rl, or --both")
            sys.exit(1)
        
        print("\n" + "="*80)
        print(" TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


def handle_monitor_command(args):
    """Handle progress monitoring"""
    from src.training_monitor import monitor_training, TrainingMonitor
    
    if args.report:
        monitor = TrainingMonitor(args.file)
        report = monitor.generate_summary_report(args.output)
        print(report)
    else:
        logger.info("Starting real-time training monitor...")
        monitor_training(args.file, args.refresh)


def handle_optimize_command(args):
    """Handle optimization utilities"""
    from src.gpu_optimization import GPUOptimizer, setup_optimal_training
    
    if args.info:
        optimizer = GPUOptimizer()
        optimizer.print_system_info()
    
    if args.benchmark:
        optimizer = GPUOptimizer()
        optimizer.benchmark_operations()
    
    if args.config:
        config = setup_optimal_training()
        print("\n Optimal configuration generated")
        print("Config saved to: config/optimal_training_config.json")
    
    if args.estimate:
        optimizer = GPUOptimizer()
        config = optimizer.get_optimal_config()
        config['assets'] = ['ETH_USDT', 'DOT_USDT', 'SOL_USDT', 'ADA_USDT', 'AVAX_USDT']
        config['rl_episodes'] = int(args.episodes) if args.episodes else 100
        
        estimates = optimizer.estimate_training_time(config)
        
        print("\n" + "="*80)
        print("TRAINING TIME ESTIMATES")
        print("="*80)
        for key, value in estimates.items():
            if 'min' in key:
                print(f"{key.replace('_', ' ').title()}: {value:.1f} minutes")
            elif 'hours' in key:
                print(f"\nTotal Estimated Time: {value:.2f} hours")
        print("="*80)


def handle_data_command(args):
    """Handle data management commands - NOW USING BINANCE"""
    from src.data.binance_historical import BinanceHistoricalData, BinanceDataConfig
    
    if args.fetch:
        logger.info("="*80)
        logger.info("FETCHING HISTORICAL DATA FROM BINANCE")
        logger.info("="*80)
        logger.info("This will download clean OHLCV data for training")
        logger.info("Live trading will still use Kraken")
        logger.info("="*80)
        
        # Configure what to fetch
        config = BinanceDataConfig(
            symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT', 'AVAXUSDT'],
            timeframes=['5m', '15m', '1h', '4h', '1d'],
            start_date=args.start_date if hasattr(args, 'start_date') and args.start_date else '2020-01-01'
        )
        
        # Override with command line args if provided
        if hasattr(args, 'symbols') and args.symbols:
            config.symbols = args.symbols
        
        connector = BinanceHistoricalData(config)
        
        logger.info("\nStarting download...")
        connector.fetch_all_data()
        
        logger.info("\nValidating downloaded data...")
        connector.validate_data()
        
        logger.info("\n" + "="*80)
        logger.info(" DATA FETCH COMPLETE")
        logger.info("="*80)
        logger.info("Data saved to: data/raw/")
        logger.info("Next step: python main.py features --calculate")
        
    elif args.update:
        logger.info("="*80)
        logger.info("UPDATING DATA WITH NEW CANDLES")
        logger.info("="*80)
        
        connector = BinanceHistoricalData()
        
        logger.info("Fetching new candles from Binance...")
        connector.update_existing_data()
        
        logger.info("\nValidating updated data...")
        connector.validate_data()
        
        logger.info("\n Data update complete")
        
    elif args.validate:
        logger.info("="*80)
        logger.info("VALIDATING DATA QUALITY")
        logger.info("="*80)
        
        connector = BinanceHistoricalData()
        connector.validate_data()
        
        logger.info("\n Validation complete")
        
    elif args.summary:
        logger.info("="*80)
        logger.info("DATA SUMMARY")
        logger.info("="*80)
        
        connector = BinanceHistoricalData()
        summary = connector.get_data_summary()
        
        import json
        print(json.dumps(summary, indent=2))
        
    elif args.clean:
        logger.info("="*80)
        logger.info("CLEANING DATA")
        logger.info("="*80)
        logger.warning("This will remove all downloaded data files!")
        
        response = input("Type 'YES' to confirm: ")
        if response == 'YES':
            import shutil
            from pathlib import Path
            
            data_path = Path('./data/raw/')
            if data_path.exists():
                shutil.rmtree(data_path)
                logger.info(" Data directory removed")
            else:
                logger.info("Data directory doesn't exist")
        else:
            logger.info("Cancelled")


def handle_features_command(args):
    """Handle feature engineering commands"""
    from src.features.feature_engineer import FeatureEngineer
    from src.features.selector import FeatureSelector
    
    feature_engineer = FeatureEngineer()
    
    if args.calculate:
        logger.info("Calculating features...")
        feature_engineer.calculate_and_save_all_features()
        logger.info(" Feature calculation complete")
        
    elif args.select:
        logger.info("Running feature selection...")
        selector = FeatureSelector()
        results = selector.select_features()
        logger.info(f" Feature selection complete: {results}")
        
    elif args.analyze:
        logger.info("Analyzing feature importance...")
        feature_engineer.analyze_feature_importance()
        logger.info(" Feature analysis complete")


def handle_backtest_command(args):
    """Handle backtesting commands"""
    
    if args.run:
        logger.info("Running standard backtest...")
        from src.backtesting.backtester import Backtester, BacktestConfig
        
        try:
            backtester = Backtester(BacktestConfig())
            results = backtester.run()
            
            # Print results
            print("\n" + "="*80)
            print(" BACKTEST COMPLETED")
            print("="*80)
            print(f"\nSharpe Ratio: {results.sharpe_ratio:.3f}")
            print(f"Total Return: {results.total_return:.2%}")
            print(f"Max Drawdown: {results.max_drawdown:.2%}")
            print(f"Total Trades: {results.total_trades}")
            print(f"Win Rate: {results.win_rate:.2%}")
            
            logger.info(" Backtest complete")
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            sys.exit(1)
        
    elif args.walk_forward:
        # ⭐ Check if explainability requested ⭐
        if hasattr(args, 'explain') and args.explain:
            logger.info("="*80)
            logger.info(" EXPLAINABILITY ENABLED FOR WALK-FORWARD VALIDATION")
            logger.info("="*80)
            logger.info(f"  Explain Frequency: Every {args.explain_freq} decisions")
            logger.info(f"  Verbose Mode: {'ON' if args.verbose else 'OFF'}")
            logger.info(f"  Save Directory: {args.explain_dir}")
            logger.info("="*80 + "\n")
        
        logger.info("="*80)
        logger.info("WALK-FORWARD VALIDATION (2025 HOLDOUT DATA)")
        logger.info("="*80)
        
        try:
            # Import the comprehensive walk-forward runner from project root
            from src.backtesting.walk_forward_runner import run_walk_forward_validation
            
            # ⭐ Run validation with explainability parameters ⭐
            results = run_walk_forward_validation(
                explain=args.explain if hasattr(args, 'explain') else False,
                explain_frequency=args.explain_freq if hasattr(args, 'explain_freq') else 100,
                verbose=args.verbose if hasattr(args, 'verbose') else False,
                explain_dir=args.explain_dir if hasattr(args, 'explain_dir') else 'logs/backtest_explanations'
            )
            
            logger.info("\n Walk-forward validation complete!")
            logger.info(f"   Test Period: Jan 1 - Oct 26, 2025")
            logger.info(f"   Average Return: {results['avg_return']:.2f}%")
            logger.info(f"   Average Sharpe: {results['avg_sharpe']:.2f}")
            logger.info(f"   Successful on: {len([r for r in results['individual_results'].values() if r['total_return'] > 0])}/{len(results['individual_results'])} assets")
            
            # ⭐ Show explainability report location if enabled ⭐
            if hasattr(args, 'explain') and args.explain:
                print("\n" + "="*80)
                print(" EXPLAINABILITY REPORTS")
                print("="*80)
                print(f"  - Decision history: {args.explain_dir}/decision_history.json")
                print(f"  - Policy report: {args.explain_dir}/backtest_policy_report.txt")
                print(f"  - Visualizations: {args.explain_dir}/visualizations/")
                print(f"\nTo view the policy report:")
                print(f"  cat {args.explain_dir}/backtest_policy_report.txt")
                print("="*80)
            
        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}", exc_info=True)
            sys.exit(1)
    
    else:
        logger.error("Please specify --walk-forward or --run")
        sys.exit(1)


def handle_paper_command(args):
    """Handle paper trading commands"""
    from src.live.paper_trader import PaperTrader
    
    if args.start:
        logger.info("Starting paper trading...")
        trader = PaperTrader()
        trader.start()
        logger.info(" Paper trading started")
        
    elif args.stop:
        logger.info("Stopping paper trading...")
        trader = PaperTrader()
        trader.stop()
        logger.info(" Paper trading stopped")
        
    elif args.status:
        logger.info("Checking paper trading status...")
        trader = PaperTrader()
        status = trader.get_status()
        logger.info(f"Status: {status}")

def handle_live_command(args):
    """Handle live trading commands"""
    from src.live.live_trader import LiveTrader, LiveTradingConfig
    import os
    
    logger.info("="*80)
    logger.info("LIVE TRADING - KRAKEN EXCHANGE")
    logger.info("="*80)
    logger.info("  REAL MONEY TRADING")
    logger.info("="*80)
    
    if args.start:
        logger.info("\n>>> Starting Live Trading <<<\n")
        
        # Load configuration
        config = LiveTradingConfig(
            initial_capital=args.capital if hasattr(args, 'capital') else 100.0,
            kraken_api_key=os.getenv('KRAKEN_API_KEY'),
            kraken_api_secret=os.getenv('KRAKEN_API_SECRET'),
            dry_run=args.dry_run if hasattr(args, 'dry_run') else False
        )
        
        # Verify API keys
        if not config.kraken_api_key or not config.kraken_api_secret:
            logger.error("Kraken API keys not found!")
            logger.error("Set KRAKEN_API_KEY and KRAKEN_API_SECRET in .env file")
            sys.exit(1)
        
        # Safety check
        if not config.dry_run:
            print("\n" + " "*20)
            print("YOU ARE ABOUT TO START LIVE TRADING WITH REAL MONEY")
            print("Capital: ${}".format(config.initial_capital))
            print("Exchange: Kraken")
            print("Symbols: {}".format(', '.join(config.symbols)))
            print(" "*20 + "\n")
            
            response = input("Type 'YES' to confirm and start live trading: ")
            if response != 'YES':
                logger.info("Live trading cancelled")
                return
        
        # Initialize and start trader
        try:
            trader = LiveTrader(config)
            trader.start()
            
            # Keep running until interrupted
            import time
            while trader.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("\nStopping live trading...")
            trader.stop()
        except Exception as e:
            logger.error(f"Live trading failed: {e}", exc_info=True)
            sys.exit(1)
        
    elif args.stop:
        logger.info("Stopping live trading...")
        from src.live.live_trader import stop_live_trading
        stop_live_trading()
        logger.info(" Live trading stopped")
        
    elif args.status:
        logger.info("Checking live trading status...")
        from src.live.live_trader import get_live_trading_status
        get_live_trading_status()


def handle_dashboard_command(args):
    """Handle dashboard commands"""
    logger.info(f"Launching monitoring dashboard on port {args.port}...")
    
    try:
        from dashboard.app import run_dashboard
        run_dashboard(port=args.port)
    except ImportError:
        logger.error("Dashboard dependencies not installed")
        logger.info("Install with: pip install dash plotly")


def main():
    """Main entry point for the trading bot."""
    
    parser = argparse.ArgumentParser(
        description='Kraken Trading Bot - Hybrid ML/RL System (OPTIMIZED + TRADE-BASED + MULTI-OBJECTIVE)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ⭐ TRADE-BASED TRAINING (learns trade quality!)
  python main.py train --both --trade-based          # Train with trade-based episodes
  python main.py train --rl --trade-based            # RL only, trade-based
  python main.py train --both --trade-based --episodes 10000  # More episodes
  
  # ⭐ MULTI-OBJECTIVE TRAINING (5 reward signals!)
  python main.py train --rl --trade-based --multi-objective   # Best of both!
  python main.py train --both --trade-based -mo               # Short flag
  python main.py train --rl --trade-based -mo --episodes 15000
  
  # Multi-objective with custom weights
  python main.py train --rl --trade-based -mo --weight-pnl 0.4 --weight-hold 0.3
  
  # OPTIMIZED TIME-BASED TRAINING (10-20x faster)
  python main.py train --both --fast --monitor       # Train with real-time monitoring
  python main.py train --rl --fast                   # Fast RL training with GPU
  
  # Monitor training progress (run in separate terminal)
  python main.py monitor                             # Live progress tracking
  python main.py monitor --report                    # Generate summary report
  
  # Optimization utilities
  python main.py optimize --info                     # Show system capabilities
  python main.py optimize --benchmark                # Run performance tests
  python main.py optimize --config                   # Generate optimal config
  python main.py optimize --estimate --episodes 500  # Estimate training time
  
  # Standard (slower) training
  python main.py train --both                        # Original training method
  
  # Data Management
  python main.py data --update                       # Update data from Kraken
  
  # Backtesting
  python main.py backtest --run                      # Run backtest
  python main.py backtest --walk-forward             # Walk-forward analysis
  
  # Paper Trading
  python main.py paper --start                       # Start paper trading
  
  # Dashboard
  python main.py dashboard                           # Launch web dashboard

TRAINING MODES COMPARISON:
  --trade-based : Each episode = 1 trade. Agent learns WHEN to enter/exit.
                  Best for: Swing trading, trade quality over quantity.
                  
  --trade-based --multi-objective (-mo) : Same as above + 5 separate reward signals.
                  Best for: Learning multiple trading objectives simultaneously.
                  Objectives: pnl_quality, hold_duration, win_achieved, loss_control, risk_reward
                  
  --fast        : Time-based episodes (fixed steps). Agent predicts next candle.
                  Best for: Scalping, high-frequency decisions.
                  
  (no flag)     : Standard training, slower but reliable.
        """
    )
    
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAIN COMMAND (with trade-based and multi-objective options)
    # ═══════════════════════════════════════════════════════════════════════════
    train_parser = subparsers.add_parser('train', help='Train models')
    
    # What to train
    train_group = train_parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument('--ml', action='store_true', help='Train ML predictor only')
    train_group.add_argument('--rl', action='store_true', help='Train RL agent only')
    train_group.add_argument('--both', action='store_true', help='Train both ML and RL')
    
    # Training mode (mutually exclusive)
    mode_group = train_parser.add_mutually_exclusive_group()
    mode_group.add_argument('--trade-based', action='store_true',
                           help='⭐ Trade-based training (1 episode = 1 trade)')
    mode_group.add_argument('--fast', action='store_true', 
                           help='Time-based optimized training (10-20x faster, GPU support)')
    
    # Multi-objective option (works with --trade-based)
    train_parser.add_argument('--multi-objective', '-mo', action='store_true',
                            help='⭐ Use Multi-Objective rewards (5 signals) - requires --trade-based')
    
    # Episode count
    train_parser.add_argument('--episodes', type=int, default=None,
                            help='Number of training episodes (default: 15000 for trade-based)')
    
    # General options
    train_parser.add_argument('--monitor', action='store_true',
                             help='Start progress monitor in new terminal')
    train_parser.add_argument('--config', help='Path to config file')
    
    # Trade-based specific options
    train_parser.add_argument('--max-wait', type=int,
                             help='[Trade-based] Max steps to wait for entry (default: 250)')
    train_parser.add_argument('--max-hold', type=int,
                             help='[Trade-based] Max steps to hold position (default: 300)')
    
    # Multi-objective weight options
    train_parser.add_argument('--weight-pnl', type=float,
                             help='[Multi-objective] Weight for pnl_quality (default: 0.35)')
    train_parser.add_argument('--weight-hold', type=float,
                             help='[Multi-objective] Weight for hold_duration (default: 0.25)')
    train_parser.add_argument('--weight-win', type=float,
                             help='[Multi-objective] Weight for win_achieved (default: 0.15)')
    train_parser.add_argument('--weight-loss', type=float,
                             help='[Multi-objective] Weight for loss_control (default: 0.15)')
    train_parser.add_argument('--weight-rr', type=float,
                             help='[Multi-objective] Weight for risk_reward (default: 0.10)')
    
    # Explainability options
    train_parser.add_argument('--explain', action='store_true', 
                             help='Enable explainability (show why agent makes decisions)')
    train_parser.add_argument('--verbose', action='store_true',
                             help='Explain EVERY decision (warning: lots of output)')
    train_parser.add_argument('--explain-freq', type=int, default=100,
                             help='How often to print explanations (default: 100)')
    train_parser.add_argument('--explain-dir', type=str, default='logs/explanations',
                             help='Directory to save explanations')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MONITOR COMMAND
    # ═══════════════════════════════════════════════════════════════════════════
    monitor_parser = subparsers.add_parser('monitor', help='Monitor training progress')
    monitor_parser.add_argument('--file', default='logs/training_progress.json',
                               help='Progress file to monitor')
    monitor_parser.add_argument('--refresh', type=float, default=2.0,
                               help='Refresh rate in seconds')
    monitor_parser.add_argument('--report', action='store_true',
                               help='Generate summary report')
    monitor_parser.add_argument('--output', help='Output file for report')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIMIZE COMMAND
    # ═══════════════════════════════════════════════════════════════════════════
    optimize_parser = subparsers.add_parser('optimize', help='Optimization utilities')
    optimize_parser.add_argument('--info', action='store_true',
                                help='Show system info and capabilities')
    optimize_parser.add_argument('--benchmark', action='store_true',
                                help='Run performance benchmarks')
    optimize_parser.add_argument('--config', action='store_true',
                                help='Generate optimal configuration')
    optimize_parser.add_argument('--estimate', action='store_true',
                                help='Estimate training time')
    optimize_parser.add_argument('--episodes', type=int,
                                help='Number of RL episodes for estimation')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA COMMAND
    # ═══════════════════════════════════════════════════════════════════════════
    data_parser = subparsers.add_parser('data', help='Data management (Binance)')
    data_group = data_parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--fetch', action='store_true', 
                           help='Fetch historical data from Binance')
    data_group.add_argument('--update', action='store_true', 
                           help='Update data with new candles')
    data_group.add_argument('--validate', action='store_true', 
                           help='Validate data quality')
    data_group.add_argument('--summary', action='store_true',
                           help='Show data summary')
    data_group.add_argument('--clean', action='store_true', 
                           help='Clean/remove all data')
    
    # Optional arguments for fetch
    data_parser.add_argument('--start-date', 
                            help='Start date for fetch (YYYY-MM-DD, default: 2020-01-01)')
    data_parser.add_argument('--symbols', nargs='+',
                            help='Specific symbols to fetch (default: all)')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURES COMMAND
    # ═══════════════════════════════════════════════════════════════════════════
    features_parser = subparsers.add_parser('features', help='Feature engineering')
    features_group = features_parser.add_mutually_exclusive_group(required=True)
    features_group.add_argument('--calculate', action='store_true', help='Calculate features')
    features_group.add_argument('--select', action='store_true', help='Select features')
    features_group.add_argument('--analyze', action='store_true', help='Analyze features')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BACKTEST COMMAND
    # ═══════════════════════════════════════════════════════════════════════════
    backtest_parser = subparsers.add_parser('backtest', help='Backtesting')
    backtest_group = backtest_parser.add_mutually_exclusive_group(required=True)
    backtest_group.add_argument('--run', action='store_true', help='Run backtest')
    backtest_group.add_argument('--walk-forward', action='store_true', 
                                help='Walk-forward analysis')
    
    backtest_parser.add_argument('--explain', action='store_true',
                                help='Explain backtest decisions')
    backtest_parser.add_argument('--verbose', action='store_true',
                                help='Verbose explanations')
    backtest_parser.add_argument('--explain-freq', type=int, default=100,
                                help='How often to print explanations (default: 100)')
    backtest_parser.add_argument('--explain-dir', type=str, default='logs/backtest_explanations',
                                help='Directory to save explanations')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIVE COMMAND
    # ═══════════════════════════════════════════════════════════════════════════
    live_parser = subparsers.add_parser('live', help='Live trading on Kraken')
    live_group = live_parser.add_mutually_exclusive_group(required=True)
    live_group.add_argument('--start', action='store_true', help='Start live trading')
    live_group.add_argument('--stop', action='store_true', help='Stop live trading')
    live_group.add_argument('--status', action='store_true', help='Check status')
    live_parser.add_argument('--capital', type=float, default=100.0, help='Starting capital')
    live_parser.add_argument('--dry-run', action='store_true', help='Simulate without real orders')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PAPER COMMAND
    # ═══════════════════════════════════════════════════════════════════════════
    paper_parser = subparsers.add_parser('paper', help='Paper trading')
    paper_group = paper_parser.add_mutually_exclusive_group(required=True)
    paper_group.add_argument('--start', action='store_true', help='Start paper trading')
    paper_group.add_argument('--stop', action='store_true', help='Stop paper trading')
    paper_group.add_argument('--status', action='store_true', help='Check status')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DASHBOARD COMMAND
    # ═══════════════════════════════════════════════════════════════════════════
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8050, help='Port number')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    from src.utils.logger import setup_logger
    global logger
    
    # Convert string log level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(args.log_level.upper(), logging.INFO)
    
    logger = setup_logger('main', level=log_level)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLE COMMANDS
    # ═══════════════════════════════════════════════════════════════════════════
    if args.command == 'train':
        # Check for multi-objective without trade-based
        if hasattr(args, 'multi_objective') and args.multi_objective:
            if not (hasattr(args, 'trade_based') and args.trade_based):
                logger.warning("--multi-objective requires --trade-based flag")
                logger.info("Adding --trade-based automatically...")
                args.trade_based = True
        
        # Route to appropriate handler
        if hasattr(args, 'trade_based') and args.trade_based:
            handle_train_command_trade_based(args)
        elif hasattr(args, 'fast') and args.fast:
            handle_train_command_optimized(args)
        else:
            handle_train_command_standard(args)
            
    elif args.command == 'monitor':
        handle_monitor_command(args)
    elif args.command == 'optimize':
        handle_optimize_command(args)
    elif args.command == 'data':
        handle_data_command(args)
    elif args.command == 'features':
        handle_features_command(args)
    elif args.command == 'backtest':
        handle_backtest_command(args)
    elif args.command == 'live':
        handle_live_command(args)
    elif args.command == 'paper':
        handle_paper_command(args)
    elif args.command == 'dashboard':
        handle_dashboard_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()