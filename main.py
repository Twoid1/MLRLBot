#!/usr/bin/env python3
"""
Kraken Trading Bot - Main Entry Point (OPTIMIZED)

Added:
- Fast training with GPU support
- Real-time progress monitoring
- Automatic optimization
- Performance benchmarking
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


def handle_train_command_optimized(args):
    """Handle optimized model training commands"""
    from src.optimized_train_system import train_ml_only_fast, train_rl_only_fast, train_both_fast
    from src.gpu_optimization import setup_optimal_training
    
    logger.info("="*80)
    logger.info("STARTING OPTIMIZED MODEL TRAINING")
    logger.info("="*80)
    
    # Setup optimal configuration and save to file
    config = setup_optimal_training(args.config if hasattr(args, 'config') else None)
    
    # Use the saved config file path
    config_path = 'config/optimal_training_config.json'
    
    # Start progress monitor in background if requested
    if args.monitor:
        logger.info("\nStarting progress monitor in new terminal...")
        try:
            if sys.platform == 'win32':
                subprocess.Popen(['cmd', '/c', 'start', 'python', 'src/training_monitor.py'])
            else:
                subprocess.Popen(['gnome-terminal', '--', 'python', 'src/training_monitor.py'])
        except Exception as e:
            logger.warning(f"Could not start monitor: {e}")
            logger.info("You can manually run: python src/training_monitor.py")
    
    try:
        if args.ml:
            logger.info("\n>>> Training ML Predictor Only (FAST) <<<\n")
            results = train_ml_only_fast(config_path)
            logger.info(" ML training complete")
            
        elif args.rl:
            logger.info("\n>>> Training RL Agent Only (FAST) <<<\n")
            results = train_rl_only_fast(config_path)
            logger.info(" RL training complete")
            
        elif args.both:
            logger.info("\n>>> Training Both ML and RL (FAST) <<<\n")
            results = train_both_fast(config_path)
            logger.info(" Complete system training finished")
            
        else:
            logger.error("Please specify --ml, --rl, or --both")
            sys.exit(1)
        
        # Print success message
        print("\n" + "="*80)
        print(" TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # Print speedup info
        if 'speedup_metrics' in results:
            speedup = results['speedup_metrics']
            print(f"\nTraining Time: {speedup['total_time_hours']:.2f} hours")
            print(f"Estimated Speedup: {speedup['estimated_speedup']:.1f}x faster")
        
        print("\nNext steps:")
        print("  1. Run backtesting: python main.py backtest --run")
        print("  2. Start paper trading: python main.py paper --start")
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
        config['assets'] = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'ADA_USDT', 'DOT_USDT']
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
    """Handle data management commands"""
    from src.data.data_manager import DataManager
    from src.data.validator import DataValidator
    
    data_manager = DataManager()
    
    if args.update:
        logger.info("Updating data from Kraken...")
        data_manager.update_all_data()
        logger.info(" Data update complete")
        
    elif args.validate:
        logger.info("Validating existing data...")
        validator = DataValidator()
        results = validator.validate_all_data()
        logger.info(f" Validation complete: {results}")
        
    elif args.clean:
        logger.info("Cleaning data...")
        data_manager.clean_data()
        logger.info(" Data cleaning complete")


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
        
        backtester = Backtester(BacktestConfig())
        results = backtester.run()
        logger.info(f" Backtest complete")
        
    elif args.walk_forward:
        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD VALIDATION (2025 HOLDOUT DATA)")
        logger.info("="*80)
        
        try:
            # Import the comprehensive walk-forward runner from project root
            from src.backtesting.walk_forward_runner import run_walk_forward_validation
            
            # Run validation on 2025 data from data/raw
            results = run_walk_forward_validation()
            
            logger.info("\n Walk-forward validation complete!")
            logger.info(f"   Test Period: Jan 1 - Oct 26, 2025")
            logger.info(f"   Average Return: {results['avg_return']:.2f}%")
            logger.info(f"   Average Sharpe: {results['avg_sharpe']:.2f}")
            logger.info(f"   Successful on: {len([r for r in results['individual_results'].values() if r['total_return'] > 0])}/{len(results['individual_results'])} assets")
            
        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}", exc_info=True)
            import sys
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
        description='Kraken Trading Bot - Hybrid ML/RL System (OPTIMIZED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OPTIMIZED TRAINING (NEW - 10-20x faster!)
  python main.py train --both --fast --monitor      # Train with real-time monitoring
  python main.py train --rl --fast                  # Fast RL training with GPU
  
  # Monitor training progress (run in separate terminal)
  python main.py monitor                            # Live progress tracking
  python main.py monitor --report                   # Generate summary report
  
  # Optimization utilities
  python main.py optimize --info                    # Show system capabilities
  python main.py optimize --benchmark               # Run performance tests
  python main.py optimize --config                  # Generate optimal config
  python main.py optimize --estimate --episodes 500 # Estimate training time
  
  # Standard (slower) training
  python main.py train --both                       # Original training method
  
  # Data Management
  python main.py data --update                      # Update data from Kraken
  
  # Backtesting
  python main.py backtest --run                     # Run backtest
  python main.py backtest --walk-forward            # Walk-forward analysis
  
  # Paper Trading
  python main.py paper --start                      # Start paper trading
  
  # Dashboard
  python main.py dashboard                          # Launch web dashboard
        """
    )
    
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command (OPTIMIZED)
    train_parser = subparsers.add_parser('train', help='Train models')
    train_group = train_parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument('--ml', action='store_true', help='Train ML predictor only')
    train_group.add_argument('--rl', action='store_true', help='Train RL agent only')
    train_group.add_argument('--both', action='store_true', help='Train both ML and RL')
    train_parser.add_argument('--fast', action='store_true', 
                             help='Use optimized training (10-20x faster, GPU support)')
    train_parser.add_argument('--monitor', action='store_true',
                             help='Start progress monitor in new terminal')
    train_parser.add_argument('--config', help='Path to config file')
    
    # Monitor command (NEW)
    monitor_parser = subparsers.add_parser('monitor', help='Monitor training progress')
    monitor_parser.add_argument('--file', default='logs/training_progress.json',
                               help='Progress file to monitor')
    monitor_parser.add_argument('--refresh', type=float, default=2.0,
                               help='Refresh rate in seconds')
    monitor_parser.add_argument('--report', action='store_true',
                               help='Generate summary report')
    monitor_parser.add_argument('--output', help='Output file for report')
    
    # Optimize command (NEW)
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
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Data management')
    data_group = data_parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--update', action='store_true', help='Update data')
    data_group.add_argument('--validate', action='store_true', help='Validate data')
    data_group.add_argument('--clean', action='store_true', help='Clean data')
    
    # Features command
    features_parser = subparsers.add_parser('features', help='Feature engineering')
    features_group = features_parser.add_mutually_exclusive_group(required=True)
    features_group.add_argument('--calculate', action='store_true', help='Calculate features')
    features_group.add_argument('--select', action='store_true', help='Select features')
    features_group.add_argument('--analyze', action='store_true', help='Analyze features')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtesting')
    backtest_group = backtest_parser.add_mutually_exclusive_group(required=True)
    backtest_group.add_argument('--run', action='store_true', help='Run backtest')
    backtest_group.add_argument('--walk-forward', action='store_true', 
                                help='Walk-forward analysis')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Paper trading')
    paper_group = paper_parser.add_mutually_exclusive_group(required=True)
    paper_group.add_argument('--start', action='store_true', help='Start paper trading')
    paper_group.add_argument('--stop', action='store_true', help='Stop paper trading')
    paper_group.add_argument('--status', action='store_true', help='Check status')
    
    # Dashboard command
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
    
    # Handle commands
    if args.command == 'train':
        if args.fast:
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
    elif args.command == 'paper':
        handle_paper_command(args)
    elif args.command == 'dashboard':
        handle_dashboard_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()