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
    logger.info("STARTING OPTIMIZED MODEL TRAINING")
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

    train_parser.add_argument('--explain', action='store_true', 
                             help='Enable explainability (show why agent makes decisions)')
    train_parser.add_argument('--verbose', action='store_true',
                             help='Explain EVERY decision (warning: lots of output)')
    train_parser.add_argument('--explain-freq', type=int, default=100,
                             help='How often to print explanations (default: 100)')
    train_parser.add_argument('--explain-dir', type=str, default='logs/explanations',
                             help='Directory to save explanations')
    
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
    
    backtest_parser.add_argument('--explain', action='store_true',
                                help='Explain backtest decisions')
    backtest_parser.add_argument('--verbose', action='store_true',
                                help='Verbose explanations')
    backtest_parser.add_argument('--explain-freq', type=int, default=100,
                                help='How often to print explanations (default: 100)')
    backtest_parser.add_argument('--explain-dir', type=str, default='logs/backtest_explanations',
                                help='Directory to save explanations')
    
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