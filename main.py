#!/usr/bin/env python3
"""
Kraken Trading Bot - Main Entry Point

This is the main entry point for the trading bot system.
It provides a CLI interface to run different components of the system.
"""

import argparse
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the trading bot."""
    
    parser = argparse.ArgumentParser(
        description='Kraken Trading Bot - Hybrid ML/RL System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data --update          # Update data from exchange
  python main.py data --validate        # Validate existing data
  python main.py features --calculate   # Calculate features
  python main.py train --ml             # Train ML model
  python main.py train --rl             # Train RL agent
  python main.py backtest --run         # Run backtest
  python main.py paper --start          # Start paper trading
  python main.py live --start           # Start live trading
  python main.py dashboard              # Launch monitoring dashboard
        """
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data management
    data_parser = subparsers.add_parser('data', help='Data management')
    data_parser.add_argument('--update', action='store_true', help='Update data from exchange')
    data_parser.add_argument('--validate', action='store_true', help='Validate existing data')
    data_parser.add_argument('--symbols', nargs='+', help='Specific symbols to process')
    data_parser.add_argument('--timeframes', nargs='+', help='Specific timeframes to process')
    
    # Feature engineering
    features_parser = subparsers.add_parser('features', help='Feature engineering')
    features_parser.add_argument('--calculate', action='store_true', help='Calculate features')
    features_parser.add_argument('--select', action='store_true', help='Run feature selection')
    features_parser.add_argument('--analyze', action='store_true', help='Analyze feature importance')
    
    # Model training
    train_parser = subparsers.add_parser('train', help='Model training')
    train_parser.add_argument('--ml', action='store_true', help='Train ML predictor')
    train_parser.add_argument('--rl', action='store_true', help='Train RL agent')
    train_parser.add_argument('--both', action='store_true', help='Train both ML and RL')
    
    # Backtesting
    backtest_parser = subparsers.add_parser('backtest', help='Backtesting')
    backtest_parser.add_argument('--run', action='store_true', help='Run backtest')
    backtest_parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward analysis')
    backtest_parser.add_argument('--optimize', action='store_true', help='Optimize parameters')
    
    # Paper trading
    paper_parser = subparsers.add_parser('paper', help='Paper trading')
    paper_parser.add_argument('--start', action='store_true', help='Start paper trading')
    paper_parser.add_argument('--stop', action='store_true', help='Stop paper trading')
    paper_parser.add_argument('--status', action='store_true', help='Check paper trading status')
    
    # Live trading
    live_parser = subparsers.add_parser('live', help='Live trading')
    live_parser.add_argument('--start', action='store_true', help='Start live trading')
    live_parser.add_argument('--stop', action='store_true', help='Stop live trading')
    live_parser.add_argument('--status', action='store_true', help='Check live trading status')
    live_parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    
    # Dashboard
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch monitoring dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    Path('./logs').mkdir(exist_ok=True)
    
    # Execute command
    try:
        if args.command == 'data':
            # Import data commands module
            from src.data.data_commands import DataCommands
            
            logger.info("Initializing data management system...")
            dc = DataCommands()
            
            if args.update:
                logger.info("Starting data update from Kraken...")
                results = dc.update_all_data(
                    symbols=args.symbols,
                    timeframes=args.timeframes,
                    verbose=True
                )
                logger.info(f"Data update completed: {results['total_candles_added']} new candles added")
                
            elif args.validate:
                logger.info("Starting data validation...")
                results = dc.validate_all_data(
                    symbols=args.symbols,
                    timeframes=args.timeframes,
                    auto_fix=False,  # Only report, don't fix
                    verbose=True
                )
                logger.info(f"Validation completed: {results['files_with_errors']} files with errors")
                
            else:
                logger.warning("No data operation specified. Use --update or --validate")
                print("Please specify an operation: --update or --validate")
                
        elif args.command == 'features':
            logger.info("Feature engineering module")
            if args.calculate:
                logger.info("Calculating features...")
                print("Feature calculation not yet implemented")
                # from src.features.feature_engineer import FeatureEngineer
                # feature_engineer = FeatureEngineer()
                # feature_engineer.calculate_all_features()
            elif args.select:
                logger.info("Running feature selection...")
                print("Feature selection not yet implemented")
                # from src.features.selector import FeatureSelector
                # selector = FeatureSelector()
                # selector.run_all_selections()
            elif args.analyze:
                logger.info("Analyzing feature importance...")
                print("Feature analysis not yet implemented")
                
        elif args.command == 'train':
            logger.info("Model training module")
            if args.ml:
                logger.info("Training ML predictor...")
                print("ML training not yet implemented")
                # from src.models.ml_predictor import MLPredictor
                # ml_predictor = MLPredictor()
                # ml_predictor.train()
            elif args.rl:
                logger.info("Training RL agent...")
                print("RL training not yet implemented")
                # from src.models.dqn_agent import DQNAgent
                # agent = DQNAgent()
                # agent.train()
            elif args.both:
                logger.info("Training both ML and RL models...")
                print("Combined training not yet implemented")
                
        elif args.command == 'backtest':
            logger.info("Backtesting module")
            if args.run:
                logger.info("Running backtest...")
                print("Backtesting not yet implemented")
                # from src.backtesting.backtester import Backtester
                # backtester = Backtester()
                # results = backtester.run()
            elif args.walk_forward:
                logger.info("Running walk-forward analysis...")
                print("Walk-forward analysis not yet implemented")
            elif args.optimize:
                logger.info("Optimizing parameters...")
                print("Parameter optimization not yet implemented")
                
        elif args.command == 'paper':
            logger.info("Paper trading module")
            if args.start:
                logger.info("Starting paper trading...")
                print("Paper trading not yet implemented")
                # from src.live.paper_trader import PaperTrader
                # trader = PaperTrader()
                # trader.start()
            elif args.stop:
                logger.info("Stopping paper trading...")
                print("Paper trading stop not yet implemented")
            elif args.status:
                logger.info("Checking paper trading status...")
                print("Paper trading status not yet implemented")
                
        elif args.command == 'live':
            logger.info("Live trading module")
            if args.start:
                logger.info("Starting live trading...")
                print("  Live trading not yet implemented")
                print("  This will trade with real money!")
                # from src.live.live_trader import LiveTrader
                # trader = LiveTrader()
                # trader.start()
            elif args.stop:
                logger.info("Stopping live trading...")
                print("Live trading stop not yet implemented")
            elif args.status:
                logger.info("Checking live trading status...")
                print("Live trading status not yet implemented")
                
        elif args.command == 'dashboard':
            logger.info("Launching monitoring dashboard...")
            print("Dashboard not yet implemented")
            # from dashboard.app import launch_dashboard
            # launch_dashboard(port=args.port)
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        print("\n\nOperation cancelled by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        print(f"\n Error: {e}")
        print("Check logs/trading.log for details")
        sys.exit(1)


if __name__ == '__main__':
    main()