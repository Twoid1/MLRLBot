"""
Logging Utility Module
Provides centralized logging configuration for the trading bot
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import os


def setup_logger(
    name: str = 'trading_bot',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (default: logs/trading_bot.log)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console
        file_output: Whether to output to file
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        # Set default log file path
        if log_file is None:
            log_file = 'logs/trading_bot.log'
        
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = 'trading_bot') -> logging.Logger:
    """
    Get an existing logger or create a new one
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def setup_trading_loggers() -> dict:
    """
    Set up multiple loggers for different components
    
    Returns:
        Dictionary of logger instances
    """
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Main application logger
    main_logger = setup_logger(
        name='main',
        log_file='logs/trading_bot.log',
        level=logging.INFO
    )
    
    # Trading execution logger
    trade_logger = setup_logger(
        name='trading',
        log_file='logs/trades.log',
        level=logging.INFO
    )
    
    # Error logger
    error_logger = setup_logger(
        name='errors',
        log_file='logs/errors.log',
        level=logging.ERROR,
        console_output=True
    )
    
    # Data logger
    data_logger = setup_logger(
        name='data',
        log_file='logs/data.log',
        level=logging.INFO
    )
    
    # Model training logger
    training_logger = setup_logger(
        name='training',
        log_file='logs/training.log',
        level=logging.INFO
    )
    
    return {
        'main': main_logger,
        'trading': trade_logger,
        'errors': error_logger,
        'data': data_logger,
        'training': training_logger
    }


def log_trade(
    logger: logging.Logger,
    action: str,
    symbol: str,
    price: float,
    quantity: float,
    side: str = 'BUY'
):
    """
    Log a trade execution
    
    Args:
        logger: Logger instance
        action: Trade action (OPEN, CLOSE, etc.)
        symbol: Trading symbol
        price: Execution price
        quantity: Trade quantity
        side: Trade side (BUY/SELL)
    """
    logger.info(
        f"TRADE | {action} | {side} | {symbol} | "
        f"Price: ${price:.2f} | Qty: {quantity:.6f} | "
        f"Value: ${price * quantity:.2f}"
    )


def log_performance(
    logger: logging.Logger,
    portfolio_value: float,
    pnl: float,
    win_rate: float,
    sharpe_ratio: float
):
    """
    Log performance metrics
    
    Args:
        logger: Logger instance
        portfolio_value: Current portfolio value
        pnl: Profit and loss
        win_rate: Win rate percentage
        sharpe_ratio: Sharpe ratio
    """
    logger.info(
        f"PERFORMANCE | Portfolio: ${portfolio_value:.2f} | "
        f"PnL: ${pnl:.2f} ({pnl/10000*100:.2f}%) | "
        f"Win Rate: {win_rate:.2%} | "
        f"Sharpe: {sharpe_ratio:.2f}"
    )


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: str = "",
    include_traceback: bool = True
):
    """
    Log an error with context
    
    Args:
        logger: Logger instance
        error: Exception object
        context: Additional context
        include_traceback: Whether to include full traceback
    """
    if context:
        logger.error(f"ERROR in {context}: {str(error)}")
    else:
        logger.error(f"ERROR: {str(error)}")
    
    if include_traceback:
        logger.exception(error)


def create_session_log_file(base_name: str = 'session') -> str:
    """
    Create a timestamped log file for a session
    
    Args:
        base_name: Base name for the log file
        
    Returns:
        Path to the created log file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs/sessions')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'{base_name}_{timestamp}.log'
    return str(log_file)


def set_log_level(logger: logging.Logger, level: str):
    """
    Set logging level from string
    
    Args:
        logger: Logger instance
        level: Level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(log_level)


# Module-level convenience functions
_default_logger = None


def get_default_logger() -> logging.Logger:
    """Get or create the default logger"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger


def info(message: str):
    """Log info message to default logger"""
    get_default_logger().info(message)


def warning(message: str):
    """Log warning message to default logger"""
    get_default_logger().warning(message)


def error(message: str):
    """Log error message to default logger"""
    get_default_logger().error(message)


def debug(message: str):
    """Log debug message to default logger"""
    get_default_logger().debug(message)


def critical(message: str):
    """Log critical message to default logger"""
    get_default_logger().critical(message)


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    print("Testing logger setup...\n")
    
    # Create a test logger
    test_logger = setup_logger(
        name='test',
        log_file='logs/test.log',
        level=logging.DEBUG
    )
    
    # Test different log levels
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    # Test trade logging
    print("\nTesting trade logging...")
    log_trade(
        test_logger,
        action='OPEN',
        symbol='BTC/USDT',
        price=45000.00,
        quantity=0.1,
        side='BUY'
    )
    
    # Test performance logging
    print("\nTesting performance logging...")
    log_performance(
        test_logger,
        portfolio_value=12500.00,
        pnl=2500.00,
        win_rate=0.65,
        sharpe_ratio=1.85
    )
    
    # Test error logging
    print("\nTesting error logging...")
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        log_error_with_context(
            test_logger,
            error=e,
            context="test function",
            include_traceback=False
        )
    
    # Test multiple loggers
    print("\nTesting multiple loggers setup...")
    loggers = setup_trading_loggers()
    
    loggers['main'].info("Main logger working")
    loggers['trading'].info("Trading logger working")
    loggers['data'].info("Data logger working")
    loggers['training'].info("Training logger working")
    
    print("\n Logger tests complete!")
    print("Check the 'logs/' directory for output files")