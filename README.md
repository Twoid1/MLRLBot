# 🤖 Kraken Trading Bot - Hybrid ML/RL System

A production-grade cryptocurrency trading bot combining Machine Learning for market prediction with Reinforcement Learning for optimal trade execution. Built from scratch with zero external dependencies for maximum control and reliability.

## 🎯 Overview

This bot implements a dual-brain architecture:
- **Brain 1 (ML)**: XGBoost/LightGBM for market direction prediction
- **Brain 2 (RL)**: Deep Q-Network for trade execution optimization
- **Multi-Asset Learning**: Trains on 15 crypto assets across 6 timeframes
- **Transfer Learning**: Patterns learned on BTC improve trading on other assets

## 📊 Supported Assets

- BTC/USDT, ETH/USDT, BNB/USDT, ADA/USDT, SOL/USDT
- XRP/USDT, DOT/USDT, DOGE/USDT, AVAX/USDT, MATIC/USDT
- LINK/USDT, LTC/USDT, ATOM/USDT, UNI/USDT, ALGO/USDT

## ⏰ Timeframes

- 5m, 15m, 1h, 4h, 1d, 1w

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 16GB RAM (32GB recommended)
- 100GB storage for data
- Kraken API keys (for live trading)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kraken-trading-bot.git
cd kraken-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (platform specific)
# Ubuntu/Debian:
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# macOS:
brew install ta-lib

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

1. Edit `config.yaml` to customize:
   - Trading pairs and timeframes
   - ML/RL parameters
   - Risk management settings
   - Performance targets

2. Set up your Kraken API keys in `.env`:
```
KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here
```

## 🏃 Running the Bot

### Data Preparation

```bash
# Process existing OHLCV data
python -m src.data.data_manager --process

# Calculate features
python -m src.features.feature_engineer --calculate
```

### Training

```bash
# Train ML model
python -m src.models.ml_predictor --train

# Train RL agent
python -m src.models.dqn_agent --train
```

### Backtesting

```bash
# Run backtest
python -m src.backtesting.backtester --run

# Walk-forward analysis
python -m src.backtesting.walk_forward --analyze
```

### Paper Trading

```bash
# Start paper trading
python -m src.live.paper_trader --start
```

### Live Trading

```bash
# Start live trading (use with caution!)
python -m src.live.live_trader --start
```

### Monitoring Dashboard

```bash
# Launch dashboard
python -m dashboard.app
# Open browser to http://localhost:8050
```

## 📁 Project Structure

```
kraken-trading-bot/
├── src/               # Source code
│   ├── data/         # Data management
│   ├── features/     # Feature engineering
│   ├── environment/  # Trading environment
│   ├── models/       # ML/RL models
│   ├── trading/      # Trading logic
│   ├── backtesting/  # Backtesting engine
│   ├── live/         # Live trading
│   └── utils/        # Utilities
├── data/             # Data storage
├── models/           # Saved models
├── notebooks/        # Jupyter notebooks
├── tests/            # Unit tests
├── logs/             # Log files
└── dashboard/        # Web dashboard
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_models/

# Run with coverage
pytest --cov=src tests/
```

## 📈 Performance Targets

- **Sharpe Ratio**: > 1.5
- **Max Drawdown**: < 15%
- **Win Rate**: > 45%
- **Annual Return**: > 20%

## 🛡️ Risk Management

The bot implements multiple safety layers:
- Kelly Criterion position sizing (25% fraction)
- Stop-loss and take-profit orders
- Maximum drawdown limits
- Daily loss limits
- Circuit breakers for consecutive losses
- Position size limits

## 📊 Features

### Technical Indicators (50 selected from 100+)
- **Momentum**: RSI, MACD, Stochastic, CCI, Williams %R
- **Volatility**: ATR, Bollinger Bands, Keltner Channels
- **Volume**: OBV, AD, VWAP, Volume ratios
- **Trend**: ADX, Aroon, PSAR, Multiple MAs
- **Patterns**: Candlestick patterns, Support/Resistance

### Multi-Timeframe Analysis
- Trend alignment across timeframes
- RSI confluence
- Volume patterns
- Volatility regime detection

## 🔧 Development Roadmap

- [x] Project setup and structure
- [ ] Data pipeline implementation
- [ ] Feature engineering system
- [ ] ML predictor development
- [ ] RL agent implementation
- [ ] Risk management system
- [ ] Backtesting framework
- [ ] Paper trading system
- [ ] Live trading deployment
- [ ] Monitoring dashboard

## 📝 Documentation

- [Architecture Overview](docs/architecture.md)
- [Feature Engineering](docs/features.md)
- [ML Model Details](docs/ml_model.md)
- [RL Agent Details](docs/rl_agent.md)
- [Risk Management](docs/risk_management.md)
- [API Reference](docs/api_reference.md)

## ⚠️ Disclaimer

**IMPORTANT**: This bot is for educational purposes. Cryptocurrency trading carries significant risk. Never trade with money you cannot afford to lose. Always test thoroughly in paper trading before considering live deployment.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built from scratch for complete control
- Inspired by modern quantitative trading strategies
- Designed for production reliability

## 📞 Support

- Create an issue for bug reports
- Discussions for feature requests
- Wiki for documentation

---

**Version**: 1.0.0  
**Status**: Development  
**Last Updated**: 2024