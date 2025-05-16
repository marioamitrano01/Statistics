# Ethereum Market Regime Detection

It is a project that identifies market regimes in Ethereum price data using Gaussian Mixture Models and detects significant change points between regimes.

## Features
- Takes Ethereum/USDT data from Binance with customizable timeframes
- Identifies Low, Medium, and High volatility regimes using GMM
- Detects change points between regimes with confidence metrics
- Generates visualizations of price, volatility, and return distributions
- Provides statistical analysis of each market regime

## Key Results
- **Low Volatility Regime** (93% of time): Avg Return 0.0009%, StdDev 0.37%
- **Medium Volatility Regime** (6% of time): Avg Return 0.0073%, StdDev 0.87%
- **High Volatility Regime** (1% of time): Avg Return 0.0325%, StdDev 1.85%

## Applications
- Trading strategy adaptation based on current regime
- Risk management with volatility-based position sizing
- Market analysis of transition patterns in crypto markets
