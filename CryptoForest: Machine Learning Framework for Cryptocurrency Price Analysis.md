# Crypto Price Prediction Framework

A machine learning pipeline for cryptocurrency price forecasting with uncertainty estimation.

## Overview

This framework fetches historical cryptocurrency data, engineers technical features, trains multiple predictive models, and generates price forecasts with confidence intervals.

## Machine Learning Techniques

- **Ensemble Methods**: 
  - Random Forest: Uses tree-based ensemble learning with bagging (bootstrap aggregation)
  - Gradient Boosting: Implements sequential ensemble learning with gradient descent optimization

- **Uncertainty Quantification**:
  - For RF: Bootstrap sampling from multiple decision trees
  - For GBM: Weight-calibrated variance estimation from boosted trees

- **Feature Engineering**:
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Time series decomposition with Fourier analysis
  - Cyclical time encoding with trigonometric functions
  - Rolling window statistics with multiple lookback periods
