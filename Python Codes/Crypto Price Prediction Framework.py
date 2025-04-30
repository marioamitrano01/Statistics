import time
from functools import wraps
from typing import Tuple, List, Union, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIMER] {func.__name__} executed in {elapsed:.2f}s")
        return result
    return wrapper

class CryptoDataFetcher:

    def __init__(self, exchange_name='binance'):
        self.exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        self.data_cache = {}  
    @timer
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 1000, 
                    since: Optional[int] = None) -> pd.DataFrame:
        cache_key = f"{symbol}_{timeframe}_{limit}_{since}"
        
        if cache_key in self.data_cache:
            print(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]
            
        try:
            df = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
            df = pd.DataFrame(df, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df['range'] = df['high'] - df['low']
            df['mid_price'] = (df['high'] + df['low']) / 2
            
            self.data_cache[cache_key] = df
            
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise

    @timer
    def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        try:
            market_data = {}
            market_data['orderbook'] = self.exchange.fetch_order_book(symbol)
            market_data['trades'] = self.exchange.fetch_trades(symbol, limit=50)
            market_data['ticker'] = self.exchange.fetch_ticker(symbol)
            return market_data
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return {}

class FeatureEngineer:
    
    @staticmethod
    @timer
    def create_features(df: pd.DataFrame,
                        lags: List[int] = [1, 24, 168],
                        windows: List[int] = [24, 72, 168],
                        include_fourier: bool = True) -> pd.DataFrame:
        data = df.copy()
        
        data['log_return'] = np.log(data['close'] / data['close'].shift(1).replace(0, np.nan))
        data['pct_change'] = data['close'].pct_change()
        
        data['price_range'] = (data['high'] - data['low']) / data['close']  
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        data['ohlc4'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        
        data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
        data['high_low_ratio'] = data['high'] / data['low']
        
        data['volume_change'] = data['volume'].pct_change()
        data['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
        data['volume_price_ratio'] = data['volume'] / data['close']
        data['dollar_volume'] = data['close'] * data['volume']
        
        if isinstance(data.index, pd.DatetimeIndex):
            data['hour'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        for lag in lags:
            data[f'lag_{lag}'] = data['log_return'].shift(lag)
            data[f'price_mom_{lag}'] = data['close'].pct_change(lag)
            data[f'log_lag_price_{lag}'] = np.log(data['close'].shift(lag).replace(0, np.nan))
            
            if lag <= 24:  
                data[f'lag_{lag}_squared'] = data[f'lag_{lag}'] ** 2
        
        for w in windows:
            roll = data['log_return'].rolling(window=w)
            data[f'roll_mean_{w}'] = roll.mean()
            data[f'roll_std_{w}'] = roll.std()
            data[f'roll_p25_{w}'] = roll.quantile(0.25)
            data[f'roll_p75_{w}'] = roll.quantile(0.75)
            data[f'roll_iqr_{w}'] = data[f'roll_p75_{w}'] - data[f'roll_p25_{w}']
            
            price_roll = data['close'].rolling(window=w)
            data[f'price_roll_mean_{w}'] = price_roll.mean()
            min_w = data['low'].rolling(window=w).min()
            max_w = data['high'].rolling(window=w).max()
            range_w = np.maximum(max_w - min_w, 1e-8)
            data[f'price_pos_in_range_{w}'] = (data['close'] - min_w) / range_w
            
            data[f'volatility_{w}'] = data['log_return'].rolling(window=w).std() * np.sqrt(w)
            
            vol_roll = data['volume'].rolling(window=w)
            data[f'vol_trend_{w}'] = data['volume'] / vol_roll.mean()
            data[f'vol_std_{w}'] = vol_roll.std() / vol_roll.mean() 
            
            if w >= 12:  
                data[f'price_vol_corr_{w}'] = data['close'].rolling(window=w).corr(data['volume'])
        
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema12 - ema26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        data['macd_hist_change'] = data['macd_hist'].pct_change()
        
        delta = data['close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  
        
        for w in [14, 21]:
            avg_gain = gain.rolling(window=w).mean()
            avg_loss = loss.rolling(window=w).mean()
            rs = np.where(avg_loss == 0, 100, avg_gain / np.maximum(avg_loss, 1e-8))
            data[f'rsi_{w}'] = 100 - (100 / (1 + rs))
            
            data[f'rsi_{w}_diff'] = data[f'rsi_{w}'].diff()
            data[f'rsi_{w}_ma'] = data[f'rsi_{w}'].rolling(window=w//2).mean()
        
        for w in [20]:
            middle_band = data['close'].rolling(window=w).mean()
            std_dev = data['close'].rolling(window=w).std()
            data[f'bb_upper_{w}'] = middle_band + 2 * std_dev
            data[f'bb_lower_{w}'] = middle_band - 2 * std_dev
            data[f'bb_width_{w}'] = (data[f'bb_upper_{w}'] - data[f'bb_lower_{w}']) / middle_band
            data[f'bb_pos_{w}'] = (data['close'] - data[f'bb_lower_{w}']) / \
                             np.maximum(data[f'bb_upper_{w}'] - data[f'bb_lower_{w}'], 1e-8)
            data[f'bb_squeeze_{w}'] = std_dev / middle_band  
        
        high_9 = data['high'].rolling(window=9).max()
        low_9 = data['low'].rolling(window=9).min()
        data['tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = data['high'].rolling(window=26).max()
        low_26 = data['low'].rolling(window=26).min()
        data['kijun_sen'] = (high_26 + low_26) / 2
        
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
        
        high_52 = data['high'].rolling(window=52).max()
        low_52 = data['low'].rolling(window=52).min()
        data['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        data['chikou_span'] = data['close'].shift(-26)
        
        data['tenkan_kijun_cross'] = data['tenkan_sen'] - data['kijun_sen']
        data['cloud_thickness'] = data['senkou_span_a'] - data['senkou_span_b']
        
        if include_fourier and len(data) > 100:
            try:
                from scipy import fftpack
                
                fft_returns = fftpack.fft(data['log_return'].fillna(0).values)
                power_spectrum = np.abs(fft_returns) ** 2
                
                freq = fftpack.fftfreq(len(power_spectrum))
                mask = freq > 0  
                peaks = power_spectrum[mask].argsort()[-3:]  
                
                for i, peak in enumerate(peaks):
                    period = 1.0 / freq[mask][peak]
                    if np.isfinite(period) and period < len(data) / 2:  
                        t = np.arange(len(data))
                        data[f'fourier_sin_{i+1}'] = np.sin(2 * np.pi * t / period)
                        data[f'fourier_cos_{i+1}'] = np.cos(2 * np.pi * t / period)
            except Exception as e:
                print(f"Error in Fourier analysis: {e}")
        
        feat_cols = [c for c in data.columns if c not in ['open','high','low','close','volume']]
        for col in feat_cols:
            arr = data[col].values
            arr = np.where(np.isinf(arr), np.nan, arr)
            mean, std = np.nanmean(arr), np.nanstd(arr)
            if std == 0 or np.isnan(std):  
                data[f'norm_{col}'] = 0
            else:
                data[f'norm_{col}'] = (arr - mean) / (std + 1e-8)

        data = data.replace([np.inf, -np.inf], np.nan)
        return data.dropna()

class GradientBoostingModel:
    
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 5, 
                learning_rate: float = 0.1, subsample: float = 0.8,
                loss: str = 'squared_error'):
        
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            loss=loss,
            random_state=42
        )
        
        self.y_mean = None
        self.y_std = None
        self.feature_importances_ = None

    @timer
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Training data contains NaN values")
        if np.isinf(X).any() or np.isinf(y).any():
            raise ValueError("Training data contains infinite values")
            
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        
        print(f"Training GBM with {X.shape[0]} samples and {X.shape[1]} features")
        self.model.fit(X, y)
        
        self.feature_importances_ = self.model.feature_importances_
        
        if hasattr(self, 'feature_names') and len(self.feature_names) == X.shape[1]:
            print("\nTop 10 features by importance:")
            indices = np.argsort(self.feature_importances_)[::-1]
            for i in range(min(10, X.shape[1])):
                print(f"{i+1}. {self.feature_names[indices[i]]}: {self.feature_importances_[indices[i]]:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        
        if np.isnan(X).any():
            raise ValueError("Test data contains NaN values")
        if np.isinf(X).any():
            raise ValueError("Test data contains infinite values")
            
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray, quantiles: List[float] = [0.05, 0.95]) -> Tuple[np.ndarray, np.ndarray]:
        
        X = np.asarray(X, dtype=np.float64)
        
        if np.isnan(X).any():
            raise ValueError("Test data contains NaN values")
        if np.isinf(X).any():
            raise ValueError("Test data contains infinite values")
            
        y_pred = self.model.predict(X)
        
        if hasattr(self.model, 'estimators_'):
            tree_preds = np.array([est[0].predict(X) for est in self.model.estimators_])
            
            lr = self.model.learning_rate
            multipliers = np.array([lr * (1.0 - lr) ** i for i in range(len(self.model.estimators_))])
            multipliers = multipliers / np.sum(multipliers) 
            
            weighted_mean = np.sum(tree_preds * multipliers[:, np.newaxis], axis=0)
            weighted_variance = np.sum(multipliers[:, np.newaxis] * (tree_preds - weighted_mean) ** 2, axis=0)
            y_std = np.sqrt(weighted_variance)
        else:
            y_std = 0.05 * np.abs(y_pred)
        
        return y_pred, y_std
        
    def predict_future(self, last_X: np.ndarray, feature_cols: List[str], 
                       n_steps: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        
        if last_X.ndim == 1:
            last_X = last_X.reshape(1, -1)
            
        X_future = last_X.copy()
        predictions = []
        uncertainties = []
        
        self.feature_names = feature_cols
        
        for i in range(n_steps):
            pred, std = self.predict_with_uncertainty(X_future)
            predictions.append(pred[0])
            uncertainties.append(std[0])
            
            if i == n_steps-1:
                break
                
            X_new = X_future.copy()
            
            
            for j, col in enumerate(feature_cols):
                if 'lag_' in col and col.replace('norm_', '').replace('lag_', 'lag_') in feature_cols:
                    lag_num = int(col.split('_')[-1])
                    if lag_num > 1:
                        prev_lag = col.replace(f'lag_{lag_num}', f'lag_{lag_num-1}')
                        prev_idx = feature_cols.index(prev_lag)
                        X_new[0, j] = X_future[0, prev_idx]
            
            pred_return = np.log(pred[0] / self.y_mean) if self.y_mean is not None else 0
            
            for j, col in enumerate(feature_cols):
                if 'log_return' in col:
                    X_new[0, j] = pred_return
                    
            X_future = X_new
                    
        return np.array(predictions), np.array(uncertainties)

class RandomForestModel:
    
    def __init__(self, n_estimators: int = 200, max_depth: Union[int, None] = 10, 
                 min_samples_leaf: int = 2, max_features: str = 'sqrt'):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1  
        )

    @timer
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Training data contains NaN values")
        if np.isinf(X).any() or np.isinf(y).any():
            raise ValueError("Training data contains infinite values")
            
        self.model.fit(X, y)

    @timer
    def predict(self, X: np.ndarray) -> np.ndarray:
        if np.isnan(X).any():
            raise ValueError("Test data contains NaN values")
        if np.isinf(X).any():
            raise ValueError("Test data contains infinite values")
            
        return self.model.predict(X)
        
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0)
        return mean_pred, std_pred
        
    def predict_future(self, last_X: np.ndarray, feature_cols: List[str], 
                      n_steps: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if last_X.ndim == 1:
            last_X = last_X.reshape(1, -1)
            
        X_future = last_X.copy()
        predictions = []
        uncertainties = []
        
        for i in range(n_steps):
            pred, std = self.predict_with_uncertainty(X_future)
            predictions.append(pred[0])
            uncertainties.append(std[0])
            
            if i == n_steps-1:
                break
                
            X_new = X_future.copy()
            
            for j, col in enumerate(feature_cols):
                if 'lag_' in col and col.replace('norm_', '').replace('lag_', 'lag_') in feature_cols:
                    lag_num = int(col.split('_')[-1])
                    if lag_num > 1:
                        prev_lag = col.replace(f'lag_{lag_num}', f'lag_{lag_num-1}')
                        if prev_lag in feature_cols:
                            prev_idx = feature_cols.index(prev_lag)
                            X_new[0, j] = X_future[0, prev_idx]
            
            last_price = pred[0]  
            pred_return = (last_price / predictions[-2]) - 1 if len(predictions) > 1 else 0
            
            for j, col in enumerate(feature_cols):
                if 'return' in col:
                    X_new[0, j] = pred_return
                    
            X_future = X_new
                    
        return np.array(predictions), np.array(uncertainties)

class ModelEvaluator:
    
    @staticmethod
    @timer
    def evaluate_model(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_std: Optional[np.ndarray] = None) -> Dict[str, float]:

        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = mape if np.isfinite(mape) else np.nan
        
        true_direction = np.sign(np.diff(np.append([y_true[0]], y_true)))
        pred_direction = np.sign(np.diff(np.append([y_true[0]], y_pred)))
        dir_accuracy = np.mean(true_direction == pred_direction)
        metrics['direction_accuracy'] = dir_accuracy * 100
        
        if y_std is not None:
            lower_bound = y_pred - 1.96 * y_std
            upper_bound = y_pred + 1.96 * y_std
            in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
            metrics['95ci_coverage'] = np.mean(in_interval) * 100
            
            metrics['mean_uncertainty'] = np.mean(y_std)
            metrics['uncertainty_ratio'] = metrics['mean_uncertainty'] / np.std(y_true)
            
        # Print results
        print(f"\n- {model_name} Performance -")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        return metrics

class Visualizer:
   
    @staticmethod
    @timer
    def plot_prediction(index: pd.DatetimeIndex,
                        y_true: np.ndarray,
                        preds: dict,
                        future_index: Optional[pd.DatetimeIndex] = None,
                        future_preds: Optional[dict] = None) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1, 
            row_heights=[0.7, 0.3],
            subplot_titles=["Price Predictions", "Prediction Errors"],
            vertical_spacing=0.1
        )
        
        fig.add_trace(go.Scatter(
            x=index, 
            y=y_true, 
            mode='markers', 
            name='Actual', 
            marker=dict(
                size=8,
                color='rgba(0, 0, 255, 0.7)',
                symbol='circle'
            ),
            hovertemplate='%{x}<br>Price: %{y:.2f} USDT'
        ), row=1, col=1)
        
        colors = {
            'GBM': 'rgba(220, 20, 60, 0.9)',  
            'RF': 'rgba(148, 0, 211, 0.9)',  
            'RF (standard)': 'rgba(60, 179, 113, 0.9)',  
            'RF (complex)': 'rgba(255, 165, 0, 0.9)',     
            'RF (regularized)': 'rgba(106, 90, 205, 0.9)'  
        }
        
        metrics = {}
        
        for i, (label, (y_pred, y_std)) in enumerate(preds.items()):
            model_color = colors.get(label, f'rgba({50*i}, {155}, {50}, 0.9)')
            
            fig.add_trace(go.Scatter(
                x=index, 
                y=y_pred, 
                mode='lines', 
                name=f"{label} Prediction",
                line=dict(
                    color=model_color,
                    width=2,
                    dash='solid'
                ),
                hovertemplate='%{x}<br>' + f'{label} Prediction: ' + '%{y:.2f} USDT'
            ), row=1, col=1)
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                if not np.isfinite(mape):
                    mape = np.nan
            
            metrics[label] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'MAE': mae,
                'MAPE': mape
            }
            
            fig.add_trace(go.Scatter(
                x=index,
                y=y_pred - y_true,
                mode='lines',
                name=f"{label} Error",
                line=dict(color=model_color, width=1),
                hovertemplate='%{x}<br>Error: %{y:.2f} USDT'
            ), row=2, col=1)
            
            if y_std is not None:
                fill_color = model_color.replace('0.9', '0.2')
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([index, index[::-1]]),
                    y=np.concatenate([y_pred - 2*y_std, (y_pred + 2*y_std)[::-1]]),
                    fill='toself', 
                    name=f"{label} ±2σ",
                    line=dict(color='rgba(255,255,255,0)'),
                    fillcolor=fill_color,
                    hoverinfo='none'
                ), row=1, col=1)
        
        if future_index is not None and future_preds is not None:
            
            last_date_str = index[-1].strftime('%Y-%m-%d %H:%M:%S')
            
            fig.add_shape(
                type="line",
                x0=last_date_str,
                y0=0,
                x1=last_date_str, 
                y1=1,
                yref="paper",
                line=dict(color="gray", width=1, dash="dash"),
            )
            
            fig.add_annotation(
                x=last_date_str,
                y=1,
                yref="paper",
                text="Forecast Start",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=12)
            )
            
            for label, (future_y, future_std) in future_preds.items():
                model_color = colors.get(label, f'rgba({50*len(preds)}, {155}, {50}, 0.9)')
                
                fig.add_trace(go.Scatter(
                    x=future_index,
                    y=future_y,
                    mode='lines',
                    name=f"{label} Forecast",
                    line=dict(
                        color=model_color,
                        width=3,
                        dash='dot'
                    ),
                    hovertemplate='%{x}<br>' + f'{label} Forecast: ' + '%{y:.2f} USDT'
                ), row=1, col=1)
                
                if future_std is not None:
                    fill_color = model_color.replace('0.9', '0.15')  
                    
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([future_index, future_index[::-1]]),
                        y=np.concatenate([future_y - 2*future_std, (future_y + 2*future_std)[::-1]]),
                        fill='toself',
                        name=f"{label} Forecast ±2σ",
                        line=dict(color='rgba(255,255,255,0)'),
                        fillcolor=fill_color,
                        hoverinfo='none'
                    ), row=1, col=1)
        
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=3, label="3d", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            row=1, col=1
        )
        
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black', row=2, col=1)
        
        annotations = []
        y_pos = 1.12
        
        annotations.append(
            dict(
                x=0.5,
                y=1.18,
                xref='paper',
                yref='paper',
                text="Model Performance Metrics",
                showarrow=False,
                font=dict(size=14, color='black')
            )
        )
        
        for i, (label, metric) in enumerate(metrics.items()):
            model_color = colors.get(label, 'black')
            metric_text = f"{label} | RMSE: {metric['RMSE']:.2f} | MAE: {metric['MAE']:.2f}"
            if np.isfinite(metric['MAPE']):
                metric_text += f" | MAPE: {metric['MAPE']:.2f}%"
                
            annotations.append(
                dict(
                    x=0.0 if i % 2 == 0 else 0.5,
                    y=y_pos - 0.05 * (i // 2),
                    xref='paper',
                    yref='paper',
                    text=metric_text,
                    showarrow=False,
                    font=dict(size=10, color=model_color)
                )
            )
        
        fig.update_layout(
            title='Cryptocurrency Price Prediction with Uncertainty Estimation',
            xaxis_title='Time',
            yaxis_title='Price (USDT)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=annotations,
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=120, b=0),
        )
        
        return fig
    
    @staticmethod
    @timer
    def plot_feature_importance(feature_names: List[str], 
                                importance_values: Dict[str, np.ndarray],
                                top_n: int = 15) -> go.Figure:
        fig = go.Figure()
        
        for model_name, importances in importance_values.items():
            if len(importances) != len(feature_names):
                print(f"Warning: Feature importance length mismatch for {model_name}")
                continue
                
            indices = np.argsort(importances)[-top_n:]
            
            fig.add_trace(go.Bar(
                y=[feature_names[i] for i in indices],
                x=[importances[i] for i in indices],
                name=model_name,
                orientation='h',
                marker=dict(
                    line=dict(width=1)
                )
            ))
        
        fig.update_layout(
            title='Top Feature Importance by Model',
            xaxis_title='Importance',
            yaxis_title='Features',
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            height=500 + 15 * min(top_n, 20)  
        )
        
        return fig
    

class ForecastManager:
    
    def __init__(self, trained_models: Dict[str, Any], feature_cols: List[str]):
        self.models = trained_models
        self.feature_cols = feature_cols
        
    @timer
    def generate_future_timeframes(self, last_timestamp: pd.Timestamp, 
                                  timeframe: str, steps: int) -> pd.DatetimeIndex:
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            freq = f'{minutes}min'
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            freq = f'{hours}H'
        elif timeframe.endswith('d'):
            days = int(timeframe[:-1])
            freq = f'{days}D'
        else:
            raise ValueError(f"Unsupported timeframe format: {timeframe}")
            
        future_times = pd.date_range(
            start=last_timestamp + pd.Timedelta(freq),
            periods=steps,
            freq=freq
        )
        
        return future_times
        
    @timer
    def forecast(self, last_X: np.ndarray, last_timestamp: pd.Timestamp,
                timeframe: str, steps: int = 24) -> Tuple[pd.DatetimeIndex, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Generate forecasts for all models"""
        future_index = self.generate_future_timeframes(last_timestamp, timeframe, steps)
        future_predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_future'):
                try:
                    y_pred, y_std = model.predict_future(last_X, self.feature_cols, steps)
                    future_predictions[name] = (y_pred, y_std)
                except Exception as e:
                    print(f"Error forecasting with {name}: {e}")
        
        return future_index, future_predictions

def select_best_features(df: pd.DataFrame, target_col: str, max_features: int = 50) -> List[str]:
    from scipy.stats import spearmanr
    
    feature_cols = [c for c in df.columns if c.startswith('norm_')]
    
    correlations = []
    for col in feature_cols:
        corr, _ = spearmanr(df[col], df[target_col], nan_policy='omit')
        correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    selected = []
    for col, corr in correlations:
        if len(selected) >= max_features:
            break
        
        if not selected:
            selected.append(col)
            continue
        
        multicollinear = False
        for sel in selected:
            corr_between, _ = spearmanr(df[col], df[sel], nan_policy='omit')
            if abs(corr_between) > 0.85:  
                multicollinear = True
                break
                
        if not multicollinear:
            selected.append(col)
    
    print(f"Selected {len(selected)} features from {len(feature_cols)} candidates")
    return selected

def run_forecast_pipeline(symbol: str = 'BTC/USDT', timeframe: str = '15m', 
                         forecast_steps: int = 48, data_limit: int = 750):
    try:
        print(f"Starting forecast pipeline for {symbol} on {timeframe} timeframe")
        
        fetcher = CryptoDataFetcher()
        
        raw_df = fetcher.fetch_ohlcv(symbol, timeframe, limit=data_limit)
        print(f"Fetched {len(raw_df)} rows of {timeframe} data")
        
        daily_df = fetcher.fetch_ohlcv(symbol, '1d', limit=60)
        print(f"Fetched {len(daily_df)} rows of daily data")
        
        time_window_map = {
            '1m': {'lags': [5, 15, 60], 'windows': [15, 60, 240]},
            '5m': {'lags': [3, 12, 72], 'windows': [12, 72, 288]},
            '15m': {'lags': [4, 16, 96], 'windows': [16, 96, 288]},
            '1h': {'lags': [1, 6, 24], 'windows': [6, 24, 168]},
            '4h': {'lags': [1, 6, 24], 'windows': [6, 24, 42]},
            '1d': {'lags': [1, 5, 20], 'windows': [5, 20, 60]}
        }
        
        settings = time_window_map.get(timeframe, {'lags': [4, 16, 96], 'windows': [16, 96, 288]})
        
        df_feat = FeatureEngineer.create_features(
            raw_df, 
            lags=settings['lags'],
            windows=settings['windows'],
            include_fourier=True
        )
        print(f"Data shape after feature engineering: {df_feat.shape}")
        
        if len(daily_df) > 30:  
            print("Adding daily trend features...")
            daily_df['daily_trend'] = daily_df['close'].pct_change(5)
            daily_df['daily_volatility'] = daily_df['close'].rolling(5).std() / daily_df['close']
            daily_df['weekend_effect'] = daily_df.index.dayofweek.isin([5, 6]).astype(float)
            
            daily_features = daily_df[['daily_trend', 'daily_volatility', 'weekend_effect']].copy()
            new_idx = pd.date_range(
                start=daily_features.index.min(),
                end=daily_features.index.max() + pd.Timedelta(days=1),
                freq=timeframe
            )
            daily_features = daily_features.reindex(new_idx, method='ffill')
            
            common_dates = df_feat.index.intersection(daily_features.index)
            if len(common_dates) > 0:
                for col in daily_features.columns:
                    df_feat.loc[common_dates, col] = daily_features.loc[common_dates, col]
                    arr = df_feat[col].values
                    arr = np.where(np.isinf(arr), np.nan, arr)
                    mean, std = np.nanmean(arr), np.nanstd(arr)
                    if std > 0:
                        df_feat[f'norm_{col}'] = (arr - mean) / (std + 1e-8)
        
        nan_check = df_feat.isna().sum().sum()
        if nan_check > 0:
            print(f"Warning: {nan_check} NaN values remaining after preprocessing")
            df_feat = df_feat.dropna()
            
        if len(df_feat) < 50:
            raise ValueError(f"Not enough data after preprocessing: {len(df_feat)} rows")
            
        feature_cols = select_best_features(df_feat, 'close', max_features=40)  # Reduced from 50
        
        X = df_feat[feature_cols].values
        y = df_feat['close'].values
        time_index = df_feat.index
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        t_train, t_test = time_index[:split], time_index[split:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        models = {}
        
        print("\nTraining Random Forest models...")
        rf1 = RandomForestModel(n_estimators=200, max_depth=12)
        rf1.train(X_train, y_train)
        models['RF (standard)'] = rf1
        
        rf2 = RandomForestModel(n_estimators=100, max_depth=15)
        rf2.train(X_train, y_train)
        models['RF (complex)'] = rf2
        
        rf3 = RandomForestModel(n_estimators=150, max_depth=8)
        rf3.train(X_train, y_train)
        models['RF (regularized)'] = rf3
        
        print("\nTraining Gradient Boosting model...")
        try:
            gbm = GradientBoostingModel(n_estimators=150, max_depth=5)
            gbm.train(X_train, y_train)
            models['GBM'] = gbm
        except Exception as e:
            print(f"Error training GBM model: {e}")
        
        predictions = {}
        importance_values = {}
        
        for name, model in models.items():
            print(f"\nPredicting with {name}...")
            
            if name == 'GBM':
                if hasattr(model, 'predict_with_uncertainty'):
                    y_pred, y_std = model.predict_with_uncertainty(X_test)
                else:
                    y_pred = model.predict(X_test)
                    y_std = None
                predictions[name] = (y_pred, y_std)
            else:
                if hasattr(model, 'predict_with_uncertainty'):
                    y_pred, y_std = model.predict_with_uncertainty(X_test)
                    predictions[name] = (y_pred, y_std)
                else:
                    y_pred = model.predict(X_test)
                    
                    if hasattr(model.model, 'estimators_'):
                        tree_preds = np.array([tree.predict(X_test) for tree in model.model.estimators_])
                        y_std = np.std(tree_preds, axis=0)
                    else:
                        y_std = None
                        
                    predictions[name] = (y_pred, y_std)
            
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                importance_values[name] = model.feature_importances_
            elif hasattr(model.model, 'feature_importances_'):
                importance_values[name] = model.model.feature_importances_
        
        for name, (y_pred, y_std) in predictions.items():
            ModelEvaluator.evaluate_model(name, y_test, y_pred, y_std)
        
        if importance_values:
            try:
                feat_importance_fig = Visualizer.plot_feature_importance(feature_cols, importance_values)
                feat_importance_fig.show()
            except Exception as e:
                print(f"Could not plot feature importance: {e}")
        
        print(f"\nGenerating {forecast_steps} future predictions...")
        forecast_manager = ForecastManager(models, feature_cols)
        future_index, future_predictions = forecast_manager.forecast(
            X_test[-1:], t_test[-1], timeframe, steps=forecast_steps
        )
        
        fig = Visualizer.plot_prediction(
            t_test, y_test, predictions, 
            future_index, future_predictions
        )
        
        fig.update_layout(
            title=f"{symbol} Prediction & {forecast_steps}-Step Forecast ({timeframe} timeframe)",
            template="plotly_white"
        )
        fig.show()
        
        if future_predictions:
            print("\nForecast Summary:")
            for model_name, (preds, _) in future_predictions.items():
                price_change = (preds[-1] / preds[0] - 1) * 100
                direction = "UP" if price_change > 0 else "DOWN"
                print(f"{model_name}: Predicts {direction} {abs(price_change):.2f}% " +
                      f"(${preds[0]:.2f} → ${preds[-1]:.2f})")
        
        return {
            'data': df_feat,
            'models': models,
            'predictions': predictions,
            'future_predictions': future_predictions,
            'future_index': future_index,
            'feature_cols': feature_cols
        }
        
    except Exception as e:
        print(f"Error in forecast pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    try:
        symbol = 'BTC/USDT'
        timeframe = '15m'
        forecast_steps = 48  
        
        results = run_forecast_pipeline(
            symbol=symbol,
            timeframe=timeframe,
            forecast_steps=forecast_steps,
            data_limit=1000
        )
        
        if results:
            print("\nPipeline completed successfully!")
            print(f"Generated {forecast_steps} future predictions for {symbol}")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
