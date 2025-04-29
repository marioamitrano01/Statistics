import time
from functools import wraps
from typing import Tuple, List, Union
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import ccxt
import plotly.graph_objects as go

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

    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})

    @timer
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 1000) -> pd.DataFrame:
        df = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(df, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

class FeatureEngineer:
    
    @staticmethod
    @timer
    def create_features(df: pd.DataFrame,
                        lags: List[int] = [1, 24, 168],
                        windows: List[int] = [24, 72, 168]) -> pd.DataFrame:
        data = df.copy()
        
        data['log_return'] = np.log(data['close'] / data['close'].shift(1).replace(0, np.nan))
        data['pct_change'] = data['close'].pct_change()
        
        data['price_range'] = (data['high'] - data['low']) / data['close']  
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3  
        
        data['volume_change'] = data['volume'].pct_change()
        data['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
        
        for lag in lags:
            data[f'lag_{lag}'] = data['log_return'].shift(lag)
            data[f'price_mom_{lag}'] = data['close'].pct_change(lag)
        
        for w in windows:
            roll = data['log_return'].rolling(window=w)
            data[f'roll_mean_{w}'] = roll.mean()
            data[f'roll_std_{w}'] = roll.std()
            data[f'roll_p25_{w}'] = roll.quantile(0.25)
            data[f'roll_p75_{w}'] = roll.quantile(0.75)
            
            price_roll = data['close'].rolling(window=w)
            data[f'price_roll_mean_{w}'] = price_roll.mean()
            min_w = data['low'].rolling(window=w).min()
            max_w = data['high'].rolling(window=w).max()
            range_w = np.maximum(max_w - min_w, 1e-8)
            data[f'price_pos_in_range_{w}'] = (data['close'] - min_w) / range_w
            
            data[f'volatility_{w}'] = data['log_return'].rolling(window=w).std() * np.sqrt(w)
            
            vol_roll = data['volume'].rolling(window=w)
            data[f'vol_trend_{w}'] = data['volume'] / vol_roll.mean()
        
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema12 - ema26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
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
        
        for w in [20]:
            middle_band = data['close'].rolling(window=w).mean()
            std_dev = data['close'].rolling(window=w).std()
            data[f'bb_upper_{w}'] = middle_band + 2 * std_dev
            data[f'bb_lower_{w}'] = middle_band - 2 * std_dev
            data[f'bb_width_{w}'] = (data[f'bb_upper_{w}'] - data[f'bb_lower_{w}']) / middle_band
            data[f'bb_pos_{w}'] = (data['close'] - data[f'bb_lower_{w}']) / \
                             np.maximum(data[f'bb_upper_{w}'] - data[f'bb_lower_{w}'], 1e-8)
        
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

class GaussianProcessModel:
    
    def __init__(self, kernel=None, n_restarts: int = 2, alpha: float = 1e-8):
        if kernel is None:
            kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=42,
            alpha=alpha  
        )

    @timer
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Training data contains NaN values")
        if np.isinf(X).any() or np.isinf(y).any():
            raise ValueError("Training data contains infinite values")
            
        y_mean, y_std = np.mean(y), np.std(y)
        if y_std == 0:
            y_std = 1
        self.y_mean = y_mean
        self.y_std = y_std
        y_scaled = (y - y_mean) / y_std
        
        print(f"Training GP with {X.shape[0]} samples and {X.shape[1]} features")
        self.model.fit(X, y_scaled)
        print(f"Optimized kernel: {self.model.kernel_}")

    @timer
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        
        if np.isnan(X).any():
            raise ValueError("Test data contains NaN values")
        if np.isinf(X).any():
            raise ValueError("Test data contains infinite values")
        
        y_pred_scaled, y_std_scaled = self.model.predict(X, return_std=True)
        
        y_pred = y_pred_scaled * self.y_std + self.y_mean
        y_std = y_std_scaled * self.y_std
            
        return y_pred, y_std

class RandomForestModel:
    
    def __init__(self, n_estimators: int = 200, max_depth: Union[int, None] = 10):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
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

class Visualizer:
   
    @staticmethod
    @timer
    def plot_prediction(index: pd.DatetimeIndex,
                        y_true: np.ndarray,
                        preds: dict) -> go.Figure:
        fig = go.Figure()
        
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
        ))
        
        colors = {
            'GP': 'rgba(220, 20, 60, 0.9)',  
            'RF': 'rgba(148, 0, 211, 0.9)'   
        }
        
        metrics = {}
        
        for i, (label, (y_pred, y_std)) in enumerate(preds.items()):
            color = colors.get(label, f'rgba({50*i}, {155}, {50}, 0.9)')
            
            fig.add_trace(go.Scatter(
                x=index, 
                y=y_pred, 
                mode='lines', 
                name=label,
                line=dict(
                    color=color,
                    width=2,
                    dash='solid'
                ),
                hovertemplate='%{x}<br>' + f'{label} Prediction: ' + '%{y:.2f} USDT'
            ))
            
            mse = mean_squared_error(y_true, y_pred)
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics[label] = {
                'MSE': mse,
                'MAE': mae,
                'MAPE': mape
            }
            
            if y_std is not None:
                fill_color = color.replace('0.9', '0.2')
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([index, index[::-1]]),
                    y=np.concatenate([y_pred - 2*y_std, (y_pred + 2*y_std)[::-1]]),
                    fill='toself', 
                    name=f"{label} ±2σ",
                    line=dict(color='rgba(255,255,255,0)'),
                    fillcolor=fill_color,
                    hoverinfo='none'
                ))
        
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
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
                font=dict(size=14)
            )
        )
        
        for label, metric in metrics.items():
            annotations.append(
                dict(
                    x=0.0 if label == 'GP' else 0.5,
                    y=y_pos,
                    xref='paper',
                    yref='paper',
                    text=f"{label} | MSE: {metric['MSE']:.2f} | MAE: {metric['MAE']:.2f} | MAPE: {metric['MAPE']:.2f}%",
                    showarrow=False,
                    font=dict(size=10, color=colors.get(label, 'black'))
                )
            )
        
        fig.update_layout(
            title='Model Confrontation: GP vs RF on Crypto Prices',
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
            margin=dict(t=120),  
        )
        
        return fig

if __name__ == '__main__':
    try:
        fetcher = CryptoDataFetcher()
        
        raw_df = fetcher.fetch_ohlcv('BTC/USDT', '15m', limit=500)
        print(f"Fetched {len(raw_df)} rows of 15-minute data")
        
        daily_df = fetcher.fetch_ohlcv('BTC/USDT', '1d', limit=60)
        
        df_feat = FeatureEngineer.create_features(
            raw_df, 
            lags=[4, 16, 96],  
            windows=[16, 96, 288]   
        )
        print(f"Data shape after feature engineering: {df_feat.shape}")
        
        if len(daily_df) > 30:  
            print("Adding daily trend features...")
            daily_df['daily_trend'] = daily_df['close'].pct_change(5)  
            daily_df['daily_volatility'] = daily_df['close'].rolling(5).std() / daily_df['close']
            
            daily_features = daily_df[['daily_trend', 'daily_volatility']].copy()
            new_idx = pd.date_range(
                start=daily_features.index.min(),
                end=daily_features.index.max() + pd.Timedelta(days=1),
                freq='15min'
            )
            daily_features = daily_features.reindex(new_idx, method='ffill')
            
            common_dates = df_feat.index.intersection(daily_features.index)
            if len(common_dates) > 0:
                for col in daily_features.columns:
                    df_feat.loc[common_dates, col] = daily_features.loc[common_dates, col]
                for col in ['daily_trend', 'daily_volatility']:
                    if col in df_feat.columns:
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
            
        feature_cols = [c for c in df_feat.columns if c.startswith('norm_')]
        
        X = df_feat[feature_cols].values
        y = df_feat['close'].values
        time_index = df_feat.index
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        t_test = time_index[split:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        print("Training Random Forest models...")
        
        rf1 = RandomForestModel(n_estimators=300, max_depth=12)
        rf1.train(X_train, y_train)
        y_rf1 = rf1.predict(X_test)
        
        rf2 = RandomForestModel(n_estimators=400, max_depth=15)
        rf2.train(X_train, y_train)
        y_rf2 = rf2.predict(X_test)
        
        rf3 = RandomForestModel(n_estimators=200, max_depth=8)
        rf3.train(X_train, y_train)
        y_rf3 = rf3.predict(X_test)
        
        if hasattr(rf1.model, 'feature_importances_'):
            importances = rf1.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print("\nFeature importance ranking:")
            for i, idx in enumerate(indices[:10]):  
                if i < len(feature_cols):
                    print(f"{i+1}. {feature_cols[idx]} ({importances[idx]:.4f})")
        
        rf1_mse = mean_squared_error(y_test, y_rf1)
        rf1_r2 = r2_score(y_test, y_rf1)
        print(f"RF1 (standard) -> MSE: {rf1_mse:.4f}, R2: {rf1_r2:.4f}")
        
        rf2_mse = mean_squared_error(y_test, y_rf2)
        rf2_r2 = r2_score(y_test, y_rf2)
        print(f"RF2 (complex) -> MSE: {rf2_mse:.4f}, R2: {rf2_r2:.4f}")
        
        rf3_mse = mean_squared_error(y_test, y_rf3)
        rf3_r2 = r2_score(y_test, y_rf3)
        print(f"RF3 (regularized) -> MSE: {rf3_mse:.4f}, R2: {rf3_r2:.4f}")
        
        y_rf_values = np.array([tree.predict(X_test) for tree in rf1.model.estimators_])
        y_rf_std = np.std(y_rf_values, axis=0)
        
        preds = {
            'RF (standard)': (y_rf1, y_rf_std),
            'RF (complex)': (y_rf2, None),
            'RF (regularized)': (y_rf3, None)
        }
        
        fig = Visualizer.plot_prediction(t_test, y_test, preds)
        
        fig.update_layout(
            title=f"BTC/USDT Prediction (15m timeframe, {len(df_feat)} samples)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig.show()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
