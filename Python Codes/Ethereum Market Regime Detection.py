import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ccxt
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class EthereumChangePointDetector:
    
    def __init__(self, timeframe='15m', start_date='2020-01-01', n_regimes=3, window_size=32):
        
        self.timeframe = timeframe
        self.start_date = start_date
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.data = None
        self.returns = None
        self.model = None
        self.volatility = None
        self.regime_probs = None
        self.most_likely_regime = None
        
    def fetch_ethereum_data(self):
        
        print(f"Fetching Ethereum data from Binance with {self.timeframe} timeframe (from {self.start_date} to present)...")
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
        end_time = datetime.now()
        start_time = datetime.strptime(self.start_date, '%Y-%m-%d')
        
        days_difference = (end_time - start_time).days
        print(f"Analysis period covers {days_difference} days")
        

        since = int(start_time.timestamp() * 1000)
        
        all_data = []
        current_since = since
        total_batches = 0
        
        try:
            while current_since < end_time.timestamp() * 1000:
                total_batches += 1
                if total_batches % 5 == 0:
                    print(f"Fetching batch {total_batches}... (gathered {len(all_data)} data points so far)")
                
                ohlcv = exchange.fetch_ohlcv('ETH/USDT', self.timeframe, since=current_since, limit=1000)
                if len(ohlcv) == 0:
                    break
                    
                all_data.extend(ohlcv)
                
                current_since = ohlcv[-1][0] + 1
                
                exchange.sleep(exchange.rateLimit / 1000)
            
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            
            self.ohlcv_data = df
            
            self.data = df['close']
            
            self.returns = np.log(self.data / self.data.shift(1)).dropna()
            
            print(f"Successfully fetched {len(self.data)} data points ({self.timeframe} intervals)")
            print(f"Data range: {self.data.index.min()} to {self.data.index.max()}")
            
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            
            print("Creating synthetic data for testing...")
            days = (datetime.now() - datetime.strptime(self.start_date, '%Y-%m-%d')).days
            periods = 96 * days
            index = pd.date_range(start=self.start_date, end=datetime.now(), freq='15min')
            
            np.random.seed(42)
            prices = [1000]
            segment_size = periods // 5 
            
            for i in range(1, len(index)):
                if i < segment_size:  
                    change = np.random.normal(0.0008, 0.002)
                elif i < 2 * segment_size:  
                    change = np.random.normal(-0.0006, 0.004)
                elif i < 3 * segment_size:  
                    change = np.random.normal(0.0001, 0.001)
                elif i < 4 * segment_size:  
                    change = np.random.normal(0.0007, 0.003)
                else:  
                    change = np.random.normal(0.0002, 0.002)
                    
                prices.append(prices[-1] * (1 + change))
            
            self.data = pd.Series(prices[:len(index)], index=index)
            self.returns = np.log(self.data / self.data.shift(1)).dropna()
            
            print(f"Created {len(self.data)} synthetic data points")
            
            return self.data
    
    def calculate_volatility(self):
        if self.returns is None:
            raise ValueError("Data not loaded. Call fetch_ethereum_data first.")
            
        self.volatility = self.returns.rolling(window=self.window_size).std().dropna()
        
        return self.volatility
    
    def fit_model(self):
        
        print(f"Fitting model with {self.n_regimes} regimes...")
        
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_ethereum_data first.")
        
        if self.volatility is None:
            self.volatility = self.calculate_volatility()
        
        X = self.volatility.values.reshape(-1, 1)
        
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            max_iter=1000,
            n_init=10
        )
        
        self.model.fit(X)
        
        self.regime_probs = self.model.predict_proba(X)
        
        self.most_likely_regime = self.model.predict(X)
        
        print("Model fitting complete")
        
        return self.model
    
    def detect_change_points(self, threshold=0.3):
        
        print("Detecting change points...")
        
        if self.model is None:
            raise ValueError("Model has not been fit. Call fit_model first.")
            
        change_points = np.where(np.diff(self.most_likely_regime) != 0)[0]
        
        dates = self.volatility.index
        
        stats = []
        
        for cp in change_points:
            if cp+1 < len(self.regime_probs):
                from_regime = self.most_likely_regime[cp]
                to_regime = self.most_likely_regime[cp+1]
                
                confidence = np.abs(
                    self.regime_probs[cp, from_regime] - self.regime_probs[cp+1, to_regime]
                )
                
                if confidence > threshold:
                    window = min(20, cp, len(self.volatility) - cp - 1)
                    
                    before_start = max(0, cp - window)
                    after_end = min(len(self.volatility), cp + window + 1)
                    
                    before_vol = self.volatility.iloc[before_start:cp].mean()
                    after_vol = self.volatility.iloc[cp+1:after_end].mean()
                    vol_change_pct = (after_vol - before_vol) / before_vol * 100 if before_vol != 0 else float('inf')
                    
                    before_ret = self.returns.iloc[before_start + self.window_size:cp + self.window_size].mean() * 100
                    after_ret = self.returns.iloc[cp + self.window_size:after_end + self.window_size].mean() * 100
                    
                    stats.append({
                        'Date': dates[cp],
                        'From_Regime': int(from_regime) + 1,
                        'To_Regime': int(to_regime) + 1,
                        'Confidence': confidence,
                        'Before_Volatility': before_vol,
                        'After_Volatility': after_vol,
                        'Volatility_Change_%': vol_change_pct,
                        'Before_Return_%': before_ret,
                        'After_Return_%': after_ret,
                        'Return_Change_%': after_ret - before_ret
                    })
        
        change_points_df = pd.DataFrame(stats)
        print(f"Detected {len(change_points_df)} change points")
        
        return change_points_df
    
    def visualize_results(self, threshold=0.3):
    
        print("Creating visualizations...")
        
        if self.model is None:
            raise ValueError("Model has not been fit. Call fit_model first.")
            
        change_points_df = self.detect_change_points(threshold=threshold)
        
        means = self.model.means_.flatten()
        regime_order = np.argsort(means)
        
        regime_descriptions = {
            regime_order[0]: "Low Volatility",
            regime_order[1]: "Medium Volatility",
            regime_order[2]: "High Volatility"
        } if self.n_regimes == 3 else {i: f"Regime {i+1}" for i in range(self.n_regimes)}
        
        
        try:
            self._create_price_with_regimes_chart(change_points_df, regime_descriptions)
        except Exception as e:
            print(f"Error creating price with regimes chart: {e}")
        
        try:
            self._create_volatility_chart(regime_descriptions)
        except Exception as e:
            print(f"Error creating volatility chart: {e}")
        
        try:
            self._create_regime_probability_chart(regime_descriptions)
        except Exception as e:
            print(f"Error creating regime probability chart: {e}")
        
        try:
            self._create_returns_distribution_chart(regime_descriptions)
        except Exception as e:
            print(f"Error creating returns distribution chart: {e}")
        
        if hasattr(self, 'ohlcv_data'):
            try:
                self._create_candlestick_chart(change_points_df)
            except Exception as e:
                print(f"Error creating candlestick chart: {e}")
        
        return change_points_df
    
    def _create_price_with_regimes_chart(self, change_points_df, regime_descriptions):
        
        plt.figure(figsize=(16, 8))
        plt.plot(self.data.index, self.data, color='blue', linewidth=1.5, alpha=0.8)
        plt.title('Ethereum Price with Market Regimes', fontsize=16)
        plt.ylabel('Price (USDT)', fontsize=12)
        plt.grid(alpha=0.3)
        
        regime_colors = ['#DDFFDD', '#FFFFDD', '#FFDDDD']  
        dates = self.volatility.index
        
        change_idx = np.where(np.diff(self.most_likely_regime) != 0)[0]
        
        start_idx = 0
        for end_idx in change_idx:
            regime = self.most_likely_regime[start_idx]
            plt.axvspan(dates[start_idx], dates[end_idx], 
                        color=regime_colors[regime], alpha=0.3)
            
            mid_idx = start_idx + (end_idx - start_idx) // 2
            if mid_idx < len(dates):
                plt.text(dates[mid_idx], plt.gca().get_ylim()[1] * 0.95, 
                         regime_descriptions[regime],
                         horizontalalignment='center',
                         backgroundcolor='white',
                         alpha=0.7,
                         fontsize=10)
            
            start_idx = end_idx + 1
        
        if start_idx < len(dates):
            regime = self.most_likely_regime[start_idx]
            plt.axvspan(dates[start_idx], dates[-1], 
                        color=regime_colors[regime], alpha=0.3)
            
            mid_idx = start_idx + (len(dates) - start_idx) // 2
            if mid_idx < len(dates):
                plt.text(dates[mid_idx], plt.gca().get_ylim()[1] * 0.95, 
                         regime_descriptions[regime],
                         horizontalalignment='center',
                         backgroundcolor='white',
                         alpha=0.7,
                         fontsize=10)
        
        for _, cp in change_points_df.iterrows():
            plt.axvline(x=cp['Date'], color='red', linestyle='--', alpha=0.7, linewidth=2)
            plt.annotate(f"{cp['Date'].strftime('%Y-%m-%d %H:%M')}", 
                         xy=(cp['Date'], plt.gca().get_ylim()[1] * 0.9),
                         rotation=90,
                         fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig("eth_price_with_regimes.png", dpi=300)
        print("Saved eth_price_with_regimes.png")
    
    def _create_volatility_chart(self, regime_descriptions):
        
        plt.figure(figsize=(16, 6))
        
        plt.plot(self.volatility.index, self.volatility, color='purple', linewidth=1.5, alpha=0.8)
        plt.title('Ethereum Volatility with Market Regimes', fontsize=16)
        plt.ylabel('Log Return Volatility', fontsize=12)
        plt.grid(alpha=0.3)
        
        means = self.model.means_.flatten()
        regime_order = np.argsort(means)
        
        regime_colors = ['#DDFFDD', '#FFFFDD', '#FFDDDD'] 
        
        for i, regime in enumerate(regime_order):
            mean_vol = means[regime]
            plt.axhline(y=mean_vol, color=regime_colors[i], linestyle='-', 
                        linewidth=2, alpha=0.8, 
                        label=f"{regime_descriptions[regime]} (Mean: {mean_vol:.5f})")
        
        dates = self.volatility.index
        start_idx = 0
        change_idx = np.where(np.diff(self.most_likely_regime) != 0)[0]
        
        for end_idx in change_idx:
            regime = self.most_likely_regime[start_idx]
            plt.axvspan(dates[start_idx], dates[end_idx], 
                        color=regime_colors[regime], alpha=0.2)
            start_idx = end_idx + 1
            
        if start_idx < len(dates):
            regime = self.most_likely_regime[start_idx]
            plt.axvspan(dates[start_idx], dates[-1], 
                        color=regime_colors[regime], alpha=0.2)
        
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("eth_volatility_regimes.png", dpi=300)
        print("Saved eth_volatility_regimes.png")
    
    def _create_regime_probability_chart(self, regime_descriptions):
        
        plt.figure(figsize=(16, 6))
        
        for i in range(self.n_regimes):
            plt.plot(self.volatility.index, self.regime_probs[:, i], 
                     linewidth=1.5, alpha=0.8,
                     label=f"{regime_descriptions[i]}")
            
        plt.title('Regime Probability Over Time', fontsize=16)
        plt.ylabel('Probability', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig("eth_regime_probabilities.png", dpi=300)
        print("Saved eth_regime_probabilities.png")
    
    def _create_returns_distribution_chart(self, regime_descriptions):
        
        fig, axes = plt.subplots(1, self.n_regimes, figsize=(18, 6), sharey=True)
        
        colors = ['green', 'gold', 'red']
        
        for i in range(self.n_regimes):
            regime_idx = self.most_likely_regime == i
            
            returns_idx = np.zeros(len(self.returns), dtype=bool)
            vol_indices = np.where(regime_idx)[0]
            
            for idx in vol_indices:
                if idx + self.window_size < len(returns_idx):
                    returns_idx[idx + self.window_size] = True
            
            regime_returns = self.returns.iloc[returns_idx] * 100  
            
            if len(regime_returns) > 0:
                sns.histplot(regime_returns, kde=True, ax=axes[i], color=colors[i], alpha=0.6)
                
                mean = regime_returns.mean()
                std = regime_returns.std()
                
                axes[i].axvline(mean, color='black', linestyle='-', 
                                label=f'Mean: {mean:.3f}%')
                axes[i].axvline(mean + std, color='black', linestyle='--', 
                                label=f'±1 StdDev: {std:.3f}%')
                axes[i].axvline(mean - std, color='black', linestyle='--')
                
                axes[i].set_title(f"{regime_descriptions[i]}", fontsize=14)
                axes[i].set_xlabel('Return (%)', fontsize=12)
                if i == 0:
                    axes[i].set_ylabel('Frequency', fontsize=12)
                axes[i].legend()
                axes[i].grid(alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, "No data for this regime", 
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=axes[i].transAxes)
        
        plt.suptitle('Return Distribution by Market Regime', fontsize=16)
        plt.tight_layout()
        plt.savefig("eth_returns_by_regime.png", dpi=300)
        print("Saved eth_returns_by_regime.png")
    
    def _create_candlestick_chart(self, change_points_df):
        """Create candlestick chart with change points if OHLC data is available"""
        if not hasattr(self, 'ohlcv_data'):
            print("OHLCV data not available for candlestick chart.")
            return
            
        try:
            import mplfinance as mpf
            
            chart_data = self.ohlcv_data.tail(1000).copy()
            
            chart_data.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            visible_change_points = []
            start_date = chart_data.index[0]
            end_date = chart_data.index[-1]
            
            for _, cp in change_points_df.iterrows():
                if start_date <= cp['Date'] <= end_date:
                    visible_change_points.append(cp['Date'])
            
            title = f'Ethereum OHLCV with Change Points (Recent {len(chart_data)} periods)'
            
            mc = mpf.make_marketcolors(
                up='green', down='red',
                wick='inherit', edge='inherit',
                volume='inherit'
            )
            s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
            
            if visible_change_points:
                mpf.plot(
                    chart_data, 
                    type='candle', 
                    title=title,
                    volume=True,
                    style=s,
                    vlines=dict(vlines=visible_change_points, colors='r', linewidths=1.5, alpha=0.8),
                    savefig='eth_candlestick_chart.png',
                    figsize=(16, 10)
                )
                print(f"Saved eth_candlestick_chart.png with {len(visible_change_points)} change points")
            else:
                mpf.plot(
                    chart_data, 
                    type='candle', 
                    title=title,
                    volume=True,
                    style=s,
                    savefig='eth_candlestick_chart.png',
                    figsize=(16, 10)
                )
                print("Saved eth_candlestick_chart.png (no change points in visible range)")
            
        except ImportError:
            print("package not found")
            print("Candlestick chart not created.")
        except Exception as e:
            print(f"Error creating candlestick chart: {e}")
            print("Attempting to create a simplified version...")
            
            try:
                plt.figure(figsize=(16, 8))
                
                plt.plot(chart_data.index, chart_data['Close'], color='blue', linewidth=1.5)
                
                for cp_date in visible_change_points:
                    plt.axvline(x=cp_date, color='red', linestyle='--', alpha=0.7)
                
                plt.title('Ethereum Price with Change Points (Simplified)', fontsize=16)
                plt.ylabel('Price (USDT)', fontsize=12)
                plt.grid(alpha=0.3)
                
                plt.tight_layout()
                plt.savefig("eth_simplified_chart.png", dpi=300)
                print("Saved simplified version as eth_simplified_chart.png")
            except Exception as e2:
                print(f"Could not create simplified chart either: {e2}")
                print("Skipping chart creation.")
    
    def extract_regime_characteristics(self):
        
        print("Extracting regime characteristics...")
        
        if self.model is None:
            raise ValueError("Model has not been fit. Call fit_model first.")
            
        regime_stats = []
        
        for i in range(self.n_regimes):
            regime_idx = self.most_likely_regime == i
            
            if np.sum(regime_idx) > 0:
                regime_vol = self.volatility.iloc[regime_idx]
                
                returns_idx = np.zeros(len(self.returns), dtype=bool)
                vol_indices = np.where(regime_idx)[0]
                
                for idx in vol_indices:
                    if idx + self.window_size < len(returns_idx):
                        returns_idx[idx + self.window_size] = True
                
                regime_ret = self.returns.iloc[returns_idx]
                
                regime_stats.append({
                    'Regime': i + 1,
                    'Periods': np.sum(regime_idx),
                    'Percentage_of_Time': np.sum(regime_idx) / len(self.most_likely_regime) * 100,
                    'Avg_Return_%': regime_ret.mean() * 100 if len(regime_ret) > 0 else None,
                    'Return_StdDev_%': regime_ret.std() * 100 if len(regime_ret) > 0 else None,
                    'Avg_Volatility': regime_vol.mean() if len(regime_vol) > 0 else None,
                    'Max_Volatility': regime_vol.max() if len(regime_vol) > 0 else None,
                })
        
        regime_stats_df = pd.DataFrame(regime_stats)
        
        means = self.model.means_.flatten()
        sorted_regimes = np.argsort(means)
        
        descriptions = ["Low Volatility", "Medium Volatility", "High Volatility"]
        if self.n_regimes <= len(descriptions):
            for i, orig_regime in enumerate(sorted_regimes):
                idx = regime_stats_df['Regime'] == orig_regime + 1
                if np.any(idx):
                    regime_stats_df.loc[idx, 'Description'] = descriptions[i]
        
        return regime_stats_df
    
    def run_analysis(self):
        
        self.fetch_ethereum_data()
        self.calculate_volatility()
        self.fit_model()
        change_points = self.detect_change_points()
        self.visualize_results()  
        regime_stats = self.extract_regime_characteristics()
        
        print("\n=Detected Change Points =")
        print(change_points)
        
        print("\n= Regime Characteristics =")
        print(regime_stats)
        
        return {
            'change_points': change_points,
            'regime_stats': regime_stats
        }


if __name__ == "__main__":
    detector = EthereumChangePointDetector(
        timeframe='15m',      
        start_date='2020-01-01', 
        n_regimes=3,          
        window_size=32        
    )
    
    results = detector.run_analysis()
