import asyncio
import aiohttp
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from scipy.cluster.hierarchy import linkage
import datetime
import logging
from functools import wraps
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




def log_call(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logging.info(f"Start: {func.__name__}")
        result = await func(*args, **kwargs)
        logging.info(f"End: {func.__name__}")
        return result
    @wraps(func)




    def sync_wrapper(*args, **kwargs):
        logging.info(f"Start: {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"End: {func.__name__}")
        return result
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

@dataclass




class ProjectConfig:
    symbols: list[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
        "SOLUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT", "TRXUSDT", "AVAXUSDT",
        "SHIBUSDT", "NEARUSDT", "ATOMUSDT", "LINKUSDT", "XLMUSDT", "ALGOUSDT",
        "VETUSDT", "ICPUSDT"
    ])
    interval: str = "1d"
    start_date: str = "2023-01-01"




    end_date: str = field(default_factory=lambda: (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))





class BinanceDataFetcher:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.base_url = "https://api.binance.com/api/v3/klines"
        
    def _get_timestamp(self, date_str: str) -> int:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)
    




    @log_call
    async def fetch_symbol_data(self, session: aiohttp.ClientSession, symbol: str) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": self.config.interval,
            "startTime": self._get_timestamp(self.config.start_date),
            "endTime": self._get_timestamp(self.config.end_date)
        }
        async with session.get(self.base_url, params=params) as response:
            data = await response.json()
        records = []
        for entry in data:
            dt = datetime.datetime.fromtimestamp(entry[0] / 1000.0)
            close_price = float(entry[4])
            records.append((dt, close_price))
        df_symbol = pd.DataFrame(records, columns=["date", symbol])
        df_symbol.set_index("date", inplace=True)
        return df_symbol
    




    @log_call
    async def fetch_data(self) -> pd.DataFrame:
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_symbol_data(session, symbol) for symbol in self.config.symbols]
            dataframes = await asyncio.gather(*tasks)
        df = pd.concat(dataframes, axis=1)
        df.sort_index(inplace=True)
        return df




class DataProcessor:
    @staticmethod
    @log_call
    def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
        returns = df.pct_change().dropna()
        return returns





class HierarchicalClustering:
    def __init__(self, returns: pd.DataFrame, symbols: list[str]):
        self.returns = returns
        self.symbols = symbols
        self.results_sumsq = {}
        self.final_rank_map = {}
        self.n = len(symbols)
        self._compute_pairwise_sumsq()
    


    @log_call
    def _compute_pairwise_sumsq(self):
        import itertools
        pairs = list(itertools.combinations(self.symbols, 2))
        for (c1, c2) in pairs:
            diff = self.returns[c1] - self.returns[c2]
            sumsq = (diff ** 2).sum()
            self.results_sumsq[(c1, c2)] = sumsq
        # Ranking the pairs by sumsq in descending order
        pairs_list = list(self.results_sumsq.items())
        pairs_list.sort(key=lambda x: x[1], reverse=True)
        rank_map = {}
        current_rank = 1
        for (pair, val) in pairs_list:
            rank_map[pair] = current_rank
            current_rank += 1
        total_pairs = len(self.results_sumsq)
        for pair in self.results_sumsq.keys():
            self.final_rank_map[pair] = (total_pairs + 1) - rank_map[pair]
    



    @log_call
    def get_linkage_matrix(self, method: str = 'single') -> np.ndarray:
        dist_list = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                c1, c2 = self.symbols[i], self.symbols[j]
                dist_list.append(self.results_sumsq[(c1, c2)])
        dist_array = np.array(dist_list)
        Z = linkage(dist_array, method=method)
        return Z
    



    @log_call
    def print_fusion_table(self, Z: np.ndarray, method_name: str):
        cluster_labels = {}
        table = []
        for step in range(Z.shape[0]):
            c1 = int(Z[step, 0])
            c2 = int(Z[step, 1])
            dist = Z[step, 2]
            new_cluster_idx = self.n + step
            labels_c1 = {self.symbols[c1]} if c1 < self.n else cluster_labels[c1]
            labels_c2 = {self.symbols[c2]} if c2 < self.n else cluster_labels[c2]
            merged = labels_c1.union(labels_c2)
            cluster_labels[new_cluster_idx] = merged
            aggregation_str = "_".join(sorted(list(merged)))
            table.append((step + 1, aggregation_str, dist))
        print(f"\n--- FUSION TABLE ({method_name.upper()} LINKAGE) ---")
        for step_no, agg, dist in table:
            print(f"{step_no}\t{agg}\t{dist}")



class PlotlyVisualizer:
    @staticmethod
    @log_call
    def create_dendrogram(Z: np.ndarray, symbols: list[str], title: str) -> None:
        dummy_data = np.zeros((len(symbols), 1))
        fig = ff.create_dendrogram(
            dummy_data,
            orientation='left',
            labels=symbols,
            linkagefun=lambda x: Z
        )
        fig.update_layout(
            title=title,
            xaxis_title="Distance (SOMSQ)",
            yaxis_title="Cryptocurrencies"
        )
        fig.show()





class RegressionAnalysis:
    @staticmethod
    @log_call




    def run_regression(returns: pd.DataFrame, target: str) -> sm.regression.linear_model.RegressionResultsWrapper:
        
        y = returns[target]
        X = returns.drop(columns=[target])
        X = sm.add_constant(X)  
        model = sm.OLS(y, X).fit()
        print("\n--- MULTIVARIATE REGRESSION ANALYSIS ---")
        print(f"Dependent variable (benchmark): {target}")
        print(model.summary())
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print("\n--- Variance Inflation Factors ---")
        print(vif_data)
        return model





    @staticmethod
    @log_call
    def plot_regression_diagnostics(model: sm.regression.linear_model.RegressionResultsWrapper, returns: pd.DataFrame, target: str) -> None:
        
        y_actual = returns[target].loc[model.fittedvalues.index]
        y_pred = model.fittedvalues
        residuals = model.resid

        fig1 = px.scatter(x=y_pred, y=y_actual, 
                          labels={'x': 'Predicted BTC Returns', 'y': 'Actual BTC Returns'},
                          title='Predicted vs. Actual BTC Returns')
        fig1.add_shape(type="line",
                       x0=y_pred.min(), y0=y_pred.min(),
                       x1=y_pred.max(), y1=y_pred.max(),
                       line=dict(color="Red", dash="dash"))
        fig1.show()

        fig2 = px.scatter(x=y_pred, y=residuals,
                          labels={'x': 'Fitted Values', 'y': 'Residuals'},
                          title='Residuals vs. Fitted Values (BTC Regression)')
        fig2.add_shape(type="line",
                       x0=y_pred.min(), y0=0,
                       x1=y_pred.max(), y1=0,
                       line=dict(color="Red", dash="dash"))
        fig2.show()






async def main():
    config = ProjectConfig()
    logging.info(f"Running analysis from {config.start_date} to {config.end_date}")
    
    fetcher = BinanceDataFetcher(config)
    raw_data = await fetcher.fetch_data()
    print("Downloaded price data:")
    print(raw_data.head())
    
    returns = DataProcessor.compute_returns(raw_data)
    print("\nCalculated returns:")
    print(returns.head())
    



    clustering = HierarchicalClustering(returns, config.symbols)
    for method in ['single', 'complete', 'average']:
        Z = clustering.get_linkage_matrix(method=method)
        clustering.print_fusion_table(Z, method)
        PlotlyVisualizer.create_dendrogram(Z, config.symbols, f"Dendrogram - {method.title()} Linkage (SOMSQ)")
    

    
   
    model = RegressionAnalysis.run_regression(returns, target="BTCUSDT")
    RegressionAnalysis.plot_regression_diagnostics(model, returns, target="BTCUSDT")

if __name__ == "__main__":
    asyncio.run(main())
