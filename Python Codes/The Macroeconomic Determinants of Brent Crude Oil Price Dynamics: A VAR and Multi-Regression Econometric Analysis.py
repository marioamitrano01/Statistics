import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from fredapi import Fred
from datetime import datetime
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings('ignore')

sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

fred_api_key = "054d79a6dffb592fd462713e98e04d85"
fred = Fred(api_key=fred_api_key)

series_ids = {
    'brent': 'POILBREUSDM',  
    'gdp': 'GDPC1',          
    'inflation': 'CPIAUCSL', 
    'interest_rate': 'FEDFUNDS', 
    'dollar_index': 'DTWEXBGS'  
}

start_date = '2015-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')  

def fetch_fred_data(series_ids, start_date, end_date):
    
    print(f"Fetching data from FRED from {start_date} to {end_date}...")
    
    data = pd.DataFrame()
    
    for name, series_id in series_ids.items():
        print(f"Retrieving {name} (Series ID: {series_id})...")
        series = fred.get_series(series_id, start_date, end_date)
        
        if data.empty:
            data = pd.DataFrame(index=series.index)
        
        data[name] = series
    
   
    data_monthly = data.resample('M').last()
    
    data_filled = data_monthly.fillna(method='ffill')
    
    data_filled = data_filled.fillna(method='bfill')
    
    return data_filled

def preprocess_data(data):
    
    print("Preprocessing data...")
    
    df = data.copy()
    
    for col in df.columns:
        df[f'{col}_pct_change'] = df[col].pct_change() * 100
    
    window_size = 12  
    for col in [c for c in df.columns if c != 'brent' and '_pct_change' not in c]:
        df[f'brent_{col}_corr'] = df['brent'].rolling(window=window_size).corr(df[col])
    
    df = df.dropna()
    
    return df

def plot_time_series(data):
    
    print("Plotting time series of all variables...")
    
    fig, axes = plt.subplots(len(series_ids), 1, figsize=(12, 15), sharex=True)
    
    for i, col in enumerate(series_ids.keys()):
        axes[i].plot(data.index, data[col], linewidth=2)
        axes[i].set_title(f'{col.replace("_", " ").title()}', fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        
        try:
            from pandas_datareader.data import DataReader
            from datetime import datetime
            
            recession_data = fred.get_series('USREC', start_date, end_date)
            recession_periods = []
            
            in_recession = False
            rec_start = None
            
            for date, value in recession_data.items():
                if value == 1 and not in_recession:
                    in_recession = True
                    rec_start = date
                elif value == 0 and in_recession:
                    in_recession = False
                    recession_periods.append((rec_start, date))
            
            for rec_start, rec_end in recession_periods:
                axes[i].axvspan(rec_start, rec_end, color='gray', alpha=0.2)
        except:
            pass
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    plt.savefig('time_series_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlations(data):
    
    print("Plotting correlations between variables...")
    
    corr_matrix = data[list(series_ids.keys())].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Economic Variables', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    for col in [c for c in data.columns if c.startswith('brent_') and c.endswith('_corr')]:
        variable_name = col.replace('brent_', '').replace('_corr', '')
        plt.plot(data.index, data[col], linewidth=2, label=variable_name)
    
    plt.title('12-Month Rolling Correlation with Brent Crude Oil Price', fontsize=16)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('rolling_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_lead_lag_relationships(data):
    
    print("Analyzing lead-lag relationships between variables...")
    
    target_var = 'brent_pct_change'
    predictor_vars = [col for col in data.columns if '_pct_change' in col and col != target_var]
    
    max_lag = 12
    
    results = pd.DataFrame(columns=['Variable', 'Lag', 'Correlation', 'R_Squared', 'P_Value'])
    
    for lag in range(1, max_lag + 1):
        for var in predictor_vars:
            lagged_var = data[var].shift(lag)
            
            valid_data = pd.DataFrame({
                'y': data[target_var],
                'x': lagged_var
            }).dropna()
            
            if len(valid_data) > 0:
                correlation = valid_data['x'].corr(valid_data['y'])
                
                X = sm.add_constant(valid_data['x'])
                model = sm.OLS(valid_data['y'], X).fit()
                
                r_squared = model.rsquared
                p_value = model.pvalues[1]  
                
                results = pd.concat([results, pd.DataFrame({
                    'Variable': [var.replace('_pct_change', '')],
                    'Lag': [lag],
                    'Correlation': [correlation],
                    'R_Squared': [r_squared],
                    'P_Value': [p_value]
                })], ignore_index=True)
    
    significant_results = results[results['P_Value'] < 0.05]
    
    significant_results = significant_results.sort_values('R_Squared', ascending=False)
    
    if not significant_results.empty:
        plt.figure(figsize=(12, 8))
        
        best_results = significant_results.loc[significant_results.groupby('Variable')['R_Squared'].idxmax()]
        
        best_results = best_results.sort_values('R_Squared', ascending=False)
        
        barplot = sns.barplot(x='Variable', y='R_Squared', data=best_results)
        
        for i, row in enumerate(best_results.itertuples()):
            plt.text(i, row.R_Squared + 0.02, f'p={row.P_Value:.4f}\nlag={row.Lag}', 
                     ha='center', va='bottom', fontsize=10)
        
        plt.title('Lead-Lag Analysis: Impact on Brent Crude Oil Price Changes', fontsize=16)
        plt.xlabel('Variable', fontsize=14)
        plt.ylabel('R-Squared (Explanatory Power)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('lead_lag_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def build_var_model(data):
    
    print("Building VAR model...")
    
    var_data = data[[col for col in data.columns if '_pct_change' in col]]
    

    model = VAR(var_data)
    

    lag_order = model.select_order(maxlags=12)
    selected_lag = lag_order.selected_orders['aic']
    print(f"Selected lag order: {selected_lag}")
    

    var_model = model.fit(selected_lag)
    

    print(var_model.summary())
    

    plt.figure(figsize=(15, 10))
    
    brent_idx = var_data.columns.get_loc('brent_pct_change')
    

    irf = var_model.irf(periods=24)  
    
    subplot_count = 1
    for i, col in enumerate(var_data.columns):
        if col != 'brent_pct_change':

            plt.subplot(2, 2, subplot_count)
            

            impulse_idx = var_data.columns.get_loc(col)
            irf_data = irf.irfs[:, brent_idx, impulse_idx]
            

            plt.plot(range(len(irf_data)), irf_data, 'b-', linewidth=2)
            
            plt.axhline(y=0, color='red', linestyle='--')
            variable_name = col.replace('_pct_change', '')
            plt.title(f'Response of Brent to {variable_name.capitalize()} Shock', fontsize=12)
            plt.xlabel('Months', fontsize=10)
            plt.ylabel('Percentage Points', fontsize=10)
            
            subplot_count += 1
    
    plt.tight_layout()
    plt.savefig('var_impulse_responses.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    plt.figure(figsize=(12, 8))
    
    try:

        fevd = var_model.fevd(periods=24)
        fevd_data = fevd.decomp[brent_idx]
        

        fevd_pct = fevd_data / np.sum(fevd_data, axis=1)[:, np.newaxis] * 100
        

        labels = [col.replace('_pct_change', '') for col in var_data.columns]
        plt.stackplot(range(len(fevd_pct)), fevd_pct.T, labels=labels, alpha=0.7)
        
        plt.title('Forecast Error Variance Decomposition of Brent Crude Oil Price', fontsize=16)
        plt.xlabel('Months', fontsize=14)
        plt.ylabel('Percentage of Variance', fontsize=14)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('var_variance_decomposition.png', dpi=300, bbox_inches='tight')
    except:
        print("Could not compute variance decomposition, skipping this plot.")
    
    plt.close()

def analyze_impact_of_variables(data):
    
    print("Analyzing impact of variables on Brent oil prices...")
    

    lag_periods = [1, 3, 6, 12]  
    

    df = data.copy()
    
    for var in [col for col in series_ids.keys() if col != 'brent']:
        for lag in lag_periods:
            df[f'{var}_lag_{lag}'] = df[var].shift(lag)
    
    df = df.dropna()
    
    X_vars = [col for col in df.columns if '_lag_' in col]
    X = df[X_vars]
    y = df['brent']
    
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    print(model.summary())
    
    coefs = model.params.drop('const').reset_index()
    coefs.columns = ['Variable', 'Coefficient']
    
    p_values = model.pvalues.drop('const').reset_index()
    p_values.columns = ['Variable', 'P_Value']
    coefs = coefs.merge(p_values, on='Variable')
    
    coefs['Significant'] = coefs['P_Value'] < 0.05
    
    coefs['Base_Variable'] = coefs['Variable'].apply(lambda x: x.split('_lag_')[0])
    coefs['Lag'] = coefs['Variable'].apply(lambda x: int(x.split('_lag_')[1]))
    
    plt.figure(figsize=(14, 10))
    
    for i, var in enumerate(coefs['Base_Variable'].unique()):
        plt.subplot(2, 2, i+1)
        
        var_coefs = coefs[coefs['Base_Variable'] == var]
        
        var_coefs = var_coefs.sort_values('Lag')
        
        colors = ['blue' if sig else 'lightblue' for sig in var_coefs['Significant']]
        
        plt.bar(var_coefs['Lag'], var_coefs['Coefficient'], color=colors)
        
        for j, row in var_coefs.iterrows():
            if row['Significant']:
                plt.text(row['Lag'], row['Coefficient'] + (0.1 if row['Coefficient'] > 0 else -0.1), 
                         '*', ha='center', fontsize=14)
        
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        plt.title(f'Impact of {var.capitalize()} on Brent Oil Price', fontsize=14)
        plt.xlabel('Lag (Months)', fontsize=12)
        plt.ylabel('Coefficient', fontsize=12)
        plt.xticks(var_coefs['Lag'])
    
    plt.tight_layout()
    plt.savefig('regression_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(14, 8))
    
    contributions = pd.DataFrame(index=df.index)
    
    for var in coefs['Base_Variable'].unique():
        var_coefs = coefs[coefs['Base_Variable'] == var]
        
        contribution = np.zeros(len(df))
        
        for _, row in var_coefs.iterrows():
            lag = row['Lag']
            coef = row['Coefficient']
            contribution += coef * df[f'{var}_lag_{lag}'].values
        
        contributions[var] = contribution
    
    contributions['constant'] = model.params['const']
    
    plt.figure(figsize=(14, 8))
    
    for col in contributions.columns:
        if col != 'constant':
            plt.plot(df.index, contributions[col], linewidth=2, label=f'{col} contribution')
    
    plt.axhline(y=contributions['constant'].mean(), color='black', linestyle='--', 
                linewidth=1, label='Average constant term')
    
    plt.plot(df.index, df['brent'], 'k-', linewidth=3, label='Actual Brent Price')
    
    plt.title('Impact of Macroeconomic Variables on Brent Oil Price', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Brent Oil Price (USD)', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('variable_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    

    avg_abs_contrib = {col: np.abs(contributions[col]).mean() for col in contributions.columns if col != 'constant'}
    
    sorted_vars = sorted(avg_abs_contrib.items(), key=lambda x: x[1], reverse=True)
    vars_names = [x[0] for x in sorted_vars]
    vars_values = [x[1] for x in sorted_vars]
    
    plt.bar(vars_names, vars_values)
    plt.title('Average Absolute Impact of Variables on Brent Oil Price', fontsize=16)
    plt.xlabel('Variable', fontsize=14)
    plt.ylabel('Average Absolute Impact (USD)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('average_variable_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Starting analysis of macroeconomic variables' impact on Brent crude oil prices...")
    
    data = fetch_fred_data(series_ids, start_date, end_date)
    
    processed_data = preprocess_data(data)
    
    plot_time_series(data)
    
    plot_correlations(processed_data)
    
    leadlag_results = analyze_lead_lag_relationships(processed_data)
    
    build_var_model(processed_data)
    

    analyze_impact_of_variables(processed_data)
    
    print("Analysis complete! Check the generated PNG files for results.")

if __name__ == "__main__":
    main()
