"""
Data Download Script for Financial Forecasting Demo
Downloads financial data from Yahoo Finance for federated learning simulation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Create data directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def clean_old_data():
    """Remove old data files to ensure fresh data on each run"""
    import glob
    import shutil
    
    # Remove old CSV files
    raw_files = glob.glob('data/raw/*.csv')
    processed_files = glob.glob('data/processed/*.csv')
    
    for file in raw_files + processed_files:
        try:
            os.remove(file)
            print(f"  Removed: {file}")
        except Exception as e:
            print(f"  Warning: Could not remove {file}: {e}")
    
    # Remove metadata if exists
    metadata_file = 'data/metadata.json'
    if os.path.exists(metadata_file):
        try:
            os.remove(metadata_file)
            print(f"  Removed: {metadata_file}")
        except Exception as e:
            print(f"  Warning: Could not remove {metadata_file}: {e}")
    
    print("  ✓ Old data cleaned")

def download_stock_data(symbol, period="2y"):
    """Download stock data from Yahoo Finance"""
    print(f"Downloading data for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        # Try different periods if default fails
        df = ticker.history(period=period)
        if df.empty:
            print(f"  Trying 1y period for {symbol}...")
            df = ticker.history(period="1y")
        if df.empty:
            print(f"  Trying 6mo period for {symbol}...")
            df = ticker.history(period="6mo")
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        print(f"  ✓ Downloaded {len(df)} records for {symbol}")
        return df
    except Exception as e:
        print(f"  ✗ Error downloading {symbol}: {str(e)}")
        # Generate synthetic data as fallback
        print(f"  Generating synthetic data for {symbol}...")
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        np.random.seed(hash(symbol) % 10000)  # Consistent seed per symbol
        base_price = 100 + hash(symbol) % 50
        prices = base_price + np.cumsum(np.random.randn(500) * 2)
        volumes = np.random.randint(1000000, 10000000, 500)
        
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(500) * 0.01),
            'High': prices * (1 + np.abs(np.random.randn(500)) * 0.02),
            'Low': prices * (1 - np.abs(np.random.randn(500)) * 0.02),
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        print(f"  ✓ Generated {len(df)} synthetic records for {symbol}")
        return df

def generate_synthetic_financial_metrics(df, institution_name):
    """
    Generate synthetic financial metrics based on stock data
    This simulates internal financial data for each institution
    """
    # Use stock data as base for generating synthetic metrics
    df = df.copy()
    
    # Generate cash flow metrics (based on volume and price movements)
    df['cash_flow'] = df['Volume'] * df['Close'] * np.random.uniform(0.8, 1.2, len(df))
    df['cash_flow'] = df['cash_flow'].rolling(window=7).mean()
    
    # Generate liquidity metrics
    df['liquidity_ratio'] = df['Close'] / df['Close'].rolling(window=30).mean()
    df['liquidity_ratio'] = df['liquidity_ratio'].fillna(1.0)
    
    # Generate default risk indicators (inverse relationship with stock performance)
    df['default_risk'] = 1 / (1 + df['Close'] / df['Close'].rolling(window=60).mean())
    df['default_risk'] = df['default_risk'].fillna(0.5)
    df['default_risk'] = np.clip(df['default_risk'], 0.01, 0.99)
    
    # Generate investment returns
    df['investment_return'] = df['Close'].pct_change(periods=1) * 100
    df['investment_return'] = df['investment_return'].fillna(0)
    
    # Add institution identifier
    df['institution'] = institution_name
    
    # Select relevant columns
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'cash_flow', 'liquidity_ratio', 'default_risk', 'investment_return']
    
    return df[features].dropna()

def prepare_forecasting_dataset(df, target_col, horizon_days=30):
    """
    Prepare time series data for forecasting
    Creates features and targets for specified horizon
    """
    df = df.copy()
    
    # Create lag features
    for lag in [1, 7, 14, 30]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Create rolling statistics
    for window in [7, 14, 30]:
        df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_std_{window}'] = df[target_col].rolling(window=window).std()
    
    # Create target variable (future value)
    df[f'{target_col}_target'] = df[target_col].shift(-horizon_days)
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df

def download_institution_data():
    """
    Download data for three simulated financial institutions
    Using different financial sector stocks to represent different institutions
    """
    institutions = {
        'Institution_A': 'JPM',  # JPMorgan Chase - represents Institution A
        'Institution_B': 'BAC',  # Bank of America - represents Institution B  
        'Institution_C': 'WFC'   # Wells Fargo - represents Institution C
    }
    
    all_data = {}
    
    for inst_name, symbol in institutions.items():
        print(f"\n{'='*50}")
        print(f"Processing {inst_name} (using {symbol} data)")
        print(f"{'='*50}")
        
        # Download stock data
        stock_data = download_stock_data(symbol, period="2y")
        
        # Generate synthetic financial metrics
        financial_data = generate_synthetic_financial_metrics(stock_data, inst_name)
        
        # Save raw data
        raw_path = f'data/raw/{inst_name}_raw.csv'
        # Write CSV with proper quoting to handle commas in data
        import csv
        financial_data.to_csv(raw_path, quoting=csv.QUOTE_ALL)
        print(f"Saved raw data to {raw_path}")
        
        # Prepare datasets for different forecasting tasks
        forecasting_datasets = {}
        
        # 1. Cash Flow Forecasting (30, 60, 90 day horizons)
        for horizon in [30, 60, 90]:
            cf_data = prepare_forecasting_dataset(financial_data, 'cash_flow', horizon)
            forecasting_datasets[f'cash_flow_{horizon}d'] = cf_data
        
        # 2. Default Risk Estimation
        dr_data = prepare_forecasting_dataset(financial_data, 'default_risk', 30)
        forecasting_datasets['default_risk'] = dr_data
        
        # 3. Investment Return Prediction
        ir_data = prepare_forecasting_dataset(financial_data, 'investment_return', 30)
        forecasting_datasets['investment_return'] = ir_data
        
        # Save processed datasets
        for task_name, task_data in forecasting_datasets.items():
            processed_path = f'data/processed/{inst_name}_{task_name}.csv'
            # Write CSV with proper quoting to handle commas in data
            import csv
            task_data.to_csv(processed_path, index=True, quoting=csv.QUOTE_ALL)
            print(f"Saved {task_name} dataset to {processed_path}")
        
        all_data[inst_name] = {
            'raw': financial_data,
            'forecasting': forecasting_datasets
        }
    
    # Create metadata file
    metadata = {
        'institutions': list(institutions.keys()),
        'data_sources': institutions,
        'download_date': datetime.now().isoformat(),
        'forecasting_tasks': ['cash_flow_30d', 'cash_flow_60d', 'cash_flow_90d', 
                             'default_risk', 'investment_return']
    }
    
    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Data download completed successfully!")
    print(f"{'='*50}")
    print(f"\nMetadata saved to data/metadata.json")
    print(f"Total institutions: {len(institutions)}")
    print(f"Forecasting tasks: {len(metadata['forecasting_tasks'])}")
    
    return all_data

if __name__ == "__main__":
    print("Starting financial data download...")
    print("This will download data from Yahoo Finance and generate synthetic metrics")
    print("Please ensure you have an internet connection.\n")
    
    # Clean old data first
    print("Cleaning old data files...")
    clean_old_data()
    print()
    
    try:
        data = download_institution_data()
        print("\n✓ Data download completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during data download: {str(e)}")
        print("Please check your internet connection and try again.")

