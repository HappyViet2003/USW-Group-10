"""
This script downloads historical 1-minute bar data for Cryptocurrency pairs (e.g., BTC/USD)
from the Alpaca Market Data API (Crypto Feed) for a configured date range.

It reads API credentials and parameters (data path, start/end dates) from YAML files.
Unlike equity data, this script does not filter for exchange opening hours, as the
cryptocurrency market operates 24/7. It retrieves the data and writes one cleaned
Parquet file per currency pair.

Inputs:
- YAML configs: ../../conf/keys.yaml (API keys), ../../conf/params.yaml (data path and date range)
- Hardcoded Symbol List: ['BTC/USD']

Outputs:
- One Parquet file per pair under <DATA_PATH>/Bars_1m_crypto/

Requirements:
- Packages: alpaca-py, pandas, pyyaml, pyarrow
"""

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime
import yaml
import os

# 1. Load API credentials from YAML configuration file
try:
    print("Loading API credentials from 'conf/keys.yaml'...")
    keys = yaml.safe_load(open("../../conf/keys.yaml"))
    API_KEY = keys['KEYS']['APCA-API-KEY-ID-Data']
    SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY-Data']
except FileNotFoundError:
    print("Error: 'conf/keys.yaml' not found. Please create it with your Alpaca keys.")
    exit()

# 2. Load data acquisition parameters from YAML configuration file
try:
    params = yaml.safe_load(open("../../conf/params.yaml"))
    # Wir ändern den Ordnernamen leicht ab, um Verwirrung mit Aktien zu vermeiden
    base_path = params['DATA_ACQUISITON']['DATA_PATH']
    OUTPUT_PATH = os.path.join(base_path, 'Bars_1m_crypto')
    START_DATE = datetime.strptime(params['DATA_ACQUISITON']['START_DATE'], "%Y-%m-%d")
    END_DATE = datetime.strptime(params['DATA_ACQUISITON']['END_DATE'], "%Y-%m-%d")
except FileNotFoundError:
    print("Error: 'conf/params.yaml' not found. Please check your config.")
    exit()

# 3. Initialize the Alpaca Crypto Client
client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# Define symbols to fetch.
symbols = ['BTC/USD']

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"Starting Data Acquisition for {symbols}")
print(f"Timeframe: 1 Minute | Range: {START_DATE} to {END_DATE}")
print("-" * 50)

for symbol in symbols:
    print(f"Fetching data for {symbol}...")

    # 4. Create Request Object
    # Krypto braucht kein 'Adjustment' (Splits) und keinen 'Calendar' Check
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=START_DATE,
        end=END_DATE
    )

    try:
        # 5. Retrieve bar data from Alpaca API
        bars = client.get_crypto_bars(request)

        # Convert to DataFrame
        df = bars.df

        # Reset index to get 'timestamp' as a column
        df.reset_index(inplace=True)

        # 6. Data Cleaning
        # Remove the 'symbol' column if it exists (redundant as we save per file)
        if 'symbol' in df.columns:
            df.drop(columns=['symbol'], inplace=True)

        # Safe filename (Slash in BTC/USD durch Unterstrich ersetzen)
        safe_symbol = symbol.replace("/", "_")
        save_path = f'{OUTPUT_PATH}/{safe_symbol}.parquet'

        # 7. Save as Parquet (Anforderung aus README erfüllt)
        df.to_parquet(save_path, index=False)

        print(f"✅ Success! Saved {len(df)} rows to: {save_path}")
        print(f"   First Timestamp: {df['timestamp'].min()}")
        print(f"   Last Timestamp:  {df['timestamp'].max()}")

    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")

print("-" * 50)
print("Data Acquisition Completed.")