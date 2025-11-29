"""
Erweiterte Version von fetch_external_data.py mit M2-Integration

Dieses Skript lädt:
1. ETF-Daten von Alpaca (QQQ, GLD, UUP)
2. M2-Geldmenge von FRED API

Alle Daten werden im gleichen Ordner gespeichert: external_data/

Voraussetzungen:
- Alpaca API Keys in conf/keys.yaml
- FRED API Key in conf/keys.yaml (kostenlos von https://fred.stlouisfed.org/docs/api/api_key.html)
"""

import os
import yaml
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment


# ============================================================================
# TEIL 1: ALPACA ETF-DATEN (Original-Code)
# ============================================================================

def load_config():
    """Lädt Konfiguration aus YAML-Dateien"""
    try:
        print("Loading configuration...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        keys_path = os.path.join(script_dir, "../../conf/keys.yaml")
        params_path = os.path.join(script_dir, "../../conf/params.yaml")

        keys = yaml.safe_load(open(keys_path))
        params = yaml.safe_load(open(params_path))

        return keys, params, script_dir

    except FileNotFoundError as e:
        print(f"❌ Error: Configuration file not found: {e}")
        print("Please ensure conf/keys.yaml and conf/params.yaml exist.")
        exit(1)


def fetch_alpaca_etfs(keys, params, script_dir):
    """Lädt ETF-Daten von Alpaca"""

    print("\n" + "=" * 70)
    print("PART 1: Fetching ETF Data from Alpaca")
    print("=" * 70)

    # API Keys
    API_KEY = keys['KEYS']['APCA-API-KEY-ID-Data']
    SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY-Data']

    # Parameter
    base_path = params['DATA_ACQUISITON']['DATA_PATH']
    OUTPUT_PATH = os.path.join(script_dir, base_path, 'external_data')
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    START_DATE = datetime.strptime(params['DATA_ACQUISITON']['START_DATE'], "%Y-%m-%d")
    END_DATE = datetime.strptime(params['DATA_ACQUISITON']['END_DATE'], "%Y-%m-%d")

    # Alpaca Client
    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

    # Strategische ETF-Auswahl:
    # QQQ = Nasdaq 100 (Tech korreliert oft mit Krypto)
    # GLD = Gold (Safe Haven Asset)
    # UUP = US Dollar Index (Starker Dollar drückt oft Krypto-Preise)
    symbols = ['QQQ', 'GLD', 'UUP']

    print(f"\nSymbols to fetch: {symbols}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output path: {OUTPUT_PATH}")
    print("-" * 70)

    results = {}

    for symbol in symbols:
        print(f"\nFetching {symbol}...")

        # Request erstellen
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=START_DATE,
            end=END_DATE,
            adjustment=Adjustment.ALL  # Berücksichtigt Splits
        )

        try:
            # Daten holen
            bars = client.get_stock_bars(request)
            df = bars.df
            df.reset_index(inplace=True)

            # Cleaning (Symbol-Spalte entfernen)
            if 'symbol' in df.columns:
                df.drop(columns=['symbol'], inplace=True)

            # Speichern
            save_path = os.path.join(OUTPUT_PATH, f"{symbol}.parquet")
            df.to_parquet(save_path, index=False)

            results[symbol] = {
                'success': True,
                'rows': len(df),
                'path': save_path,
                'date_range': (df['timestamp'].min(), df['timestamp'].max())
            }

            print(f"  ✅ Success: {len(df):,} rows")
            print(f"     First: {df['timestamp'].min()}")
            print(f"     Last:  {df['timestamp'].max()}")
            print(f"     Saved to: {save_path}")

        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            print(f"  ❌ Error: {e}")
            print(f"     Note: If you see 'Subscription' error, you may need a paid Alpaca plan")
            print(f"           for 1-minute stock data. Try TimeFrame.Hour or TimeFrame.Day instead.")

    return results, OUTPUT_PATH


# ============================================================================
# TEIL 2: FRED M2-DATEN (NEU)
# ============================================================================

def fetch_fred_m2(keys, params, output_path):
    """Lädt M2-Geldmenge von FRED API"""

    print("\n" + "=" * 70)
    print("PART 2: Fetching M2 Money Supply from FRED")
    print("=" * 70)

    # FRED API Key prüfen
    fred_api_key = keys['KEYS'].get('FRED-API-KEY')

    if not fred_api_key:
        print("\n⚠️  FRED-API-KEY not found in conf/keys.yaml")
        print("\nTo add M2 data, please:")
        print("1. Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Add to conf/keys.yaml:")
        print("   KEYS:")
        print("     FRED-API-KEY: 'your_api_key_here'")
        print("\nSkipping M2 data for now...")
        return {'success': False, 'error': 'No API key'}

    # Parameter
    START_DATE = params['DATA_ACQUISITON']['START_DATE']
    END_DATE = params['DATA_ACQUISITON']['END_DATE']

    # FRED API Endpoint
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    # M2-Serie: WM2NS (Weekly M2 Money Stock)
    # Wöchentlich ist besser als monatlich für 1-Minuten-Trading
    series_id = 'WM2NS'

    params_api = {
        'series_id': series_id,
        'api_key': fred_api_key,
        'file_type': 'json',
        'observation_start': START_DATE,
        'observation_end': END_DATE
    }

    print(f"\nFetching FRED series: {series_id}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("-" * 70)

    try:
        # API Call
        response = requests.get(base_url, params=params_api)

        if response.status_code != 200:
            raise Exception(f"FRED API Error: {response.status_code} - {response.text}")

        data = response.json()

        if 'observations' not in data:
            raise Exception(f"No observations found in FRED response")

        # In DataFrame konvertieren
        observations = data['observations']
        df = pd.DataFrame(observations)

        # Datum parsen
        df['timestamp'] = pd.to_datetime(df['date'])

        # --> DIESE ZEILE EINFÜGEN:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

        # Wert zu float konvertieren (FRED gibt manchmal '.' für fehlende Werte)
        df['close'] = pd.to_numeric(df['value'], errors='coerce')

        # Fehlende Werte entfernen
        df = df.dropna(subset=['close'])

        # Nur relevante Spalten behalten (gleiche Struktur wie ETF-Daten)
        df = df[['timestamp', 'close']].copy()

        # Speichern (im gleichen Ordner wie ETF-Daten)
        save_path = os.path.join(output_path, 'M2.parquet')
        df.to_parquet(save_path, index=False)

        print(f"  ✅ Success: {len(df):,} observations")
        print(f"     First: {df['timestamp'].min()}")
        print(f"     Last:  {df['timestamp'].max()}")
        print(f"     Latest M2 value: ${df['close'].iloc[-1]:,.2f} Billion")
        print(f"     Saved to: {save_path}")

        return {
            'success': True,
            'rows': len(df),
            'path': save_path,
            'date_range': (df['timestamp'].min(), df['timestamp'].max())
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {'success': False, 'error': str(e)}


# ============================================================================
# TEIL 3: YFinance Zinsen-DATEN (NEU)
# ============================================================================

# --- METHODE 3: YFINANCE (Zinsen) ---
def fetch_yfinance_rates(params, OUTPUT_PATH):
    print("\n--- [3/3] YFinance: US-Zinsen (Treasury Yields) ---")
    # ^TNX = CBOE Interest Rate 10 Year Treasury Note
    ticker = "^TNX"

    # Parameter
    START_DATE = datetime.strptime(params['DATA_ACQUISITON']['START_DATE'], "%Y-%m-%d")
    END_DATE = datetime.strptime(params['DATA_ACQUISITON']['END_DATE'], "%Y-%m-%d")


    try:
        print(f"   Lade {ticker}...")
        # Intervall 1d (Zinsen gibt es nicht minütlich via yfinance free)
        df = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1d", progress=False)

        # Yahoo MultiIndex Fix
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        # Wir brauchen nur Datum und Close (Yield)
        df = df[['Date', 'Close']].rename(columns={'Date': 'timestamp', 'Close': 'US_10Y_YIELD'})

        # Zeitzone (Yahoo liefert oft naive oder lokale Zeit -> UTC)
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

        # Speichern
        save_path = os.path.join(OUTPUT_PATH, "US_INTEREST_RATES.parquet")
        df.to_parquet(save_path, index=False)
        print(f"   ✅ US-Zinsen gespeichert ({len(df)} Zeilen, täglich)")

        return {
            'success': True,
            'rows': len(df),
            'path': save_path,
            'date_range': (df['timestamp'].min(), df['timestamp'].max())
        }

    except Exception as e:
        print(f"   ❌ Fehler bei YFinance Zinsen: {e}")
        return {'success': False, 'error': str(e)}

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Hauptfunktion"""

    print("=" * 70)
    print("External Data Acquisition (ETFs + M2)")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Konfiguration laden
    keys, params, script_dir = load_config()

    # Teil 1: Alpaca ETF-Daten
    alpaca_results, output_path = fetch_alpaca_etfs(keys, params, script_dir)

    # Teil 2: FRED M2-Daten
    m2_result = fetch_fred_m2(keys, params, output_path)

    # Teil 3: YFinance Zinsen-Daten
    interest_result = fetch_yfinance_rates(params, output_path)

    # Zusammenfassung
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nAlpaca ETF Data:")
    for symbol, result in alpaca_results.items():
        if result['success']:
            print(f"  ✅ {symbol}: {result['rows']:,} rows")
        else:
            print(f"  ❌ {symbol}: {result['error']}")

    print("\nFRED M2 Data:")
    if m2_result['success']:
        print(f"  ✅ M2: {m2_result['rows']:,} observations")
    else:
        print(f"  ⚠️  M2: {m2_result['error']}")

    print("\nYFinance Interest Data:")
    if interest_result['success']:
        print(f"  ✅ Yields: {interest_result['rows']:,} observations")
    else:
        print(f"  ⚠️  Yields: {interest_result['error']}")

    print(f"\n💾 All data saved to: {output_path}")

    # Nächste Schritte
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Check the data files in:", output_path)
    print("2. Run merge_all_data.py to combine BTC + External data")
    print("3. Run data_cleaning.py for data quality checks")

    print(f"\n✅ External data acquisition completed!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()