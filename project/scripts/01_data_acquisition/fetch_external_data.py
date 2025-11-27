import os
import yaml
import pandas as pd
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

# --- 1. SETUP & KONFIGURATION (Identisch zu deinem Bitcoin-Skript) ---
try:
    print("Loading API credentials from 'conf/keys.yaml'...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    keys_path = os.path.join(script_dir, "../../conf/keys.yaml")
    params_path = os.path.join(script_dir, "../../conf/params.yaml")
except FileNotFoundError as e:
    print("Error: 'conf/keys.yaml' not found. Please create it with your Alpaca keys.")
    exit()


# 2. Load data acquisition parameters from YAML configuration file
try:
    keys = yaml.safe_load(open(keys_path))
    API_KEY = keys['KEYS']['APCA-API-KEY-ID-Data']
    SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY-Data']

    params = yaml.safe_load(open(params_path))
    base_path = params['DATA_ACQUISITON']['DATA_PATH']
    OUTPUT_PATH = os.path.join(script_dir, base_path, 'external_data_alpaca')
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    START_DATE = datetime.strptime(params['DATA_ACQUISITON']['START_DATE'], "%Y-%m-%d")
    END_DATE = datetime.strptime(params['DATA_ACQUISITON']['END_DATE'], "%Y-%m-%d")
except FileNotFoundError:
    print("Error: 'conf/params.yaml' not found. Please check your config.")
    exit()

client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# Strategische Auswahl der ETFs für Korrelations-Analyse:
# QQQ = Nasdaq 100 (Tech korreliert oft mit Krypto)
# GLD = Gold (Safe Haven Asset)
# UUP = US Dollar Index (Starker Dollar drückt oft Krypto-Preise)
symbols = ['QQQ', 'GLD', 'UUP']

print(f"Starte Alpaca-Download für ETFs: {symbols}")
print(f"Zeitraum: {START_DATE} bis {END_DATE}")

for symbol in symbols:
    print(f"Lade Daten für {symbol}...")

    # Request erstellen
    # Wir nutzen Adjustment.ALL damit Splits berücksichtigt werden
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=START_DATE,
        end=END_DATE,
        adjustment=Adjustment.ALL
    )

    try:
        # Daten holen
        bars = client.get_stock_bars(request)
        df = bars.df
        df.reset_index(inplace=True)

        # Cleaning (Symbol Spalte weg)
        if 'symbol' in df.columns:
            df.drop(columns=['symbol'], inplace=True)

        # Speichern
        save_path = os.path.join(OUTPUT_PATH, f"{symbol}.parquet")
        df.to_parquet(save_path, index=False)

        print(f"✅ {symbol}: {len(df)} Zeilen gespeichert.")
        print(f"   Erster: {df['timestamp'].min()} | Letzter: {df['timestamp'].max()}")

    except Exception as e:
        print(f"⚠️ Warnung bei {symbol}: {e}")
        print("   (Falls Fehler 'Subscription', hast du evtl. kein Zugriff auf 1-Min-US-Aktien im Free Plan.")
        print("   Versuche in dem Fall TimeFrame.Hour oder TimeFrame.Day)")

print("Externe Datenakquise fertig.")