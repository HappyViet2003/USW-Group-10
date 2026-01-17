"""
03_pre_split_prep/features.py

Experiment 2:
Berechnet technische Indikatoren UND On-Chain Features.
Wichtig: Wandelt absolute On-Chain Werte in relative Änderungen um (pct_change).
"""

import os
import yaml
import pandas as pd
import pandas_ta as ta
import numpy as np

# --- KONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Pfade
base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
processed_dir = os.path.join(base_data_path, "processed")

print("=" * 70)
print("FEATURE ENGINEERING (EXP 2: ON-CHAIN)")
print("=" * 70)

# 1. Daten laden
file_path = os.path.join(processed_dir, "merged_raw_data.parquet")
if not os.path.exists(file_path):
    print(f"❌ Datei nicht gefunden: {file_path}")
    exit(1)

df = pd.read_parquet(file_path)
df = df.sort_values('timestamp')

print(f"   Input Shape: {df.shape}")

# 2. Target generieren (Zukunft)
# Wir wollen vorhersagen: Ist Close in 60 Min höher als jetzt?
df['future_close'] = df['close'].shift(-60)
df['target'] = (df['future_close'] > df['close']).astype(int)

# 3. Basis Features (Returns)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# 4. Technische Indikatoren (Pandas TA)
# RSI, ATR, VWAP Distance
df['rsi_14'] = ta.rsi(df['close'], length=14)
df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['atr_pct'] = df['atr_14'] / df['close'] # Normalisierte Volatilität
df['dist_vwap'] = (df['close'] - df['vwap']) / df['vwap']

# Slopes (Steigungen von Durchschnitten)
df['sma_50'] = ta.sma(df['close'], length=50)
df['slope_sma_50'] = ta.slope(df['sma_50'], length=5)

# 5. Externe Features (Stocks & Macro)
# Rolling Correlation (Beta) zu QQQ (Tech Sector)
if 'qqq_close' in df.columns:
    rolling_corr = df['close'].rolling(60).corr(df['qqq_close'])
    df['beta_qqq'] = rolling_corr.fillna(0)

# 6. --- NEU: ON-CHAIN FEATURES ---
# Hashrate und Adressen sind absolute Werte. Wir brauchen die Änderung.
# On-Chain Daten ändern sich langsam (oft nur 1x am Tag aktualisiert).
# Wir nehmen die Änderung über 24h (1440 Minuten).

if 'onchain_hash-rate' in df.columns:
    print("   ✅ Processing On-Chain: Hashrate")
    # Prozentuale Änderung der Hashrate (24h)
    df['hashrate_change_24h'] = df['onchain_hash-rate'].pct_change(1440)
    # Entferne NaN (am Anfang)
    df['hashrate_change_24h'] = df['hashrate_change_24h'].fillna(0)

if 'onchain_n-unique-addresses' in df.columns:
    print("   ✅ Processing On-Chain: Active Addresses")
    # Prozentuale Änderung der aktiven Adressen (24h)
    df['addresses_change_24h'] = df['onchain_n-unique-addresses'].pct_change(1440)
    df['addresses_change_24h'] = df['addresses_change_24h'].fillna(0)

# 7. Zeit-Features (Cyclical Encoding)
# Damit das Modell Tageszeiten lernt (z.B. US-Market Open)
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# 8. Sample Weights (Concept Drift)
# Neuere Daten bekommen mehr Gewicht im Training
time_idx = np.arange(len(df))
df['sample_weight'] = 0.5 + (time_idx / len(df))

# Cleanup
df = df.dropna()
df = df.sort_values('timestamp')

# Speichern
output_path = os.path.join(processed_dir, "training_data.parquet")

df.to_parquet(output_path)

print(f"\n✅ Feature Engineering abgeschlossen.")
print(f"   Output: {output_path}")
print(f"   Shape: {df.shape}")
print(f"   On-Chain Features dabei? {'hashrate_change_24h' in df.columns}")