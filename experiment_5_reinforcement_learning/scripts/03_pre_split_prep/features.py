"""
03_pre_split_prep/features.py (Simplified - ohne pandas_ta)

Experiment 4: Market Regime Detection + Ensemble Model
Berechnet technische Indikatoren, On-Chain Features UND Market Regime.
"""

import os
import yaml
import pandas as pd
import numpy as np

# --- KONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Pfade
base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
processed_dir = os.path.join(base_data_path, "processed")

print("=" * 70)
print("FEATURE ENGINEERING (EXP 4: MARKET REGIME + ENSEMBLE)")
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
df['future_close'] = df['close'].shift(-60)
df['target'] = (df['future_close'] > df['close']).astype(int)

# 3. Basis Features (Returns)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# 4. Technische Indikatoren (manuell implementiert)

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-8)
df['rsi_14'] = 100 - (100 / (1 + rs))

# ATR (Average True Range)
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr_14'] = true_range.rolling(14).mean()
df['atr_pct'] = df['atr_14'] / df['close']

# VWAP Distance
df['dist_vwap'] = (df['close'] - df['vwap']) / df['vwap']

# SMA Slope
df['sma_50'] = df['close'].rolling(50).mean()
df['slope_sma_50'] = df['sma_50'].pct_change(5)

# 5. Externe Features
if 'qqq_close' in df.columns:
    rolling_corr = df['close'].rolling(60).corr(df['qqq_close'])
    df['beta_qqq'] = rolling_corr.fillna(0)

# 6. On-Chain Features
if 'onchain_hash-rate' in df.columns:
    print("   ✅ Processing On-Chain: Hashrate")
    df['hashrate_change_24h'] = df['onchain_hash-rate'].pct_change(1440)
    df['hashrate_change_24h'] = df['hashrate_change_24h'].fillna(0)

if 'onchain_n-unique-addresses' in df.columns:
    print("   ✅ Processing On-Chain: Active Addresses")
    df['addresses_change_24h'] = df['onchain_n-unique-addresses'].pct_change(1440)
    df['addresses_change_24h'] = df['addresses_change_24h'].fillna(0)

# 7. --- NEU: MARKET REGIME DETECTION ---
print("   ✅ Detecting Market Regimes...")

# Moving Averages für Regime Detection
df['ma_20'] = df['close'].rolling(20).mean()
df['ma_50'] = df['close'].rolling(50).mean()
df['ma_200'] = df['close'].rolling(200).mean()

# Trend Slope (20-day MA slope)
df['ma_slope'] = df['ma_20'].pct_change(20)

# Volatility (20-day rolling std of returns)
df['volatility'] = df['log_return'].rolling(20).std()

# Volume Trend
df['volume_ma'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma']

# Regime Classification
# Bull: MA slope > 2%, volatility < 33rd percentile, price > MA200
# Bear: MA slope < -2%, volatility > 67th percentile, price < MA200
# Sideways: Everything else

vol_33 = df['volatility'].quantile(0.33)
vol_67 = df['volatility'].quantile(0.67)

conditions = [
    (df['ma_slope'] > 0.02) & (df['volatility'] < vol_33) & (df['close'] > df['ma_200']),  # Bull
    (df['ma_slope'] < -0.02) & (df['volatility'] > vol_67) & (df['close'] < df['ma_200']), # Bear
]

df['regime'] = np.select(conditions, ['bull', 'bear'], default='sideways')

# Regime als numerische Features (One-Hot Encoding)
df['regime_bull'] = (df['regime'] == 'bull').astype(int)
df['regime_bear'] = (df['regime'] == 'bear').astype(int)
df['regime_sideways'] = (df['regime'] == 'sideways').astype(int)

# Regime Stability (wie lange im aktuellen Regime?)
df['regime_duration'] = df.groupby((df['regime'] != df['regime'].shift()).cumsum()).cumcount()

print(f"   Regime Distribution:")
print(df['regime'].value_counts())

# 8. Zeit-Features (Cyclical Encoding)
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# 9. Sample Weights (Concept Drift)
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
print(f"   Features: {len(df.columns)}")
print("=" * 70)
