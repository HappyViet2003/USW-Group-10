import os
import yaml
import pandas as pd
import numpy as np
import pandas_ta as ta  # pip install pandas_ta

# --- KONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
output_dir = os.path.join(base_data_path, "processed")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "training_data.parquet")

merged_data_path = os.path.join(base_data_path, "processed", "merged_raw_data.parquet")

print(f"Lade Daten von: {merged_data_path}")
df = pd.read_parquet(merged_data_path)
df = df.sort_values('timestamp')


# ==============================================================================
# HELPER FUNKTIONEN
# ==============================================================================
def z_norm(series, window=1440):
    """
    Z-Score Normalisierung (Rolling).
    Macht Werte vergleichbar (Skala ca. -3 bis +3).
    Window 1440 = 1 Tag (bei Minuten-Daten).
    """
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-8)


def slope(series, period=5):
    """Berechnet die Steigung (Geschwindigkeit) einer Kurve."""
    return (series - series.shift(period)) / period


# ==============================================================================
# FEATURE ENGINEERING (ENHANCED & SENTIMENT)
# ==============================================================================
print("\n   Berechne Indikatoren (Enhanced + Sentiment)...")

# ------------------------------------------------------------------------------
# 1. RETURNS & HISTORISCHE VOLATILITÄT
# ------------------------------------------------------------------------------
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['log_ret_5m'] = np.log(df['close'] / df['close'].shift(5))
df['log_ret_15m'] = np.log(df['close'] / df['close'].shift(15))
df['log_ret_30m'] = np.log(df['close'] / df['close'].shift(30))

df['volatility_60m'] = df['log_ret'].rolling(window=60).std()

# Historische Volatilität (sqrt Anpassung)
df['hist_vol_30'] = df['log_ret'].rolling(30).std() * np.sqrt(30)
df['hist_vol_60'] = df['log_ret'].rolling(60).std() * np.sqrt(60)
df['hist_vol_30_norm'] = z_norm(df['hist_vol_30'])

# ------------------------------------------------------------------------------
# 2. VOLUMEN-INDIKATOREN
# ------------------------------------------------------------------------------
# OBV (On-Balance Volume)
df['obv'] = ta.obv(df['close'], df['volume'])
df['obv_slope'] = slope(df['obv'], period=5)

# MFI (Money Flow Index - "RSI mit Volumen")
df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
df['mfi_14_norm'] = z_norm(df['mfi_14'])

# VWAP (Volume Weighted Average Price)
# Initialisierungstrick um NaNs am Anfang zu vermeiden
df.set_index('timestamp', inplace=True)
df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
df.reset_index(inplace=True)
df['vwap'] = df['vwap'].fillna(df['close'])

df['dist_vwap'] = (df['close'] - df['vwap']) / df['close']
df['dist_vwap_norm'] = z_norm(df['dist_vwap'])

# ------------------------------------------------------------------------------
# 3. VOLATILITÄT & TREND (ATR, BBands, ADX)
# ------------------------------------------------------------------------------
# ATR (Average True Range)
df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['atr_pct'] = df['atr_14'] / df['close']
df['atr_norm'] = z_norm(df['atr_pct'])

# Bollinger Bands
bb = ta.bbands(df['close'], length=20, std=2.0)
if bb is not None:
    df = pd.concat([df, bb], axis=1)
    # BBB = Bandwidth, BBP = %B (Position)
    if 'BBB_20_2.0' in df.columns:
        df['bb_width_norm'] = z_norm(df['BBB_20_2.0'])
    if 'BBP_20_2.0' in df.columns:
        df['bb_position'] = df['BBP_20_2.0']

# ADX (Trendstärke)
adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
if adx_df is not None:
    df['adx'] = adx_df['ADX_14']
    df['adx_norm'] = z_norm(df['adx'])

# ------------------------------------------------------------------------------
# 4. MOMENTUM & OSZILLATOREN
# ------------------------------------------------------------------------------
# RSI
df['rsi_14'] = ta.rsi(df['close'], length=14)
df['rsi_14_norm'] = z_norm(df['rsi_14'])

# Stochastic
stoch = ta.stoch(df['high'], df['low'], df['close'])
if stoch is not None:
    df = pd.concat([df, stoch], axis=1)
    if 'STOCHk_14_3_3' in df.columns:
        df['stoch_k_norm'] = z_norm(df['STOCHk_14_3_3'])

# CCI
df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
df['cci_norm'] = z_norm(df['cci'])

# Williams %R
df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)
df['willr_norm'] = z_norm(df['willr'])

# ------------------------------------------------------------------------------
# 5. SUPPORT / RESISTANCE & CANDLES (BUGFIXED: Kein Look-Ahead!)
# ------------------------------------------------------------------------------
# Lokale Hochs/Tiefs (WICHTIG: center=False, sonst Blick in die Zukunft)
df['local_high'] = df['high'].rolling(20).max()
df['local_low'] = df['low'].rolling(20).min()

df['dist_local_high'] = (df['close'] - df['local_high']) / df['close']
df['dist_local_low'] = (df['close'] - df['local_low']) / df['close']
df['dist_local_high_norm'] = z_norm(df['dist_local_high'])

# Candlestick Geometrie
df['body_size'] = np.abs(df['close'] - df['open']) / (df['close'] + 1e-8)
df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['close'] + 1e-8)
df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['close'] + 1e-8)
df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['body_size'] + 1e-8)

df['body_size_norm'] = z_norm(df['body_size'])
df['wick_ratio_norm'] = z_norm(df['wick_ratio'])

# ------------------------------------------------------------------------------
# 6. SMA/EMA & SLOPES
# ------------------------------------------------------------------------------
df['sma_50'] = ta.sma(df['close'], length=50)
df['ema_200'] = ta.ema(df['close'], length=200)

df['slope_close_5'] = slope(df['close'], period=5)
df['slope_sma_50'] = slope(df['sma_50'], period=5)
df['slope_ema_200'] = slope(df['ema_200'], period=5)

df['slope_close_norm'] = z_norm(df['slope_close_5'])
df['slope_sma_norm'] = z_norm(df['slope_sma_50'])

# ------------------------------------------------------------------------------
# 7. MAKRO-FEATURES
# ------------------------------------------------------------------------------
print("   Berechne Makro-Korrelationen...")

if 'nq_close' in df.columns:
    df['ratio_btc_nq'] = df['close'] / df['nq_close']
elif 'qqq_close' in df.columns:
    df['ratio_btc_qqq'] = df['close'] / df['qqq_close']
    # Beta zu QQQ
    btc_ret = df['close'].pct_change()
    qqq_ret = df['qqq_close'].pct_change()
    cov = btc_ret.rolling(60).cov(qqq_ret)
    var = qqq_ret.rolling(60).var()
    df['beta_qqq'] = cov / (var + 1e-8)
    df['beta_qqq_norm'] = z_norm(df['beta_qqq'])

if 'nvda_close' in df.columns:
    df['ratio_btc_nvda'] = df['close'] / df['nvda_close']
    df['corr_btc_nvda_1w'] = df['close'].rolling(10080).corr(df['nvda_close'])

if 'gld_close' in df.columns:
    df['corr_btc_gld_60m'] = df['close'].rolling(60).corr(df['gld_close'])
if 'uup_close' in df.columns:
    df['corr_btc_uup_60m'] = df['close'].rolling(60).corr(df['uup_close'])

if 'rates_US_10Y_YIELD' in df.columns:
    df['real_rate_impact'] = df['close'] / (df['rates_US_10Y_YIELD'] + 1)

# ------------------------------------------------------------------------------
# 8. SENTIMENT (NEU: FEAR & GREED)
# ------------------------------------------------------------------------------
# Prüfen, ob durch merge_all_data.py eine Sentiment-Spalte da ist.
# Meistens heißt sie 'sentiment_fear_greed' oder 'sentiment_value'.
# Wir suchen dynamisch danach.
sentiment_col = [c for c in df.columns if 'fear_greed' in c or 'sentiment' in c]

if sentiment_col:
    col_name = sentiment_col[0]
    print(f"   Integriere Sentiment-Daten ({col_name})...")

    # 1. Forward Fill (da Daten nur täglich kommen, wir aber Minuten haben)
    df[col_name] = df[col_name].ffill()

    # 2. Änderung zum Vortag (1440 Minuten)
    df['sentiment_change'] = df[col_name].diff(1440)

    # 3. Normalisierung (30-Tage Fenster für Kontext)
    df['sentiment_norm'] = z_norm(df[col_name], window=1440 * 30)
else:
    print("⚠️  Keine Sentiment-Daten gefunden (FEAR_GREED). Überspringe.")

# ------------------------------------------------------------------------------
# 9. ZEIT-FEATURES & OUTLIER
# ------------------------------------------------------------------------------
print("   Erstelle Zeit-Features & Targets...")
df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)

df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(float)
df['is_us_trading'] = ((df['timestamp'].dt.hour >= 14) & (df['timestamp'].dt.hour < 21)).astype(float)

df['volume_norm'] = z_norm(df['volume'])

# Outlier Filter (Z-Score auf Log-Returns)
z_scores = ((df['log_ret'] - df['log_ret'].mean()) / df['log_ret'].std()).abs()
df = df[z_scores < 10]

# ------------------------------------------------------------------------------
# 10. TARGET VARIABLE & CLEANUP
# ------------------------------------------------------------------------------
prediction_window = 60
df['future_close'] = df['close'].shift(-prediction_window)
df['target'] = (df['future_close'] > df['close']).astype(int)

df.dropna(subset=['future_close'], inplace=True)
df.drop(columns=['future_close'], inplace=True)

# ------------------------------------------------------------------------------
# 11. SAMPLE WEIGHTS (MONATSBASIERT)
# ------------------------------------------------------------------------------
print("   Berechne Sample Weights (Treppen-Funktion)...")
df['year_month'] = df['timestamp'].dt.to_period('M')
unique_months = df['year_month'].unique()
num_months = len(unique_months)

# Gewichte von 0.5 (alt) bis 1.5 (neu)
month_weights = np.linspace(0.5, 1.5, num_months)
month_to_weight = dict(zip(unique_months, month_weights))

df['sample_weight'] = df['year_month'].map(month_to_weight)
df = df.drop(columns=['year_month'])

# Final Cleanup
print(f"   Vor Cleanup: {len(df):,} Zeilen")
df.dropna(inplace=True)
print(f"   Nach Cleanup: {len(df):,} Zeilen")

# Speichern
print(f"\n💾 Speichere finalen Datensatz: {output_file}")
df.to_parquet(output_file, index=False)

print("-" * 50)
print(f"FERTIG! Enhanced Features + Sentiment + Bugfix aktiv.")
print("-" * 50)