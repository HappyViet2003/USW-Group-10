import os
import yaml
import pandas as pd
import numpy as np  # ### NEU: Numpy für Logarithmus benötigt ###
import pandas_ta as ta

# --- KONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
data_path = os.path.join(base_data_path, "processed")

output_dir = os.path.join(base_data_path, "processed")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "training_data.parquet")

merged_data_path = os.path.join(base_data_path, "processed", "merged_raw_data.parquet")

df = pd.read_parquet(merged_data_path)
df = df.sort_values('timestamp')

# ==============================================================================
# ### NEU: HELPER FUNKTIONEN (Aus dem Prof-Code adaptiert) ###
# ==============================================================================
def z_norm(series, window=1440):
    """
    Z-Score Normalisierung (Rolling).
    Macht Werte vergleichbar (Skala -3 bis +3), egal ob Bitcoin bei 20k oder 90k steht.
    Window 1440 = 1 Tag (bei Minuten-Daten).
    """
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-8)

def slope(series, period=5):
    """
    Berechnet die Steigung (Geschwindigkeit) einer Kurve.
    Statt absoluten Preis nutzen wir die Änderung über 'period' Minuten.
    """
    return (series - series.shift(period)) / period
# ==============================================================================


# --- 4. FEATURE ENGINEERING ---
print("\n   Berechne Indikatoren...")

# ------------------------------------------------------------------------------
# ### NEU: 1. LOG-RETURNS (Die mathematisch saubere Rendite) ###
# ------------------------------------------------------------------------------
# Statt pct_change() nutzen wir Logarithmus -> Besser für KI-Modelle (Normalverteilung)
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['log_ret_5m'] = np.log(df['close'] / df['close'].shift(5))

# Volatilität auf Basis von Log-Returns (Rollend über 60 Min)
df['volatility_60m'] = df['log_ret'].rolling(window=60).std()


# A. Makro-Features (Verhältnisse)
if 'nq_close' in df.columns:
    df['ratio_btc_nq'] = df['close'] / df['nq_close']
    # ### NEU: Auch das Ratio normalisieren wir später! ###
elif 'qqq_close' in df.columns:
    df['ratio_btc_qqq'] = df['close'] / df['qqq_close']

if 'nvda_close' in df.columns:
    df['ratio_btc_nvda'] = df['close'] / df['nvda_close']
    df['corr_btc_nvda_1w'] = df['close'].rolling(10080).corr(df['nvda_close'])

if 'rates_US_10Y_YIELD' in df.columns:
    df['real_rate_impact'] = df['close'] / (df['rates_US_10Y_YIELD'] + 1)


# B. Technische Indikatoren
# RSI
df['rsi_14'] = ta.rsi(df['close'], length=14)
# ### NEU: RSI normalisieren (damit 80 nicht "anders" wirkt als 75) ###
df['rsi_14_norm'] = z_norm(df['rsi_14'], window=1440)

# Bollinger Bands
bb = ta.bbands(df['close'], length=20)
if bb is not None:
    df = pd.concat([df, bb], axis=1)
    # ### NEU: BB-Breite normalisieren (Zeigt "Squeeze" Phasen) ###
    if 'BBP_20_2.0' in df.columns: # %B Feature von pandas_ta
        df['bb_width_norm'] = z_norm(df['BBP_20_2.0'], window=1440)

# SMA / EMA
df['sma_50'] = ta.sma(df['close'], length=50)
df['ema_200'] = ta.ema(df['close'], length=200)

# ------------------------------------------------------------------------------
# ### NEU: 2. SLOPES (Steigungen/Trends) ###
# ------------------------------------------------------------------------------
# Zeigt dem Modell: "Wie steil geht es gerade bergauf?" (Trendstärke)
# Wir berechnen Slopes für den Preis und die Durchschnitte
df['slope_close_5'] = slope(df['close'], period=5)
df['slope_sma_50'] = slope(df['sma_50'], period=5)
df['slope_ema_200'] = slope(df['ema_200'], period=5)

# ------------------------------------------------------------------------------
# ### NEU: 3. Z-NORMALISIERUNG (Alles auf eine Skala bringen) ###
# ------------------------------------------------------------------------------
# Wir normalisieren Volumen und Slopes, damit das Modell nicht durch riesige Zahlen verwirrt wird.
print("   Wende Z-Normalisierung an...")
df['volume_norm'] = z_norm(df['volume'], window=1440)
df['slope_close_norm'] = z_norm(df['slope_close_5'], window=1440)
df['slope_sma_norm'] = z_norm(df['slope_sma_50'], window=1440)


# C. Zeit-Features
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour'] = df['timestamp'].dt.hour


# --- 5. OUTLIER DETECTION (Z-Score auf Log-Returns) ---
print("   Filtere Ausreißer (Z-Score)...")
# ### ÄNDERUNG: Wir nutzen jetzt die sauberen Log-Returns für den Filter ###
z_scores = ((df['log_ret'] - df['log_ret'].mean()) / df['log_ret'].std()).abs()
df = df[z_scores < 10]


# --- 6. TARGET VARIABLE ---
prediction_window = 60
df['future_close'] = df['close'].shift(-prediction_window)
df['target'] = (df['future_close'] > df['close']).astype(int)

# Letzte 60 Zeilen entfernen
df.dropna(subset=['future_close'], inplace=True)
df.drop(columns=['future_close'], inplace=True)

# --- FINALER CLEANUP ---
# ### ÄNDERUNG: Wir müssen jetzt mehr Zeilen am Anfang droppen ###
# Grund: Z-Norm und EMA brauchen "Anlaufzeit". Wir nehmen sicherheitshalber 1 Tag (1440 min).
print(f"   Vor dem finalen Cleanup: {len(df):,} Zeilen")
df.dropna(inplace=True)
print(f"   Nach dem finalen Cleanup: {len(df):,} Zeilen (Warmup für Z-Norm entfernt)")


# --- 7. SPEICHERN ---
print(f"\n💾 Speichere finalen Datensatz: {output_file}")
df.to_parquet(output_file, index=False)

print("-" * 50)
print(f"FERTIG! Features erstellt.")
print("Neue Profi-Features: log_ret, slope_close_norm, volume_norm, volatility_60m")
print("-" * 50)