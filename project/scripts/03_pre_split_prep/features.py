import os
import yaml
import pandas as pd
import numpy as np
import pandas_ta as ta

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
    """Z-Score Normalisierung (Rolling)."""
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-8)

def slope(series, period=5):
    """Berechnet die Steigung (Geschwindigkeit) einer Kurve."""
    return (series - series.shift(period)) / period

# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================
print("\n   Berechne Indikatoren...")

# 1. LOG-RETURNS
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['log_ret_5m'] = np.log(df['close'] / df['close'].shift(5))
df['volatility_60m'] = df['log_ret'].rolling(window=60).std()

# A. Makro-Features
if 'nq_close' in df.columns:
    df['ratio_btc_nq'] = df['close'] / df['nq_close']
elif 'qqq_close' in df.columns:
    df['ratio_btc_qqq'] = df['close'] / df['qqq_close']

if 'nvda_close' in df.columns:
    df['ratio_btc_nvda'] = df['close'] / df['nvda_close']
    df['corr_btc_nvda_1w'] = df['close'].rolling(10080).corr(df['nvda_close'])

if 'rates_US_10Y_YIELD' in df.columns:
    df['real_rate_impact'] = df['close'] / (df['rates_US_10Y_YIELD'] + 1)

# B. Technische Indikatoren
df['rsi_14'] = ta.rsi(df['close'], length=14)
df['rsi_14_norm'] = z_norm(df['rsi_14'], window=1440)

bb = ta.bbands(df['close'], length=20)
if bb is not None:
    df = pd.concat([df, bb], axis=1)
    if 'BBP_20_2.0' in df.columns:
        df['bb_width_norm'] = z_norm(df['BBP_20_2.0'], window=1440)

df['sma_50'] = ta.sma(df['close'], length=50)
df['ema_200'] = ta.ema(df['close'], length=200)

# 2. SLOPES
df['slope_close_5'] = slope(df['close'], period=5)
df['slope_sma_50'] = slope(df['sma_50'], period=5)
df['slope_ema_200'] = slope(df['ema_200'], period=5)

# 3. Z-NORMALISIERUNG
print("   Wende Z-Normalisierung an...")
df['volume_norm'] = z_norm(df['volume'], window=1440)
df['slope_close_norm'] = z_norm(df['slope_close_5'], window=1440)
df['slope_sma_norm'] = z_norm(df['slope_sma_50'], window=1440)

# C. Zeit-Features
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour'] = df['timestamp'].dt.hour

# --- 5. OUTLIER DETECTION ---
print("   Filtere Ausreißer (Z-Score)...")
z_scores = ((df['log_ret'] - df['log_ret'].mean()) / df['log_ret'].std()).abs()
#df = df[z_scores < 10]
# Statt löschen: Werte auf +/- 10 Sigma begrenzen
df['log_ret'] = df['log_ret'].clip(lower=-10 * df['log_ret'].std(),
                                   upper= 10 * df['log_ret'].std())

# --- 6. TARGET VARIABLE ---
prediction_window = 60
df['future_close'] = df['close'].shift(-prediction_window)
df['target'] = (df['future_close'] > df['close']).astype(int)

# Aufräumen (Zukunft entfernen)
df.dropna(subset=['future_close'], inplace=True)
df.drop(columns=['future_close'], inplace=True)

# --- 7. SAMPLE WEIGHTS (Monatsbasiert) ---
# Gewichtung nach Monat: Ältere Monate = niedrigere Gewichtung, neuere Monate = höhere Gewichtung
# Das sorgt dafür, dass der "Concept Drift" berücksichtigt wird.
print("   Berechne Zeit-Gewichtung (Sample Weights nach Monat)...")

# Erstelle Jahr-Monat-Spalte
df['year_month'] = df['timestamp'].dt.to_period('M')

# Zähle eindeutige Monate und weise jedem eine Gewichtung zu
unique_months = df['year_month'].unique()
num_months = len(unique_months)

# Erstelle Mapping: Monat → Gewichtung (linear von 0.5 bis 1.5)
month_weights = np.linspace(0.5, 1.5, num_months)
month_to_weight = dict(zip(unique_months, month_weights))

# Weise jedem Datensatz die Gewichtung seines Monats zu
df['sample_weight'] = df['year_month'].map(month_to_weight)

# Entferne temporäre Spalte
df = df.drop(columns=['year_month'])

print(f"   ✅ {num_months} Monate gewichtet (Min: {df['sample_weight'].min():.2f}, Max: {df['sample_weight'].max():.2f})")

# --- FINALER CLEANUP ---
print(f"   Vor dem finalen Cleanup: {len(df):,} Zeilen")
df.dropna(inplace=True)
print(f"   Nach dem finalen Cleanup: {len(df):,} Zeilen")

# --- 8. SPEICHERN ---
print(f"\n💾 Speichere finalen Datensatz: {output_file}")
df.to_parquet(output_file, index=False)

print("-" * 50)
print(f"FERTIG! Features erstellt.")
print(f"Sample Weights hinzugefügt (Min: {df['sample_weight'].min():.2f}, Max: {df['sample_weight'].max():.2f})")
print("-" * 50)