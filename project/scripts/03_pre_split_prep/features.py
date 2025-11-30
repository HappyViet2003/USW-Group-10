import os
import yaml
import pandas as pd
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

# --- 4. FEATURE ENGINEERING ---
print("\n   Berechne Indikatoren...")

# A. Makro-Features (Verhältnisse)
# Wie stark ist Bitcoin im Vergleich zu Tech-Aktien?
# Wir prüfen jetzt auf 'nq_close' (Future) statt 'qqq_close' (ETF)

if 'nq_close' in df.columns:
    # Verhältnis Bitcoin zu Nasdaq Future
    df['ratio_btc_nq'] = df['close'] / df['nq_close']
    print("   Feature erstellt: ratio_btc_nq (basierend auf Futures)")

elif 'qqq_close' in df.columns:
    # Fallback: Falls Futures fehlen, nimm QQQ
    df['ratio_btc_qqq'] = df['close'] / df['qqq_close']
    print("   Feature erstellt: ratio_btc_qqq (basierend auf ETF)")

if 'nvda_close' in df.columns:
    # NEU: Das "AI-Rotation-Ratio"
    # Zeigt an: Ist Bitcoin stärker als der AI-Hype?
    df['ratio_btc_nvda'] = df['close'] / df['nvda_close']

    # NEU: Hype-Korrelation (rollend über 30 Tage / 43200 Minuten)
    # Wenn Korrelation hoch ist (nahe 1), bewegen sich beide im Gleichschritt (Risk-On)
    # Wir nehmen hier ein kürzeres Fenster, z.B. 1 Woche (ca. 10.000 Minuten)
    df['corr_btc_nvda_1w'] = df['close'].rolling(10080).corr(df['nvda_close'])

    print("   Feature erstellt: ratio_btc_nvda & corr_btc_nvda")

# Wie stark drückt der Zins?
if 'rates_US_10Y_YIELD' in df.columns:
    # Zinsen sind in Prozent (z.B. 4.5), wir skalieren das
    df['real_rate_impact'] = df['close'] / (df['rates_US_10Y_YIELD'] + 1)

# B. Technische Indikatoren (pandas_ta)
# RSI
df['rsi_14'] = ta.rsi(df['close'], length=14)

# Bollinger Bands
bb = ta.bbands(df['close'], length=20)
if bb is not None:
    df = pd.concat([df, bb], axis=1)

# SMA / EMA
df['sma_50'] = ta.sma(df['close'], length=50)
df['ema_200'] = ta.ema(df['close'], length=200)

# C. Zeit-Features (für das Wochenend-Problem)
# Damit das Modell lernt: "Sonntags passiert beim Nasdaq nichts"
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour'] = df['timestamp'].dt.hour

# --- 5. OUTLIER DETECTION (Z-Score) ---
# Partner-Wunsch: Extreme Spikes entfernen
print("   Filtere Ausreißer (Z-Score)...")
df['returns'] = df['close'].pct_change()
z_scores = ((df['returns'] - df['returns'].mean()) / df['returns'].std()).abs()
# Wir entfernen alles über 10 Sigma (extreme Datenfehler)
df = df[z_scores < 10]

# --- 6. TARGET VARIABLE ---
# Ziel: Steigt der Kurs in 60 Minuten?
prediction_window = 60
df['future_close'] = df['close'].shift(-prediction_window)
df['target'] = (df['future_close'] > df['close']).astype(int)

# Letzte 60 Zeilen entfernen (haben kein Target)
df.dropna(subset=['future_close'], inplace=True)
df.drop(columns=['future_close'], inplace=True)  # Nicht zum Training nutzen!

# --- NEU: FINALER CLEANUP ---
# Entfernt die ersten ~200 Zeilen, wo SMA/EMA noch "aufwärmen" (NaN sind)
print(f"   Vor dem finalen Cleanup: {len(df):,} Zeilen")
df.dropna(inplace=True)
print(f"   Nach dem finalen Cleanup: {len(df):,} Zeilen (Indikator-Warmup entfernt)")

# --- 7. SPEICHERN ---
print(f"\n💾 Speichere finalen Datensatz: {output_file}")
df.to_parquet(output_file, index=False)

print("-" * 50)
print(f"FERTIG! Finaler Datensatz: {len(df):,} Zeilen, {len(df.columns)} Spalten")
print("Enthält:", [c for c in df.columns if 'm2' in c or 'rates' in c or 'qqq' in c])
print("-" * 50)

