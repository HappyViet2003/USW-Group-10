import os
import yaml
import pandas as pd
import numpy as np
import pandas_ta as ta

# --- KONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Pfade
base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])  # ../../data
crypto_path = os.path.join(base_data_path, "Bars_1m_crypto", "BTC_USD.parquet")

# WICHTIG: Hier liegen jetzt ALLE deine externen Daten (laut deinem Log)
external_path = os.path.join(base_data_path, "external_data")

output_dir = os.path.join(base_data_path, "processed")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "training_data_final.parquet")

print("🚀 Starte Advanced Feature Engineering & Merging...")

# --- 1. DATEN LADEN ---
# A. Bitcoin (Basis)
if not os.path.exists(crypto_path):
    print(f"❌ CRITICAL: BTC Datei fehlt: {crypto_path}")
    exit(1)

print("   Lade Bitcoin Basis-Daten...")
df = pd.read_parquet(crypto_path)
df = df.sort_values('timestamp')
print(f"   ✅ BTC: {len(df):,} Zeilen")

# B. Externe Daten Definitionen
# (Dateiname -> Prefix für Spalten)
external_sources = {
    'QQQ.parquet': 'qqq',
    'GLD.parquet': 'gold',
    'UUP.parquet': 'usd',
    'M2.parquet': 'm2',  # Wöchentlich
    'US_INTEREST_RATES.parquet': 'rates'  # Täglich
}

# --- 2. INTELLIGENTES MERGING (Merge AsOf) ---
print("\n   Füge externe Daten hinzu (Merge AsOf)...")

for filename, prefix in external_sources.items():
    path = os.path.join(external_path, filename)

    if os.path.exists(path):
        print(f"   Processing {filename}...")
        df_ext = pd.read_parquet(path)

        # Spalten umbenennen (close -> qqq_close)
        # Wir behalten alle Spalten außer timestamp zum Mergen
        cols_to_rename = {c: f"{prefix}_{c}" for c in df_ext.columns if c != 'timestamp'}
        df_ext.rename(columns=cols_to_rename, inplace=True)

        # Sortieren ist Pflicht für merge_asof
        df_ext = df_ext.sort_values('timestamp')

        # MERGE ASOF:
        # Findet für jeden BTC-Zeitpunkt den aktuellsten verfügbaren Wert der externen Quelle.
        # Das löst das Problem "M2 kommt nur wöchentlich" automatisch (Werte werden wiederholt).
        df = pd.merge_asof(
            df,
            df_ext,
            on='timestamp',
            direction='backward'  # Nimm den letzten bekannten Wert (kein Blick in die Zukunft!)
        )
    else:
        print(f"⚠️  Warnung: Datei {filename} nicht gefunden in {external_path}")

# --- 3. DATENBEREINIGUNG ---
print("\n   Bereinigung...")
initial_len = len(df)

# Forward Fill für verbleibende Lücken (z.B. Feiertage)
df.ffill(inplace=True)

# Am Anfang fehlen oft externe Daten (bevor M2 startete), diese Zeilen löschen
df.dropna(inplace=True)
print(f"   Zeilen nach DropNA: {len(df):,} (Start-Lücken entfernt)")


##TODO Feature In seperate script?

# --- 4. FEATURE ENGINEERING ---
print("\n   Berechne Indikatoren...")

# A. Makro-Features (Verhältnisse)
# Wie stark ist Bitcoin im Vergleich zu Tech-Aktien?
if 'qqq_close' in df.columns:
    df['ratio_btc_qqq'] = df['close'] / df['qqq_close']

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