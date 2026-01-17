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
output_file = os.path.join(output_dir, "merged_raw_data.parquet")

print("🚀 Starte Advanced Feature Engineering & Merging (inkl. On-Chain)...")

# --- 1. DATEN LADEN ---
# A. Bitcoin (Basis)
if not os.path.exists(crypto_path):
    print(f"❌ CRITICAL: BTC Datei fehlt: {crypto_path}")
    exit(1)

print("   Lade Bitcoin Basis-Daten...")
df = pd.read_parquet(crypto_path)
df = df.drop_duplicates(subset=['timestamp'])
df = df.sort_values('timestamp')
print(f"   ✅ BTC: {len(df):,} Zeilen")

# B. Externe Daten Definitionen
# (Dateiname -> Prefix für Spalten)
external_sources = {
    # Bestehende Daten
    'QQQ.parquet': 'qqq',
    'NASDAQ_FUTURE.parquet': 'nq',
    'GLD.parquet': 'gold',
    'UUP.parquet': 'usd',
    'M2.parquet': 'm2',
    'US_INTEREST_RATES.parquet': 'rates',
    'NVDA.parquet': 'nvda',
    'FEAR_GREED.parquet': 'sentiment',

    # --- NEU: On-Chain Daten ---
    'onchain_hashrate.parquet': 'onchain',
    'onchain_active_addresses.parquet': 'onchain'
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
        df = pd.merge_asof(
            df,
            df_ext,
            on='timestamp',
            direction='backward'  # Nimm den letzten bekannten Wert
        )
    else:
        print(f"⚠️  Warnung: Datei {filename} nicht gefunden in {external_path}")

# --- 3. DATENBEREINIGUNG ---
print("\n   Bereinigung...")
initial_len = len(df)

# Forward Fill für verbleibende Lücken (z.B. Feiertage oder fehlende On-Chain Tage)
df.ffill(inplace=True)

# Am Anfang fehlen oft externe Daten, diese Zeilen löschen
df.dropna(inplace=True)
print(f"   Zeilen nach DropNA: {len(df):,} (Start-Lücken entfernt)")

# --- SPEICHERN ---
print(f"\n💾 Speichere finalen Datensatz: {output_file}")
df.to_parquet(output_file, index=False)

print("-" * 50)
print(f"FERTIG! Finaler Datensatz: {len(df):,} Zeilen, {len(df.columns)} Spalten")
# Kurzer Check ob Onchain dabei ist
cols = [c for c in df.columns if 'onchain' in c]
print(f"Enthält On-Chain Features: {cols[:3]} ... ({len(cols)} total)")
print("-" * 50)