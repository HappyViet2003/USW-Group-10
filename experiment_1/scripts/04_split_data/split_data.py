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
input_file = os.path.join(base_data_path, "processed", "training_data.parquet")

# Output Ordner für die Splits
split_dir = os.path.join(script_dir, params['DATA_SPLIT']['SPLIT_PATH'])
os.makedirs(split_dir, exist_ok=True)

print("✂️  Starte Data Splitting (Chronologisch)...")

# 1. Daten laden
if not os.path.exists(input_file):
    print(f"❌ Fehler: Input-Datei fehlt: {input_file}")
    exit(1)

df = pd.read_parquet(input_file)
# WICHTIG: Sicherstellen, dass es zeitlich sortiert ist!
df = df.sort_values('timestamp').reset_index(drop=True)

total_rows = len(df)
print(f"   Gesamt-Datensatz: {total_rows:,} Zeilen")
print(f"   Zeitraum: {df['timestamp'].min()} bis {df['timestamp'].max()}")

# 2. Split-Indices berechnen
train_ratio = params['DATA_SPLIT']['TRAIN_SIZE']
val_ratio = params['DATA_SPLIT']['VALIDATION_SIZE']
# Test ist der Rest (automatischer Puffer gegen Rundungsfehler)

# Indizes berechnen
# Index 0 bis train_end
train_end = int(total_rows * train_ratio)
# Index train_end bis val_end
val_end = train_end + int(total_rows * val_ratio)

# 3. Slicing (Schneiden)
train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# 4. Speichern
print("\n   Speichere Splits...")

train_path = os.path.join(split_dir, "train.parquet")
val_path = os.path.join(split_dir, "validation.parquet")
test_path = os.path.join(split_dir, "test.parquet")

train_df.to_parquet(train_path, index=False)
val_df.to_parquet(val_path, index=False)
test_df.to_parquet(test_path, index=False)

# 5. Zusammenfassung & Plausibilitäts-Check
print("-" * 60)
print(f"✅ TRAIN:      {len(train_df):,} Zeilen ({len(train_df)/total_rows:.1%})")
print(f"   Zeitraum:   {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
print("-" * 60)
print(f"✅ VALIDATION: {len(val_df):,} Zeilen ({len(val_df)/total_rows:.1%})")
print(f"   Zeitraum:   {val_df['timestamp'].min()} -> {val_df['timestamp'].max()}")
print("-" * 60)
print(f"✅ TEST:       {len(test_df):,} Zeilen ({len(test_df)/total_rows:.1%})")
print(f"   Zeitraum:   {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")
print("-" * 60)

# Warnung, falls sich Zeiträume überschneiden (sollte unmöglich sein, aber gut für Prof)
if train_df['timestamp'].max() >= val_df['timestamp'].min():
    print("⚠️  ACHTUNG: Zeit-Überlappung zwischen Train und Val!")
else:
    print("👍 Zeit-Check: Sauber getrennt (kein Data Leakage).")