"""
05_preparation/prepare_for_modeling.py

Bereitet die gesplitteten Daten für das Modeling vor:
1. Data Cleaning (NaN, Inf entfernen)
2. Feature Selection (Korrelation, Varianz)
3. Scaling (StandardScaler)
4. Sicherstellung, dass NUR Zahlen verwendet werden (Text-Filter)
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pickle # Wichtig zum Speichern des Scalers/Selectors

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
split_dir = os.path.join(script_dir, params['DATA_SPLIT']['SPLIT_PATH'])

# Output Ordner
prep_dir = os.path.join(base_data_path, "processed", "prepared")
os.makedirs(prep_dir, exist_ok=True)
models_dir = os.path.join(script_dir, "../../models") # Zum Speichern des Scalers
os.makedirs(models_dir, exist_ok=True)

print("=" * 70)
print("PREPARATION FOR MODELING (Robust)")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================
print("\n[1/5] Lade Train/Val/Test Splits...")

train_path = os.path.join(split_dir, "train.parquet")
val_path = os.path.join(split_dir, "validation.parquet")
test_path = os.path.join(split_dir, "test.parquet")

if not os.path.exists(train_path):
    print("❌ Fehler: Train-Daten fehlen. Führe erst 'split_data.py' aus.")
    exit(1)

train = pd.read_parquet(train_path)
val = pd.read_parquet(val_path)
test = pd.read_parquet(test_path)

print(f"   Train: {len(train):,} rows")
print(f"   Val:   {len(val):,} rows")
print(f"   Test:  {len(test):,} rows")

# ==============================================================================
# 2. FEATURE EXTRAKTION & FILTERUNG (BUGFIX)
# ==============================================================================
print("\n[2/5] Trenne Features & Targets (Filtere Text-Spalten)...")

# Spalten, die KEINE Features sind
exclude_cols = ['timestamp', 'target', 'sample_weight', 'future_close', 'year_month']

# Targets sichern
y_train = train['target']
y_val = val['target']
y_test = test['target']

# Weights sichern (falls vorhanden)
w_train = train['sample_weight'] if 'sample_weight' in train.columns else None

# Features isolieren (DROP)
X_train_raw = train.drop(columns=exclude_cols, errors='ignore')
X_val_raw = val.drop(columns=exclude_cols, errors='ignore')
X_test_raw = test.drop(columns=exclude_cols, errors='ignore')

# --- WICHTIG: NUR NUMERISCHE SPALTEN BEHALTEN ---
# Das behebt den "Extreme Greed" Fehler
X_train = X_train_raw.select_dtypes(include=[np.number])
X_val = X_val_raw.select_dtypes(include=[np.number])
X_test = X_test_raw.select_dtypes(include=[np.number])

dropped_cols = set(X_train_raw.columns) - set(X_train.columns)
if dropped_cols:
    print(f"⚠️  Gefilterte Text-Spalten (werden ignoriert): {dropped_cols}")
else:
    print("✅  Alle Spalten sind numerisch.")

# ==============================================================================
# 3. DATA CLEANING (NaN / Inf)
# ==============================================================================
print("\n[3/5] Data Cleaning (NaN/Inf entfernen)...")

def clean_data(df):
    # Ersetze unendliche Werte mit NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fülle NaNs mit 0 (oder Mittelwert, aber 0 ist sicherer bei Returns)
    df = df.fillna(0)
    return df

X_train = clean_data(X_train)
X_val = clean_data(X_val)
X_test = clean_data(X_test)

print("   ✅ Bereinigung abgeschlossen.")

# ==============================================================================
# 4. FEATURE SELECTION (Variance Threshold)
# ==============================================================================
print("\n[4/5] Feature Selection (Entferne konstante Features)...")
print(f"   Features vor Filter: {X_train.shape[1]}")

# Entferne Features mit 0 Varianz (konstante Werte)
selector = VarianceThreshold(threshold=0)
selector.fit(X_train)

# Wende auf alle Sets an (Wichtig: Transform liefert numpy array, wir wollen DataFrame)
features_selected = X_train.columns[selector.get_support()]

X_train = pd.DataFrame(selector.transform(X_train), columns=features_selected, index=X_train.index)
X_val = pd.DataFrame(selector.transform(X_val), columns=features_selected, index=X_val.index)
X_test = pd.DataFrame(selector.transform(X_test), columns=features_selected, index=X_test.index)

print(f"   Features nach Filter: {X_train.shape[1]}")

# ==============================================================================
# 5. SCALING (StandardScaler)
# ==============================================================================
print("\n[5/5] Scaling (Z-Score Normalisierung)...")

scaler = StandardScaler()
# Fit nur auf Train!
scaler.fit(X_train)

# Transform auf alle
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Target & Meta-Daten wieder anhängen (fürs Speichern)
# Wir nutzen assign, um Index-Probleme zu vermeiden
train_prepared = X_train_scaled.assign(target=y_train)
val_prepared = X_val_scaled.assign(target=y_val)
test_prepared = X_test_scaled.assign(target=y_test)

if w_train is not None:
    train_prepared['sample_weight'] = w_train

# ==============================================================================
# SPEICHERN
# ==============================================================================
print("\n💾 Speichere vorbereitete Daten...")

train_out = os.path.join(prep_dir, "train_prepared.parquet")
val_out = os.path.join(prep_dir, "val_prepared.parquet")
test_out = os.path.join(prep_dir, "test_prepared.parquet")

train_prepared.to_parquet(train_out)
val_prepared.to_parquet(val_out)
test_prepared.to_parquet(test_out)

# Speichere Feature-Liste (Wichtig für XGBoost später)
feature_list_path = os.path.join(prep_dir, "feature_list.txt")
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(features_selected))

# Speichere Scaler (für Deployment später wichtig)
scaler_path = os.path.join(models_dir, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

print(f"   ✅ Train: {train_out}")
print(f"   ✅ Scaler: {scaler_path}")
print("-" * 70)
print("FERTIG! Daten sind bereit für das Training (numeric-only).")
print("-" * 70)