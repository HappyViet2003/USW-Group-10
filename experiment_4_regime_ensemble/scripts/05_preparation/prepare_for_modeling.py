"""
05_preparation/prepare_for_modeling.py

FIXED VERSION:
- Behält regime_bull, regime_bear, regime_sideways bei!
- Behält timestamp bei
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pickle

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
models_dir = os.path.join(script_dir, "../../models")
os.makedirs(models_dir, exist_ok=True)

print("=" * 70)
print("PREPARATION FOR MODELING (FIXED - Keep Regime Columns)")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================
print("\n[1/5] Lade Train/Val/Test Splits...")

train = pd.read_parquet(os.path.join(split_dir, "train.parquet"))
val = pd.read_parquet(os.path.join(split_dir, "validation.parquet"))
test = pd.read_parquet(os.path.join(split_dir, "test.parquet"))

print(f"   Train: {len(train):,} rows")

# ==============================================================================
# 2. TRENNUNG & FILTERUNG
# ==============================================================================
print("\n[2/5] Trenne Features & Targets...")

# Meta-Daten sichern (werden nicht trainiert, aber später gebraucht)
meta_cols = ['timestamp', 'target', 'sample_weight', 'future_close', 'year_month']

# WICHTIG: Regime-Spalten sichern (werden NICHT skaliert!)
regime_cols = ['regime_bull', 'regime_bear', 'regime_sideways']

# Targets
y_train = train['target']
y_val = val['target']
y_test = test['target']

# Weights
w_train = train['sample_weight'] if 'sample_weight' in train.columns else None

# Regime-Spalten sichern
regime_train = train[regime_cols] if all(c in train.columns for c in regime_cols) else None
regime_val = val[regime_cols] if all(c in val.columns for c in regime_cols) else None
regime_test = test[regime_cols] if all(c in test.columns for c in regime_cols) else None

# Features: Alles außer Meta-Daten UND Regime-Spalten
exclude_cols = meta_cols + regime_cols
X_train_raw = train.drop(columns=exclude_cols, errors='ignore')
X_val_raw = val.drop(columns=exclude_cols, errors='ignore')
X_test_raw = test.drop(columns=exclude_cols, errors='ignore')

# NUR ZAHLEN (Filtert Text wie "Extreme Greed" raus)
X_train = X_train_raw.select_dtypes(include=[np.number])
X_val = X_val_raw.select_dtypes(include=[np.number])
X_test = X_test_raw.select_dtypes(include=[np.number])

print(f"   Numerische Features: {X_train.shape[1]}")
print(f"   Regime Columns: {regime_cols if regime_train is not None else 'None'}")

# ==============================================================================
# 3. DATA CLEANING & SELECTION
# ==============================================================================
print("\n[3/5] Cleaning & Selection...")

# NaN/Inf entfernen
def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

X_train = clean(X_train)
X_val = clean(X_val)
X_test = clean(X_test)

# Konstante Features entfernen
selector = VarianceThreshold(threshold=0)
selector.fit(X_train)
feat_names = X_train.columns[selector.get_support()]

X_train = pd.DataFrame(selector.transform(X_train), columns=feat_names, index=X_train.index)
X_val = pd.DataFrame(selector.transform(X_val), columns=feat_names, index=X_val.index)
X_test = pd.DataFrame(selector.transform(X_test), columns=feat_names, index=X_test.index)

print(f"   Nach Variance Threshold: {X_train.shape[1]} Features")

# ==============================================================================
# 4. SCALING
# ==============================================================================
print("\n[4/5] Scaling...")

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ==============================================================================
# 5. ZUSAMMENFÜGEN & SPEICHERN
# ==============================================================================
print("\n[5/5] Speichern (mit Timestamp & Regime Columns)...")

# Wir hängen Target, Timestamp UND Regime-Spalten wieder an!
train_prepared = X_train_scaled.assign(target=y_train, timestamp=train['timestamp'])
val_prepared = X_val_scaled.assign(target=y_val, timestamp=val['timestamp'])
test_prepared = X_test_scaled.assign(target=y_test, timestamp=test['timestamp'])

# Regime-Spalten anhängen (WICHTIG!)
if regime_train is not None:
    train_prepared = train_prepared.assign(**regime_train)
    val_prepared = val_prepared.assign(**regime_val)
    test_prepared = test_prepared.assign(**regime_test)
    print(f"   ✅ Regime columns added: {regime_cols}")

if w_train is not None:
    train_prepared['sample_weight'] = w_train

train_prepared.to_parquet(os.path.join(prep_dir, "train_prepared.parquet"))
val_prepared.to_parquet(os.path.join(prep_dir, "val_prepared.parquet"))
test_prepared.to_parquet(os.path.join(prep_dir, "test_prepared.parquet"))

# Scaler speichern
with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"\n✅ Fertig! Daten gespeichert in: {prep_dir}")
print(f"   Train: {train_prepared.shape}")
print(f"   Val: {val_prepared.shape}")
print(f"   Test: {test_prepared.shape}")