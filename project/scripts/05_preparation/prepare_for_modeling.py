"""
05_preparation/prepare_for_modeling.py

Bereitet die gesplitteten Daten für das Modeling vor:
1. Data Cleaning (NaN, Inf entfernen)
2. Feature Selection (Korrelation, Multikollinearität)
3. Scaling (Optional: StandardScaler)
4. Final Check (Class Balance, Feature Count)
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
split_dir = os.path.join(script_dir, params['DATA_SPLIT']['SPLIT_PATH'])

# Output
prep_dir = os.path.join(base_data_path, "processed", "prepared")
os.makedirs(prep_dir, exist_ok=True)

print("=" * 70)
print("PREPARATION FOR MODELING")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================
print("\n[1/5] Lade Train/Val/Test Splits...")

train_df = pd.read_parquet(os.path.join(split_dir, "train.parquet"))
val_df = pd.read_parquet(os.path.join(split_dir, "validation.parquet"))
test_df = pd.read_parquet(os.path.join(split_dir, "test.parquet"))

print(f"   Train: {len(train_df):,} rows")
print(f"   Val:   {len(val_df):,} rows")
print(f"   Test:  {len(test_df):,} rows")

# ==============================================================================
# 2. DATA CLEANING
# ==============================================================================
print("\n[2/5] Data Cleaning...")

def clean_data(df, name=""):
    """Entfernt NaN, Inf und fehlerhafte Zeilen"""
    
    initial_rows = len(df)
    
    # Inf zu NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # NaN-Statistik
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    if len(nan_cols) > 0:
        print(f"\n   {name} - NaN gefunden:")
        for col, count in nan_cols.items():
            print(f"      {col}: {count} ({count/len(df)*100:.2f}%)")
    
    # Zeilen mit NaN entfernen
    df = df.dropna()
    
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"   {name} - Entfernt: {removed:,} Zeilen ({removed/initial_rows*100:.2f}%)")
    else:
        print(f"   {name} - ✅ Keine NaN/Inf gefunden")
    
    return df

train_df = clean_data(train_df, "Train")
val_df = clean_data(val_df, "Val")
test_df = clean_data(test_df, "Test")

# ==============================================================================
# 3. FEATURE SELECTION
# ==============================================================================
print("\n[3/5] Feature Selection...")

# Features und Target trennen
feature_cols = [col for col in train_df.columns 
                if col not in ['timestamp', 'target']]

X_train = train_df[feature_cols]
y_train = train_df['target']

X_val = val_df[feature_cols]
y_val = val_df['target']

X_test = test_df[feature_cols]
y_test = test_df['target']

print(f"   Initial Features: {len(feature_cols)}")

# A. Entferne Features mit zu niedriger Varianz
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0001)
selector.fit(X_train)

low_var_features = [feature_cols[i] for i, var in enumerate(selector.variances_) 
                    if var < 0.0001]

if len(low_var_features) > 0:
    print(f"\n   Entferne {len(low_var_features)} Features mit niedriger Varianz:")
    for feat in low_var_features[:5]:  # Zeige nur erste 5
        print(f"      - {feat}")
    if len(low_var_features) > 5:
        print(f"      ... und {len(low_var_features)-5} weitere")
    
    X_train = selector.transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    
    # Feature-Namen aktualisieren
    feature_cols = [col for col in feature_cols if col not in low_var_features]
    
    # Zurück zu DataFrame
    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_val = pd.DataFrame(X_val, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

print(f"   Features nach Variance Threshold: {len(feature_cols)}")

# B. Entferne hochkorrelierte Features (Multikollinearität)
print("\n   Prüfe Multikollinearität...")

corr_matrix = X_train.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Features mit Korrelation > 0.95
high_corr_features = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > 0.95)]

if len(high_corr_features) > 0:
    print(f"   Entferne {len(high_corr_features)} hochkorrelierte Features (>0.95):")
    for feat in high_corr_features[:5]:
        print(f"      - {feat}")
    if len(high_corr_features) > 5:
        print(f"      ... und {len(high_corr_features)-5} weitere")
    
    X_train = X_train.drop(columns=high_corr_features)
    X_val = X_val.drop(columns=high_corr_features)
    X_test = X_test.drop(columns=high_corr_features)
    
    feature_cols = [col for col in feature_cols if col not in high_corr_features]

print(f"   Final Features: {len(feature_cols)}")

# ==============================================================================
# 4. SCALING
# ==============================================================================
print("\n[4/5] Feature Scaling...")

# StandardScaler (z-score normalization)
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = pd.DataFrame(
    scaler.transform(X_train),
    columns=feature_cols,
    index=X_train.index
)

X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=feature_cols,
    index=X_val.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_cols,
    index=X_test.index
)

# Scaler speichern für später
scaler_path = os.path.join(prep_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"   ✅ Scaler gespeichert: {scaler_path}")

# ==============================================================================
# 5. FINAL CHECK & SAVE
# ==============================================================================
print("\n[5/5] Final Check & Save...")

# Class Balance prüfen
train_balance = y_train.value_counts(normalize=True)
print(f"\n   Train Class Balance:")
print(f"      0 (Short): {train_balance[0]:.2%}")
print(f"      1 (Long):  {train_balance[1]:.2%}")

if train_balance[0] < 0.3 or train_balance[1] < 0.3:
    print("   ⚠️  WARNUNG: Starkes Class Imbalance!")
else:
    print("   ✅ Class Balance OK")

# Timestamps wieder hinzufügen
X_train_scaled['timestamp'] = train_df['timestamp'].values
X_val_scaled['timestamp'] = val_df['timestamp'].values
X_test_scaled['timestamp'] = test_df['timestamp'].values

# Target hinzufügen
X_train_scaled['target'] = y_train.values
X_val_scaled['target'] = y_val.values
X_test_scaled['target'] = y_test.values

# Speichern
train_path = os.path.join(prep_dir, "train_prepared.parquet")
val_path = os.path.join(prep_dir, "val_prepared.parquet")
test_path = os.path.join(prep_dir, "test_prepared.parquet")

X_train_scaled.to_parquet(train_path, index=False)
X_val_scaled.to_parquet(val_path, index=False)
X_test_scaled.to_parquet(test_path, index=False)

print(f"\n   ✅ Train: {train_path}")
print(f"   ✅ Val:   {val_path}")
print(f"   ✅ Test:  {test_path}")

# Feature-Liste speichern
feature_list_path = os.path.join(prep_dir, "feature_list.txt")
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"   ✅ Feature List: {feature_list_path}")

# ==============================================================================
# ZUSAMMENFASSUNG
# ==============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Final Dataset Sizes:")
print(f"   Train: {len(X_train_scaled):,} rows × {len(feature_cols)} features")
print(f"   Val:   {len(X_val_scaled):,} rows × {len(feature_cols)} features")
print(f"   Test:  {len(X_test_scaled):,} rows × {len(feature_cols)} features")
print(f"\nClass Balance (Train):")
print(f"   Short (0): {train_balance[0]:.2%}")
print(f"   Long (1):  {train_balance[1]:.2%}")
print(f"\n✅ Preparation completed!")
print("=" * 70)
