"""
06_modeling/baseline_model.py

Trainiert ein einfaches Baseline-Modell (Logistische Regression)
um eine Referenz für das XGBoost-Modell zu haben.
Update: Nutzt dieselbe strenge Feature-Filterung wie XGBoost.
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import json

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
prep_dir = os.path.join(base_data_path, "processed", "prepared")

# Output
models_dir = os.path.join(base_data_path, "models")
os.makedirs(models_dir, exist_ok=True)

print("=" * 70)
print("BASELINE MODEL: Logistic Regression (Fair Comparison)")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================
print("\n[1/4] Lade vorbereitete Daten...")

train_df = pd.read_parquet(os.path.join(prep_dir, "train_prepared.parquet"))
val_df = pd.read_parquet(os.path.join(prep_dir, "val_prepared.parquet"))
test_df = pd.read_parquet(os.path.join(prep_dir, "test_prepared.parquet"))

# --- FEATURE SELECTION (Kopie vom XGBoost Skript) ---
exclude_cols = [
    # Metadaten
    'timestamp', 'target', 'sample_weight', 'future_close', 'year_month',

    # Absolute Preise (Bitcoin)
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'local_high', 'local_low',
    'sma_50', 'ema_200', 'obv',

    # Absolute Preise (Externe)
    'qqq_open', 'qqq_high', 'qqq_low', 'qqq_close', 'qqq_volume', 'qqq_vwap', 'qqq_trade_count',
    'nvda_open', 'nvda_high', 'nvda_low', 'nvda_close', 'nvda_volume', 'nvda_vwap', 'nvda_trade_count',
    'nq_open', 'nq_high', 'nq_low', 'nq_close', 'nq_volume', 'nq_vwap', 'nq_trade_count',
    'gld_open', 'gld_high', 'gld_low', 'gld_close', 'gld_volume', 'gld_vwap', 'gld_trade_count',
    'gold_open', 'gold_high', 'gold_low', 'gold_close', 'gold_volume', 'gold_vwap', 'gold_trade_count',
    'uup_open', 'uup_high', 'uup_low', 'uup_close', 'uup_volume', 'uup_vwap',
    'usd_open', 'usd_high', 'usd_low', 'usd_close', 'usd_volume', 'usd_vwap', 'usd_trade_count',

    'rates_open', 'rates_high', 'rates_low', 'rates_close',
    'm2_close', 'm2_value',
    'atr_14', 'adx', 'trade_count'
]

# Filtere Features
existing_exclude = [c for c in exclude_cols if c in train_df.columns]
feature_cols = [col for col in train_df.columns if col not in existing_exclude]

print(f"   Ignoriere {len(existing_exclude)} Raw-Features.")
print(f"   Nutze {len(feature_cols)} smarte Features.")

# Datensätze erstellen
X_train = train_df[feature_cols]
y_train = train_df['target']
w_train = train_df['sample_weight'] if 'sample_weight' in train_df.columns else None

X_val = val_df[feature_cols]
y_val = val_df['target']

X_test = test_df[feature_cols]
y_test = test_df['target']

# ==============================================================================
# 2. TRAINING
# ==============================================================================
print("\n[2/4] Trainiere Logistische Regression...")

# Wir nutzen sample_weight auch hier, um fair zu bleiben!
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced' # Hilft gegen Klassen-Ungleichgewicht
)

model.fit(X_train, y_train, sample_weight=w_train)

print("   ✅ Training abgeschlossen.")

# ==============================================================================
# 3. EVALUATION
# ==============================================================================
print("\n[3/4] Evaluation...")

def evaluate(model, X, y, name):
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, probs)

    print(f"   {name} >> Acc: {acc:.2%} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return {'accuracy': acc, 'f1': f1, 'roc_auc': auc}

train_metrics = evaluate(model, X_train, y_train, "TRAIN")
val_metrics = evaluate(model, X_val, y_val, "VALIDATION")
test_metrics = evaluate(model, X_test, y_test, "TEST")

# ==============================================================================
# 4. SPEICHERN
# ==============================================================================
print("\n[4/4] Speichern...")

metrics_dict = {
    'model_name': 'Logistic Regression (Baseline)',
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics
}

metrics_path = os.path.join(models_dir, "baseline_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics_dict, f, indent=2)

# Feature Importance (Koeffizienten)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

importance_path = os.path.join(models_dir, "baseline_feature_importance.csv")
feature_importance.to_csv(importance_path, index=False)

print(f"   ✅ Metriken: {metrics_path}")
print(f"   ✅ Feature Importance: {importance_path}")

print("-" * 70)
print(f"BASELINE TEST ACCURACY: {test_metrics['accuracy']:.2%}")
print("-" * 70)