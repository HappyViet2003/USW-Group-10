"""
06_modeling/baseline_model.py

Trainiert ein einfaches Baseline-Modell (Logistische Regression).
FIX: Berechnet jetzt auch Precision und Recall für den Vergleich.
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
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
print("BASELINE MODEL: Logistic Regression (Full Metrics)")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================
print("\n[1/4] Lade vorbereitete Daten...")

train_df = pd.read_parquet(os.path.join(prep_dir, "train_prepared.parquet"))
val_df = pd.read_parquet(os.path.join(prep_dir, "val_prepared.parquet"))
test_df = pd.read_parquet(os.path.join(prep_dir, "test_prepared.parquet"))

# --- FEATURE SELECTION (Muss identisch zu XGBoost sein) ---
exclude_cols = [
    'timestamp', 'target', 'sample_weight', 'future_close', 'year_month',
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'local_high', 'local_low', 'sma_50', 'ema_200', 'obv',
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

existing_exclude = [c for c in exclude_cols if c in train_df.columns]
feature_cols = [col for col in train_df.columns if col not in existing_exclude]

print(f"   Nutze {len(feature_cols)} smarte Features.")

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

model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train, sample_weight=w_train)

print("   ✅ Training abgeschlossen.")

# ==============================================================================
# 3. EVALUATION
# ==============================================================================
print("\n[3/4] Evaluation...")

def evaluate(model, X, y, name):
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    # Metriken berechnen
    metrics = {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds, zero_division=0), # <--- NEU
        'recall': recall_score(y, preds, zero_division=0),       # <--- NEU
        'f1': f1_score(y, preds, zero_division=0),
        'roc_auc': roc_auc_score(y, probs)
    }

    print(f"   {name} >> Acc: {metrics['accuracy']:.2%} | F1: {metrics['f1']:.4f}")
    return metrics

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

print(f"   ✅ Metriken gespeichert: {metrics_path}")