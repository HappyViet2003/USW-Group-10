"""
06_modelling/xgboost_model.py

Experiment 2:
Trainiert das XGBoost Modell inkl. On-Chain Features.
WICHTIG: Die "Scorched Earth Policy" wurde erweitert, um rohe On-Chain-Werte zu verbieten.
"""

import os
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Pfade
base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
prep_dir = os.path.join(base_data_path, "processed", "prepared")
models_dir = os.path.join(base_data_path, "models")
os.makedirs(models_dir, exist_ok=True)

print("=" * 70)
print("XGBOOST TRAINING (EXP 2: ON-CHAIN)")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================
print("[1] Lade vorbereitete Daten...")
train_df = pd.read_parquet(os.path.join(prep_dir, "train_prepared.parquet"))
val_df = pd.read_parquet(os.path.join(prep_dir, "val_prepared.parquet"))
test_df = pd.read_parquet(os.path.join(prep_dir, "test_prepared.parquet"))

print(f"   Train Shape: {train_df.shape}")
print(f"   Test Shape:  {test_df.shape}")

# ==============================================================================
# 2. FEATURE SELECTION ("SCORCHED EARTH POLICY")
# ==============================================================================
exclude_cols = [
    # Metadaten
    'timestamp', 'target', 'sample_weight', 'future_close', 'year_month',

    # Absolute Krypto-Werte (VERBOTEN)
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'trade_count', 'local_high', 'local_low',
    'sma_50', 'ema_200',

    # Absolute Externe Werte - JETZT MIT KORREKTEN NAMEN
    # Aktien (QQQ, NVDA)
    'qqq_open', 'qqq_high', 'qqq_low', 'qqq_close', 'qqq_volume', 'qqq_trade_count', 'qqq_vwap',
    'nvda_open', 'nvda_high', 'nvda_low', 'nvda_close', 'nvda_volume', 'nvda_trade_count', 'nvda_vwap',

    # Rohstoffe & Währungen (Gold, USD) -> Hier war der Fehler (gold statt gld)
    'gold_open', 'gold_high', 'gold_low', 'gold_close', 'gold_volume', 'gold_trade_count', 'gold_vwap',
    'usd_open', 'usd_high', 'usd_low', 'usd_close', 'usd_volume', 'usd_trade_count', 'usd_vwap',

    # Macro & Futures
    'nq_open', 'nq_high', 'nq_low', 'nq_close', 'nq_volume',  # Nasdaq Futures
    'm2_close', 'm2_open', 'm2_high', 'm2_low',  # Money Supply (M2)
    'rates_close', 'rates_open',  # Zinsen

    # On-Chain Absolute Werte
    'onchain_hash-rate',
    'onchain_n-unique-addresses'
]

# Filtern
feature_cols = [c for c in train_df.columns if c not in exclude_cols]
print(f"\n   Training mit {len(feature_cols)} Features.")
print(f"   Beispiel Features: {feature_cols[:5]} ...")

# Prüfen, ob On-Chain Features dabei sind (die RELATIVEN)
onchain_feats = [c for c in feature_cols if 'change' in c or 'onchain' in c]
print(f"   ✅ Aktive On-Chain Features: {onchain_feats}")

X_train = train_df[feature_cols]
y_train = train_df['target']
X_val = val_df[feature_cols]
y_val = val_df['target']
X_test = test_df[feature_cols]
y_test = test_df['target']

# Sample Weights nutzen (falls vorhanden)
w_train = train_df['sample_weight'] if 'sample_weight' in train_df.columns else None

# ==============================================================================
# 3. TRAINING
# ==============================================================================
print("\n[2] Trainiere XGBoost...")

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Hyperparameter (Konservativ gegen Overfitting)
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,           # Geringe Tiefe erzwingt Generalisierung
    'eta': 0.02,              # Langsames Lernen
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Training mit Early Stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=100
)

# ==============================================================================
# 4. EVALUATION
# ==============================================================================
print("\n[3] Evaluation...")

# Vorhersagen
preds_prob = model.predict(dtest)
preds_class = (preds_prob > 0.5).astype(int)

# Metriken
acc = accuracy_score(y_test, preds_class)
auc = roc_auc_score(y_test, preds_prob)

print("-" * 50)
print(f"ERGEBNIS EXPERIMENT 2 (On-Chain):")
print(f"   ✅ Test Accuracy: {acc:.2%}")
print(f"   ✅ Test AUC:      {auc:.4f}")
print("-" * 50)

# Feature Importance (Was war wichtig?)
importance = model.get_score(importance_type='gain')
imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain'])
imp_df = imp_df.sort_values('Gain', ascending=False)

print("\n🏆 Top 10 Wichtigste Features:")
print(imp_df.head(10))

# Prüfen wo On-Chain gelandet ist
print("\n🔍 Ranking der On-Chain Features:")
print(imp_df[imp_df['Feature'].str.contains('change') | imp_df['Feature'].str.contains('onchain')])

# ==============================================================================
# 5. SPEICHERN
# ==============================================================================
# Modell speichern (eigener Name für Experiment 2)
model_path = os.path.join(models_dir, "xgboost_onchain.json")
model.save_model(model_path)
print(f"\n💾 Modell gespeichert: {model_path}")

# Predictions speichern für Backtest
test_df['prob'] = preds_prob
test_df['pred'] = preds_class
# Wir speichern nur die nötigen Spalten
pred_path = os.path.join(models_dir, "test_predictions.parquet")
test_df[['timestamp', 'target', 'prob', 'pred']].to_parquet(pred_path)
print(f"💾 Vorhersagen gespeichert: {pred_path}")