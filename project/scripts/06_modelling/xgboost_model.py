"""
06_modelling/xgboost_model.py

Trainiert ein XGBoost-Modell für die BTC-Prognose.
Update: Entfernt radikal alle absoluten Preis-Features, um Overfitting zu verhindern.
"""

import os
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
prep_dir = os.path.join(base_data_path, "processed", "prepared")

# Output Ordner
models_dir = os.path.join(base_data_path, "models")
os.makedirs(models_dir, exist_ok=True)

print("=" * 70)
print("XGBOOST MODEL TRAINING (No-Overfit Edition)")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================
print("\n[1/5] Lade vorbereitete Daten...")

# Wir laden die Daten (Timestamp muss enthalten sein durch den Fix in step 05)
train_df = pd.read_parquet(os.path.join(prep_dir, "train_prepared.parquet"))
val_df = pd.read_parquet(os.path.join(prep_dir, "val_prepared.parquet"))
test_df = pd.read_parquet(os.path.join(prep_dir, "test_prepared.parquet"))

# --- FEATURE SELECTION: Absolute Preise verbannen ---
# Wir entfernen alles, was "Dollar-Beträge" sind, da diese sich über Jahre ändern (Instationarität).
exclude_cols = [
    # 1. Metadaten
    'timestamp', 'target', 'sample_weight', 'future_close', 'year_month',

    # 2. Bitcoin Absolute Preise
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'local_high', 'local_low',
    'sma_50', 'ema_200',
    'obv',  # Nur obv_slope erlaubt

    # 3. Externe Absolute Preise (Alles was Open/High/Low/Close/Vol/VWAP heißt)
    'qqq_open', 'qqq_high', 'qqq_low', 'qqq_close', 'qqq_volume', 'qqq_vwap',
    'nvda_open', 'nvda_high', 'nvda_low', 'nvda_close', 'nvda_volume', 'nvda_vwap',
    'nq_open', 'nq_high', 'nq_low', 'nq_close', 'nq_volume',
    'gld_open', 'gld_high', 'gld_low', 'gld_close', 'gld_volume', 'gld_vwap',
    'uup_open', 'uup_high', 'uup_low', 'uup_close', 'uup_volume', 'uup_vwap',
    'usd_open', 'usd_high', 'usd_low', 'usd_close', 'usd_volume', 'usd_vwap',  # <-- NEU: usd_vwap raus

    'rates_open', 'rates_high', 'rates_low', 'rates_close',
    'm2_close', 'm2_value',

    # 4. Versteckte Absolute Werte
    'trade_count', 'qqq_trade_count', 'nvda_trade_count', 'gold_trade_count',  # <-- NEU: Counts raus

    # Bollinger Bands: Wir wollen BBP (Prozent), aber NICHT BBM (Mittelwert = Preis) oder BBU/BBL
    'BBM_20_2.0', 'BBU_20_2.0', 'BBL_20_2.0',  # Pandas TA Namen
    'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBL_20_2.0_2.0',  # Manchmal heißen die so

    # ATR in Dollar (wir wollen atr_pct oder atr_norm)
    'atr_14',
    'adx'  # ADX absolut ist okay, aber adx_norm ist sicherer. (Kannst du drin lassen oder rausnehmen)
]

# Wir filtern nur Spalten, die tatsächlich existieren
existing_exclude = [c for c in exclude_cols if c in train_df.columns]
feature_cols = [col for col in train_df.columns if col not in existing_exclude]

print(f"   Ignoriere {len(existing_exclude)} Raw-Features (z.B. {existing_exclude[:3]})...")
print(f"   Nutze {len(feature_cols)} smarte Features (z.B. {feature_cols[:5]})...")

# Training Data
X_train = train_df[feature_cols]
y_train = train_df['target']
w_train = train_df['sample_weight'] if 'sample_weight' in train_df.columns else None

# Validation Data
X_val = val_df[feature_cols]
y_val = val_df['target']

# Test Data
X_test = test_df[feature_cols]
y_test = test_df['target']

if w_train is not None:
    print(f"   ✅ Sample Weights aktiv (Min: {w_train.min():.2f}, Max: {w_train.max():.2f})")

# ==============================================================================
# 2. XGBOOST SETUP
# ==============================================================================
print("\n[2/5] Erstelle DMatrix...")

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, weight=w_train)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

# ==============================================================================
# 3. HYPERPARAMETER (Balance Tuning)
# ==============================================================================
print("\n[3/5] Trainiere XGBoost...")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc', 'error'],

    # --- OPTIMIERT: Weniger Overfitting, bessere Generalisierung ---
    'max_depth': 3,          # FLACHER (weniger Overfitting)
    'eta': 0.01,             # LANGSAMER (besseres Lernen)

    'subsample': 0.7,        # 70% der Daten pro Baum
    'colsample_bytree': 0.6, # 60% der Features pro Baum (mehr Diversität)

    'min_child_weight': 10,  # MEHR Samples pro Blatt (robuster)
    'gamma': 0.2,            # HÖHER (weniger unnötige Splits)

    'lambda': 5.0,           # STÄRKERE L2 Regularisierung
    'alpha': 1.0,            # STÄRKERE L1 Regularisierung

    'scale_pos_weight': scale_pos_weight,
    'seed': 42,
    'n_jobs': -1
}


# Training
evals = [(dtrain, 'train'), (dval, 'val')]
evals_result = {}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=3000,       # Genug Raum für eta 0.03
    evals=evals,
    early_stopping_rounds=150,  # Stoppt, wenn 150 Runden keine Verbesserung
    evals_result=evals_result,
    verbose_eval=100
)

print(f"\n   ✅ Training abgeschlossen")
print(f"   Best iteration: {model.best_iteration}")
print(f"   Best AUC Score (Val): {model.best_score:.4f}")

# ==============================================================================
# 4. EVALUATION
# ==============================================================================
print("\n[4/5] Evaluation...")

def evaluate(model, dmatrix, y_true, name):
    probs = model.predict(dmatrix)
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probs)

    print(f"   {name} >> Acc: {acc:.2%} | Prec: {prec:.2%} | Rec: {rec:.2%} | AUC: {auc:.4f}")
    return acc, probs

_, _ = evaluate(model, dtrain, y_train, "TRAIN")
_, _ = evaluate(model, dval, y_val, "VALIDATION")
test_acc, test_probs = evaluate(model, dtest, y_test, "TEST")

# ==============================================================================
# 5. FEATURE IMPORTANCE & SPEICHERN
# ==============================================================================
print("\n[5/5] Feature Importance...")

importance = model.get_score(importance_type='gain')
imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain'])
imp_df = imp_df.sort_values(by='Gain', ascending=False)

print("\n🏆 TOP 10 WICHTIGSTE FEATURES (Sollte keine Raw-Preise enthalten):")
print(imp_df.head(10).to_string(index=False))

# Speichern
model.save_model(os.path.join(models_dir, "xgboost_final.json"))
imp_df.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)

# Predictions speichern (Nur wenn Timestamp vorhanden)
if 'timestamp' in test_df.columns:
    preds_df = test_df[['timestamp']].copy()
    preds_df['target'] = y_test.values
    preds_df['prob'] = test_probs
    preds_df['pred'] = (test_probs > 0.5).astype(int)
    preds_df.to_parquet(os.path.join(models_dir, "test_predictions.parquet"))
    print(f"   ✅ Predictions gespeichert mit Timestamps.")
else:
    print("   ⚠️ Warnung: 'timestamp' fehlt im Test-Set. Predictions ohne Zeit gespeichert.")

print(f"\n🏁 Fertig! Test Accuracy: {test_acc:.2%}")