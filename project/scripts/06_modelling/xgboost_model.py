"""
06_modeling/02_xgboost_model.py

Trainiert ein XGBoost-Modell für die BTC-Prognose.
XGBoost ist besonders gut für strukturierte/tabellarische Daten.
"""

import os
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
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
print("XGBOOST MODEL")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================
print("\n[1/5] Lade vorbereitete Daten...")

train_df = pd.read_parquet(os.path.join(prep_dir, "train_prepared.parquet"))
val_df = pd.read_parquet(os.path.join(prep_dir, "val_prepared.parquet"))
test_df = pd.read_parquet(os.path.join(prep_dir, "test_prepared.parquet"))

# Features und Target trennen
feature_cols = [col for col in train_df.columns 
                if col not in ['timestamp', 'target']]

X_train = train_df[feature_cols]
y_train = train_df['target']

X_val = val_df[feature_cols]
y_val = val_df['target']

X_test = test_df[feature_cols]
y_test = test_df['target']

print(f"   Train: {len(X_train):,} samples × {len(feature_cols)} features")
print(f"   Val:   {len(X_val):,} samples")
print(f"   Test:  {len(X_test):,} samples")

# ==============================================================================
# 2. XGBOOST DATENFORMAT
# ==============================================================================
print("\n[2/5] Erstelle DMatrix...")

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

print("   ✅ DMatrix erstellt")

# ==============================================================================
# 3. HYPERPARAMETER & TRAINING
# ==============================================================================
print("\n[3/5] Trainiere XGBoost...")

# Class Balance berechnen
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

params = {
    # Objective
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc', 'error'],
    
    # Tree Parameters
    'max_depth': 6,
    'eta': 0.1,  # Learning rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    
    # Regularization
    'lambda': 1.0,  # L2
    'alpha': 0.1,   # L1
    
    # Class Imbalance
    'scale_pos_weight': scale_pos_weight,
    
    # Other
    'seed': 42,
    'tree_method': 'hist',  # Schneller
    'device': 'cpu'  # Oder 'cuda' wenn GPU verfügbar
}

print(f"   Hyperparameters:")
for key, value in params.items():
    print(f"      {key}: {value}")

# Training mit Early Stopping
evals = [(dtrain, 'train'), (dval, 'val')]
evals_result = {}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    evals_result=evals_result,
    verbose_eval=100
)

print(f"\n   ✅ Training abgeschlossen")
print(f"   Best iteration: {model.best_iteration}")
print(f"   Best score: {model.best_score:.4f}")

# ==============================================================================
# 4. EVALUATION
# ==============================================================================
print("\n[4/5] Evaluation...")

def evaluate_xgb_model(model, dmatrix, X, y, dataset_name=""):
    """Evaluiert das XGBoost-Modell"""
    
    y_pred_proba = model.predict(dmatrix)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    cm = confusion_matrix(y, y_pred)
    
    print(f"\n   {dataset_name} Metrics:")
    print(f"      Accuracy:  {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall:    {metrics['recall']:.4f}")
    print(f"      F1-Score:  {metrics['f1']:.4f}")
    print(f"      ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\n   Confusion Matrix:")
    print(f"      TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
    print(f"      FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
    
    return metrics, cm, y_pred_proba

# Train
train_metrics, train_cm, _ = evaluate_xgb_model(
    model, dtrain, X_train, y_train, "TRAIN"
)

# Validation
val_metrics, val_cm, _ = evaluate_xgb_model(
    model, dval, X_val, y_val, "VALIDATION"
)

# Test
test_metrics, test_cm, test_pred_proba = evaluate_xgb_model(
    model, dtest, X_test, y_test, "TEST"
)

# ==============================================================================
# 5. FEATURE IMPORTANCE & SPEICHERN
# ==============================================================================
print("\n[5/5] Feature Importance & Speichern...")

# Feature Importance
importance_dict = model.get_score(importance_type='gain')
feature_importance = pd.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in importance_dict.items()
]).sort_values('importance', ascending=False)

print(f"\n   Top 10 wichtigste Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature']}: {row['importance']:.2f}")

# Modell speichern
model_path = os.path.join(models_dir, "xgboost_model.json")
model.save_model(model_path)
print(f"\n   ✅ Modell: {model_path}")

# Metriken speichern
metrics_dict = {
    'model_name': 'XGBoost',
    'best_iteration': int(model.best_iteration),
    'hyperparameters': params,
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics
}

metrics_path = os.path.join(models_dir, "xgboost_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print(f"   ✅ Metriken: {metrics_path}")

# Feature Importance speichern
importance_path = os.path.join(models_dir, "xgboost_feature_importance.csv")
feature_importance.to_csv(importance_path, index=False)
print(f"   ✅ Feature Importance: {importance_path}")

# Predictions speichern (für Backtesting)
test_predictions = test_df[['timestamp']].copy()
test_predictions['target'] = y_test.values
test_predictions['prediction'] = (test_pred_proba > 0.5).astype(int)
test_predictions['probability'] = test_pred_proba

pred_path = os.path.join(models_dir, "xgboost_test_predictions.parquet")
test_predictions.to_parquet(pred_path, index=False)
print(f"   ✅ Test Predictions: {pred_path}")

# ==============================================================================
# ZUSAMMENFASSUNG
# ==============================================================================
print("\n" + "=" * 70)
print("XGBOOST MODEL SUMMARY")
print("=" * 70)
print(f"Model: XGBoost Classifier")
print(f"Features: {len(feature_cols)}")
print(f"Best Iteration: {model.best_iteration}")
print(f"\nPerformance:")
print(f"   Train Accuracy:      {train_metrics['accuracy']:.4f}")
print(f"   Validation Accuracy: {val_metrics['accuracy']:.4f}")
print(f"   Test Accuracy:       {test_metrics['accuracy']:.4f}")
print(f"\n   Validation F1-Score: {val_metrics['f1']:.4f}")
print(f"   Test F1-Score:       {test_metrics['f1']:.4f}")
print(f"\n   Test ROC-AUC:        {test_metrics['roc_auc']:.4f}")

# Overfitting Check
if train_metrics['accuracy'] - val_metrics['accuracy'] > 0.05:
    print(f"\n   ⚠️  Mögliches Overfitting (Train-Val Gap: {train_metrics['accuracy'] - val_metrics['accuracy']:.4f})")
else:
    print(f"\n   ✅ Kein starkes Overfitting erkennbar")

# Vergleich mit Baseline
baseline_metrics_path = os.path.join(models_dir, "baseline_metrics.json")
if os.path.exists(baseline_metrics_path):
    with open(baseline_metrics_path, 'r') as f:
        baseline = json.load(f)
    
    improvement = test_metrics['accuracy'] - baseline['test']['accuracy']
    print(f"\n   Vergleich mit Baseline:")
    print(f"      Baseline Test Accuracy: {baseline['test']['accuracy']:.4f}")
    print(f"      XGBoost Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"      Improvement: {improvement:+.4f} ({improvement/baseline['test']['accuracy']*100:+.2f}%)")

print("=" * 70)
