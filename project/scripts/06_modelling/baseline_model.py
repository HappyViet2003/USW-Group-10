"""
06_modeling/01_baseline_model.py

Trainiert ein einfaches Baseline-Modell (Logistische Regression)
um eine Referenz für komplexere Modelle zu haben.
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
print("BASELINE MODEL: Logistic Regression")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN
# ==============================================================================
print("\n[1/4] Lade vorbereitete Daten...")

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
# 2. MODELL TRAINIEREN
# ==============================================================================
print("\n[2/4] Trainiere Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Hilft bei Class Imbalance
)

model.fit(X_train, y_train)

print("   ✅ Training abgeschlossen")

# ==============================================================================
# 3. EVALUATION
# ==============================================================================
print("\n[3/4] Evaluation...")

def evaluate_model(model, X, y, dataset_name=""):
    """Evaluiert das Modell und gibt Metriken zurück"""
    
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
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
    
    return metrics, cm

# Train
train_metrics, train_cm = evaluate_model(model, X_train, y_train, "TRAIN")

# Validation
val_metrics, val_cm = evaluate_model(model, X_val, y_val, "VALIDATION")

# Test
test_metrics, test_cm = evaluate_model(model, X_test, y_test, "TEST")

# ==============================================================================
# 4. MODELL SPEICHERN
# ==============================================================================
print("\n[4/4] Speichere Modell und Metriken...")

# Modell speichern
model_path = os.path.join(models_dir, "baseline_logistic_regression.pkl")
joblib.dump(model, model_path)
print(f"   ✅ Modell: {model_path}")

# Metriken speichern
metrics_dict = {
    'model_name': 'Logistic Regression (Baseline)',
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics
}

metrics_path = os.path.join(models_dir, "baseline_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print(f"   ✅ Metriken: {metrics_path}")

# Feature Importance (Koeffizienten)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

importance_path = os.path.join(models_dir, "baseline_feature_importance.csv")
feature_importance.to_csv(importance_path, index=False)
print(f"   ✅ Feature Importance: {importance_path}")

print(f"\n   Top 10 wichtigste Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature']}: {row['coefficient']:.4f}")

# ==============================================================================
# ZUSAMMENFASSUNG
# ==============================================================================
print("\n" + "=" * 70)
print("BASELINE MODEL SUMMARY")
print("=" * 70)
print(f"Model: Logistic Regression")
print(f"Features: {len(feature_cols)}")
print(f"\nPerformance:")
print(f"   Train Accuracy:      {train_metrics['accuracy']:.4f}")
print(f"   Validation Accuracy: {val_metrics['accuracy']:.4f}")
print(f"   Test Accuracy:       {test_metrics['accuracy']:.4f}")
print(f"\n   Validation F1-Score: {val_metrics['f1']:.4f}")
print(f"   Test F1-Score:       {test_metrics['f1']:.4f}")

# Overfitting Check
if train_metrics['accuracy'] - val_metrics['accuracy'] > 0.05:
    print(f"\n   ⚠️  Mögliches Overfitting (Train-Val Gap: {train_metrics['accuracy'] - val_metrics['accuracy']:.4f})")
else:
    print(f"\n   ✅ Kein starkes Overfitting erkennbar")

print("=" * 70)
