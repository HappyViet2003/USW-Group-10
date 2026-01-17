"""
06_modelling/ensemble_model.py

Experiment 4: Ensemble Model (XGBoost + RandomForest + LogisticRegression)
Kombiniert 3 Modelle via Soft Voting für robustere Predictions.
"""

import os
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# --- KONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Pfade
base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
prep_dir = os.path.join(base_data_path, "processed/prepared")
models_dir = os.path.join(base_data_path, "models")
os.makedirs(models_dir, exist_ok=True)

print("=" * 70)
print("ENSEMBLE MODEL TRAINING (EXP 4)")
print("=" * 70)

# 1. Daten laden
print("\n[1/5] Lade vorbereitete Daten...")
train_df = pd.read_parquet(os.path.join(prep_dir, "train_prepared.parquet"))
val_df = pd.read_parquet(os.path.join(prep_dir, "val_prepared.parquet"))
test_df = pd.read_parquet(os.path.join(prep_dir, "test_prepared.parquet"))

# Features und Target trennen
exclude_cols = ['timestamp', 'target', 'sample_weight', 'regime']  # Regime nicht als Feature
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X_train = train_df[feature_cols]
y_train = train_df['target']
w_train = train_df['sample_weight'] if 'sample_weight' in train_df.columns else None

X_val = val_df[feature_cols]
y_val = val_df['target']

X_test = test_df[feature_cols]
y_test = test_df['target']

print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"   Features: {len(feature_cols)}")

# 2. Modelle definieren
print("\n[2/5] Definiere Ensemble-Modelle...")

# Model 1: XGBoost (optimiert aus Exp 1)
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    max_depth=3,
    eta=0.01,
    subsample=0.7,
    colsample_bytree=0.6,
    min_child_weight=10,
    gamma=0.2,
    reg_lambda=5.0,
    reg_alpha=1.0,
    n_estimators=500,  # Weniger Bäume für Ensemble
    random_state=42,
    n_jobs=-1
)

# Model 2: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Model 3: Logistic Regression (mit L2 Regularisierung)
lr_model = LogisticRegression(
    C=0.1,  # Starke Regularisierung
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

# Ensemble via Soft Voting (averaged probabilities)
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lr', lr_model)
    ],
    voting='soft',  # Average probabilities
    weights=[2, 1, 1],  # XGBoost gets more weight (performed best in Exp 1)
    n_jobs=-1
)

print("   ✅ XGBoost (weight=2)")
print("   ✅ RandomForest (weight=1)")
print("   ✅ LogisticRegression (weight=1)")

# 3. Training
print("\n[3/5] Trainiere Ensemble...")
ensemble.fit(X_train, y_train, sample_weight=w_train)

print("   ✅ Training abgeschlossen")

# 4. Evaluation
print("\n[4/5] Evaluiere Modelle...")

# Predictions
train_pred = ensemble.predict(X_train)
train_proba = ensemble.predict_proba(X_train)[:, 1]

val_pred = ensemble.predict(X_val)
val_proba = ensemble.predict_proba(X_val)[:, 1]

test_pred = ensemble.predict(X_test)
test_proba = ensemble.predict_proba(X_test)[:, 1]

# Metriken
train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

train_auc = roc_auc_score(y_train, train_proba)
val_auc = roc_auc_score(y_val, val_proba)
test_auc = roc_auc_score(y_test, test_proba)

print("\n" + "=" * 70)
print("ENSEMBLE RESULTS")
print("=" * 70)
print(f"Train Accuracy: {train_acc:.4f} | AUC: {train_auc:.4f}")
print(f"Val   Accuracy: {val_acc:.4f} | AUC: {val_auc:.4f}")
print(f"Test  Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}")
print("=" * 70)

# Classification Report
print("\nTest Set Classification Report:")
print(classification_report(y_test, test_pred, target_names=['Down', 'Up']))

# 5. Speichern
print("\n[5/5] Speichere Modell und Predictions...")

# Modell speichern
model_path = os.path.join(models_dir, "ensemble_model.pkl")
joblib.dump(ensemble, model_path)
print(f"   ✅ Modell: {model_path}")

# Predictions speichern (für Backtest)
test_df['prob'] = test_proba
test_df['pred'] = test_pred
pred_path = os.path.join(models_dir, "test_predictions.parquet")
test_df.to_parquet(pred_path)
print(f"   ✅ Predictions: {pred_path}")

# Feature Importance (nur von XGBoost extrahieren)
xgb_model_trained = ensemble.named_estimators_['xgb']
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model_trained.feature_importances_
}).sort_values('Importance', ascending=False)

importance_path = os.path.join(models_dir, "feature_importance.csv")
importance.to_csv(importance_path, index=False)
print(f"   ✅ Feature Importance: {importance_path}")

print("\n" + "=" * 70)
print("✅ ENSEMBLE TRAINING ABGESCHLOSSEN")
print("=" * 70)
