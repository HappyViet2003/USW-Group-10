"""
06_modeling/model_comparison.py

Vergleicht die Performance von:
1. Baseline (Logistische Regression)
2. XGBoost (Experiment 2: On-Chain)

Erstellt einen Plot mit Accuracy, Precision, Recall und F1-Score.
"""

import os
import yaml
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
models_dir = os.path.join(base_data_path, "models")
images_dir = os.path.join(script_dir, "../../images")
os.makedirs(images_dir, exist_ok=True)

print("=" * 70)
print("MODEL COMPARISON: Baseline vs. XGBoost (On-Chain)")
print("=" * 70)

# ==============================================================================
# 1. BASELINE METRIKEN LADEN
# ==============================================================================
baseline_path = os.path.join(models_dir, "baseline_metrics.json")
if not os.path.exists(baseline_path):
    print(f"❌ Baseline Metrics fehlen: {baseline_path}")
    print("   Bitte erst 'baseline_model.py' ausführen!")
    exit(1)

with open(baseline_path, "r") as f:
    baseline_data = json.load(f)
    # Wir nehmen die TEST-Metriken
    bl_metrics = baseline_data['test']

print(f"   Baseline Acc: {bl_metrics['accuracy']:.2%}")

# ==============================================================================
# 2. XGBOOST METRIKEN BERECHNEN
# ==============================================================================
# XGBoost speichert keine JSON, sondern die rohen Vorhersagen. Wir berechnen die Metriken frisch.
xgb_pred_path = os.path.join(models_dir, "test_predictions.parquet")
if not os.path.exists(xgb_pred_path):
    print(f"❌ XGBoost Predictions fehlen: {xgb_pred_path}")
    print("   Bitte erst 'xgboost_model.py' ausführen!")
    exit(1)

df_xgb = pd.read_parquet(xgb_pred_path)
y_true = df_xgb['target']
y_pred = df_xgb['pred']
y_prob = df_xgb['prob']

xgb_metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred, zero_division=0),
    'recall': recall_score(y_true, y_pred, zero_division=0),
    'f1': f1_score(y_true, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_true, y_prob)
}

print(f"   XGBoost Acc:  {xgb_metrics['accuracy']:.2%}")

# ==============================================================================
# 3. VERGLEICHS-PLOT ERSTELLEN
# ==============================================================================
metrics_names = ['accuracy', 'precision', 'recall', 'f1']
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

bl_values = [bl_metrics[m] for m in metrics_names]
xgb_values = [xgb_metrics[m] for m in metrics_names]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, bl_values, width, label='Baseline (LogReg)', color='gray', alpha=0.7)
rects2 = ax.bar(x + width/2, xgb_values, width, label='XGBoost (On-Chain)', color='#1f77b4')

ax.set_ylabel('Score')
ax.set_title('Model Comparison: Experiment 2')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.45, 0.60) # Fokus auf den relevanten Bereich (50%)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)

# Werte über den Balken anzeigen
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

# Speichern
output_path = os.path.join(images_dir, "06_model_comparison_exp2.png")
plt.savefig(output_path, dpi=150)
print(f"\n✅ Plot gespeichert: {output_path}")

# Fazit ausgeben
diff = xgb_metrics['accuracy'] - bl_metrics['accuracy']
if diff > 0:
    print(f"🚀 FAZIT: XGBoost ist {diff:.2%} besser als die Baseline.")
else:
    print(f"⚠️ FAZIT: Baseline ist {abs(diff):.2%} besser. Prüfe Overfitting!")