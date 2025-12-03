"""
06_modeling/03_model_comparison.py

Vergleicht alle trainierten Modelle und erstellt einen Report.
"""

import os
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
models_dir = os.path.join(base_data_path, "models")

# Output
output_dir = os.path.join(script_dir, "../../images")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# ==============================================================================
# 1. LADE ALLE METRIKEN
# ==============================================================================
print("\n[1/3] Lade Modell-Metriken...")

models = []

# Baseline
baseline_path = os.path.join(models_dir, "baseline_metrics.json")
if os.path.exists(baseline_path):
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    models.append(baseline)
    print(f"   ✅ Baseline geladen")

# XGBoost
xgb_path = os.path.join(models_dir, "xgboost_metrics.json")
if os.path.exists(xgb_path):
    with open(xgb_path, 'r') as f:
        xgboost = json.load(f)
    models.append(xgboost)
    print(f"   ✅ XGBoost geladen")

if len(models) == 0:
    print("   ❌ Keine Modelle gefunden!")
    exit(1)

# ==============================================================================
# 2. VERGLEICHSTABELLE ERSTELLEN
# ==============================================================================
print("\n[2/3] Erstelle Vergleichstabelle...")

comparison_data = []

for model in models:
    for dataset in ['train', 'validation', 'test']:
        comparison_data.append({
            'Model': model['model_name'],
            'Dataset': dataset.capitalize(),
            'Accuracy': model[dataset]['accuracy'],
            'Precision': model[dataset]['precision'],
            'Recall': model[dataset]['recall'],
            'F1-Score': model[dataset]['f1'],
            'ROC-AUC': model[dataset]['roc_auc']
        })

df_comparison = pd.DataFrame(comparison_data)

# Speichern
comparison_path = os.path.join(models_dir, "model_comparison.csv")
df_comparison.to_csv(comparison_path, index=False)
print(f"   ✅ Vergleichstabelle: {comparison_path}")

# Anzeigen
print("\n   Model Comparison (Test Set):")
test_comparison = df_comparison[df_comparison['Dataset'] == 'Test']
print(test_comparison.to_string(index=False))

# ==============================================================================
# 3. VISUALISIERUNG
# ==============================================================================
print("\n[3/3] Erstelle Visualisierungen...")

# A. Metriken-Vergleich (Test Set)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

test_df = df_comparison[df_comparison['Dataset'] == 'Test']

# Plot 1: Alle Metriken
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
test_metrics = test_df[['Model'] + metrics].set_index('Model')

test_metrics.T.plot(kind='bar', ax=axes[0], rot=45)
axes[0].set_title('Model Comparison (Test Set)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].set_ylim([0, 1])
axes[0].legend(title='Model')
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Train vs Val vs Test Accuracy
accuracy_df = df_comparison[['Model', 'Dataset', 'Accuracy']].pivot(
    index='Dataset', columns='Model', values='Accuracy'
)

accuracy_df.plot(kind='bar', ax=axes[1], rot=0)
axes[1].set_title('Accuracy: Train vs Val vs Test', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1])
axes[1].legend(title='Model')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(output_dir, "06_model_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"   ✅ Visualisierung: {plot_path}")

# ==============================================================================
# ZUSAMMENFASSUNG
# ==============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Bestes Modell finden
best_model_row = test_df.loc[test_df['F1-Score'].idxmax()]

print(f"\nBestes Modell (basierend auf Test F1-Score):")
print(f"   Model: {best_model_row['Model']}")
print(f"   Test Accuracy:  {best_model_row['Accuracy']:.4f}")
print(f"   Test Precision: {best_model_row['Precision']:.4f}")
print(f"   Test Recall:    {best_model_row['Recall']:.4f}")
print(f"   Test F1-Score:  {best_model_row['F1-Score']:.4f}")
print(f"   Test ROC-AUC:   {best_model_row['ROC-AUC']:.4f}")

# Interpretation
if best_model_row['Accuracy'] > 0.55:
    print(f"\n   ✅ Das Modell ist besser als Zufall (>50%)")
else:
    print(f"\n   ⚠️  Das Modell ist kaum besser als Zufall")

if best_model_row['Accuracy'] > 0.60:
    print(f"   ✅ Das Modell zeigt vielversprechende Ergebnisse (>60%)")

print("\n" + "=" * 70)
