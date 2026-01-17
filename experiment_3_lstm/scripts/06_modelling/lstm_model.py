"""
06_modelling/lstm_model.py

Experiment 3: Deep Learning (LSTM)
FIX: Memory-Optimierte Sequenz-Erstellung (Pre-Allocation + Float32).
Verhindert RAM-Absturz durch Vermeidung von Python-Listen-Overhead.
"""

import os
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score
import gc  # Garbage Collector

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

prep_dir = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'], "processed", "prepared")
models_dir = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'], "models")
os.makedirs(models_dir, exist_ok=True)

# LSTM Parameter
SEQ_LEN = 60   # 60 Minuten Rückblick
BATCH_SIZE = 2048 # Größerer Batch-Size für schnelleres Training auf GPU/CPU
EPOCHS = 20

print("="*70 + "\nLSTM TRAINING (Experiment 3: Optimized)\n" + "="*70)

# 1. DATEN LADEN
print("[1] Lade Daten...")
train_df = pd.read_parquet(os.path.join(prep_dir, "train_prepared.parquet"))
val_df = pd.read_parquet(os.path.join(prep_dir, "val_prepared.parquet"))
test_df = pd.read_parquet(os.path.join(prep_dir, "test_prepared.parquet"))

# 2. FEATURE SELECTION
exclude_cols = [
    'timestamp', 'target', 'sample_weight', 'future_close', 'year_month',
    'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count', 'local_high', 'local_low',
    'sma_50', 'ema_200',
    'qqq_open', 'qqq_high', 'qqq_low', 'qqq_close', 'qqq_volume', 'qqq_trade_count', 'qqq_vwap',
    'nvda_open', 'nvda_high', 'nvda_low', 'nvda_close', 'nvda_volume', 'nvda_trade_count', 'nvda_vwap',
    'gold_open', 'gold_high', 'gold_low', 'gold_close', 'gold_volume', 'gold_trade_count', 'gold_vwap',
    'usd_open', 'usd_high', 'usd_low', 'usd_close', 'usd_volume', 'usd_trade_count', 'usd_vwap',
    'nq_open', 'nq_high', 'nq_low', 'nq_close', 'nq_volume',
    'm2_close', 'm2_open', 'm2_high', 'm2_low',
    'rates_close', 'rates_open',
    'onchain_hash-rate', 'onchain_n-unique-addresses'
]
feature_cols = [c for c in train_df.columns if c not in exclude_cols]
print(f"   Features: {len(feature_cols)} (Using float32 to save RAM)")

# 3. SEQUENZEN ERSTELLEN (MEMORY OPTIMIZED)
def create_sequences_optimized(df, seq_len):
    """
    Erstellt Sequenzen durch Pre-Allocation statt Listen-Append.
    Spart extrem viel Arbeitsspeicher.
    """
    # 1. Daten als Float32 extrahieren (Halbiert Speicherbedarf vs Float64)
    data_array = df[feature_cols].values.astype(np.float32)
    target_array = df['target'].values.astype(np.float32)

    num_samples = len(data_array) - seq_len
    num_features = data_array.shape[1]

    print(f"   Allocating Array: ({num_samples}, {seq_len}, {num_features}) ...")

    # 2. Leeren Container erstellen (Sofortige Reservierung des Blocks)
    X_seq = np.zeros((num_samples, seq_len, num_features), dtype=np.float32)
    y_seq = np.zeros((num_samples,), dtype=np.float32)

    # 3. Array füllen (Schneller Loop)
    # Hinweis: In Python ist dieser Loop bei >1 Mio Zeilen etwas langsam,
    # aber er verhindert den Memory-Crash.
    for i in range(num_samples):
        X_seq[i] = data_array[i:i+seq_len]
        y_seq[i] = target_array[i+seq_len]

    return X_seq, y_seq

print("\n[2] Erstelle Sequenzen (Geduld, Loop läuft)...")

# Train
print("   Processing Train Set...")
X_train, y_train = create_sequences_optimized(train_df, SEQ_LEN)
# DataFrame sofort löschen um RAM freizugeben
del train_df
gc.collect()

# Val
print("   Processing Val Set...")
X_val, y_val = create_sequences_optimized(val_df, SEQ_LEN)
del val_df
gc.collect()

# Test
print("   Processing Test Set...")
X_test, y_test = create_sequences_optimized(test_df, SEQ_LEN)
del test_df
gc.collect()

print(f"\n   Final Train Shape: {X_train.shape}")

# 4. MODELL BAUEN
print("\n[3] Baue LSTM...")
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    # Layer 1: Mehr Neuronen, um Muster zu finden
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    # Layer 2: Verdichtung
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# 5. TRAINING
print("\n[4] Starte Training...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE, # Größere Batches = Weniger Overhead
    callbacks=[early_stop],
    verbose=1
)

# 6. EVALUATION
print("\n[5] Evaluation auf Test-Set...")
preds_prob = model.predict(X_test, batch_size=BATCH_SIZE).flatten()
preds_class = (preds_prob > 0.5).astype(int)

acc = accuracy_score(y_test, preds_class)
auc = roc_auc_score(y_test, preds_prob)

print("-" * 50)
print(f"ERGEBNIS EXPERIMENT 3 (LSTM):")
print(f"   ✅ Test Accuracy: {acc:.2%}")
print(f"   ✅ Test AUC:      {auc:.4f}")
print("-" * 50)

# Speichern
model_path = os.path.join(models_dir, "lstm_model.keras")
model.save(model_path)
print(f"💾 Modell gespeichert: {model_path}")