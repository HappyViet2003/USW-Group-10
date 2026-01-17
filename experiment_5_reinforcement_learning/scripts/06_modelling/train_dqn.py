"""
06_modelling/train_dqn.py

Experiment 5: Deep Q-Network (DQN) Training
Trainiert RL Agent für Bitcoin Trading mit Stable-Baselines3.
"""

import os
import yaml
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from trading_env import make_trading_env

# --- KONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Pfade
base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
processed_dir = os.path.join(base_data_path, "processed/prepared")
models_dir = os.path.join(base_data_path, "../../models")
os.makedirs(models_dir, exist_ok=True)

print("=" * 70)
print("DQN TRAINING (EXP 5: REINFORCEMENT LEARNING)")
print("=" * 70)

# 1. Daten laden
print("\n[1/4] Lade Trainingsdaten...")
train_df = pd.read_parquet(os.path.join(processed_dir, "train_prepared.parquet"))
val_df = pd.read_parquet(os.path.join(processed_dir, "val_prepared.parquet"))

print(f"   Train: {train_df.shape}")
print(f"   Val: {val_df.shape}")

# 2. Environments erstellen
print("\n[2/4] Erstelle Trading Environments...")

train_env = make_trading_env(
    train_df, 
    initial_balance=100000, 
    fee_rate=0.001,
    max_steps=len(train_df)
)

val_env = make_trading_env(
    val_df,
    initial_balance=100000,
    fee_rate=0.001,
    max_steps=len(val_df)
)

# Wrap with Monitor for logging
train_env = Monitor(train_env)
val_env = Monitor(val_env)

print("   ✅ Training Environment")
print("   ✅ Validation Environment")

# 3. DQN Model definieren
print("\n[3/4] Definiere DQN Model...")

model = DQN(
    policy='MlpPolicy',
    env=train_env,
    learning_rate=0.0001,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=0.005,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10,
    tensorboard_log=os.path.join(models_dir, "tensorboard"),
    verbose=1,
    seed=42
)

print("   ✅ DQN Model erstellt")
print(f"   Policy: MlpPolicy")
print(f"   Learning Rate: 0.0001")
print(f"   Buffer Size: 100,000")
print(f"   Batch Size: 64")

# 4. Callbacks
print("\n[4/4] Setup Callbacks...")

# Evaluation Callback (testet auf Validation Set)
eval_callback = EvalCallback(
    val_env,
    best_model_save_path=models_dir,
    log_path=models_dir,
    eval_freq=5000,
    deterministic=True,
    render=False,
    verbose=1
)

# Checkpoint Callback (speichert regelmäßig)
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=models_dir,
    name_prefix='dqn_checkpoint'
)

print("   ✅ Eval Callback (every 5k steps)")
print("   ✅ Checkpoint Callback (every 10k steps)")

# 5. Training
print("\n" + "=" * 70)
print("STARTE DQN TRAINING")
print("=" * 70)
print("⚠️  Training kann 30-60 Minuten dauern!")
print("⚠️  Überwache mit TensorBoard:")
print(f"    tensorboard --logdir {os.path.join(models_dir, 'tensorboard')}")
print("=" * 70)

TOTAL_TIMESTEPS = 500000

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    log_interval=100,
    progress_bar=False  # Disabled (tqdm not installed)
)

# 6. Speichern
print("\n[6/6] Speichere finales Modell...")
final_model_path = os.path.join(models_dir, "dqn_final.zip")
model.save(final_model_path)
print(f"   ✅ Modell: {final_model_path}")

print("\n" + "=" * 70)
print("✅ DQN TRAINING ABGESCHLOSSEN")
print("=" * 70)
print(f"\nNächster Schritt:")
print(f"  python 07_deployment/01_backtesting/run_rl_backtest.py")
