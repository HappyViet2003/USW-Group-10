"""
06_modelling/train_dqn.py
Smart-Version: Schnellere Validierung + GPU Support.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Importiere Environment
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from trading_env import make_trading_env

# --- KONFIGURATION ---
params_path = os.path.join(script_dir, "../../conf/params.yaml")
if os.path.exists(params_path):
    params = yaml.safe_load(open(params_path))

base_data_path = os.path.join(script_dir, "../../data")
processed_dir = os.path.join(base_data_path, "processed/prepared")
models_dir = os.path.join(base_data_path, "models")
log_dir = os.path.join(models_dir, "tensorboard")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Hyperparameter
TOTAL_TIMESTEPS = 500000
LEARNING_RATE = 0.0001
BUFFER_SIZE = 100000
BATCH_SIZE = 64

def main():
    print("=" * 70)
    print("STARTE DQN TRAINING (OPTIMIZED)")
    print("=" * 70)

    # 1. Hardware Check
    device = "auto"
    if torch.cuda.is_available():
        print(f"   🚀 GPU gefunden: {torch.cuda.get_device_name(0)}")
    else:
        print("   🐢 Keine GPU gefunden - Training läuft auf CPU.")

    # Multiprocessing Setup
    cpu_count = os.cpu_count() or 1
    n_envs = max(1, min(12, cpu_count - 1)) # Nutze Power!
    print(f"   ⚡ Multiprocessing: {n_envs} Envs.")

    # 2. Daten laden
    print("\n[1/4] Lade Trainingsdaten...")
    train_path = os.path.join(processed_dir, "train_prepared.parquet")
    val_path = os.path.join(processed_dir, "val_prepared.parquet")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    print(f"   Train: {train_df.shape}")

    # 3. Environments erstellen
    print(f"\n[2/4] Erstelle Environments...")

    def make_train_env():
        return make_trading_env(train_df, initial_balance=100000, fee_rate=0.001)

    # Train Env: Multiprocessing für Speed
    env = make_vec_env(make_train_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # Validation Env: Single-Core, aber limitiert auf 10k Steps (SPEED FIX!)
    eval_env = DummyVecEnv([lambda: make_trading_env(val_df, initial_balance=100000, fee_rate=0.001, max_steps=10000)])

    # 4. Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=max(10000 // n_envs, 1),
        deterministic=True,
        render=False,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path=models_dir,
        name_prefix="dqn_checkpoint"
    )

    # 5. Modell Definition
    print("\n[3/4] Definiere DQN Model...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=0.99,
        tau=0.005,
        exploration_fraction=0.1,
        verbose=1,
        device=device,
        tensorboard_log=log_dir
    )

    # 6. Training
    print("\n" + "="*70)
    print(f"STARTE TRAINING... (Lehn dich zurück)")
    print("="*70)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback, checkpoint_callback])

    final_path = os.path.join(models_dir, "dqn_final.zip")
    model.save(final_path)
    print(f"\n✅ Training fertig! Modell: {final_path}")

if __name__ == "__main__":
    import multiprocessing
    if sys.platform == 'win32':
        multiprocessing.freeze_support()
    main()