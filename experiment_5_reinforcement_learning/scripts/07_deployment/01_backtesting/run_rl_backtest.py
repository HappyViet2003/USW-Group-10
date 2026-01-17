"""
07_deployment/01_backtesting/run_rl_backtest.py

Experiment 5: RL Agent Backtesting
Evaluiert trainierten DQN Agent auf Test-Set.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

# Import Trading Environment
sys.path.append(os.path.join(os.path.dirname(__file__), "../../06_modelling"))
from trading_env import make_trading_env

# --- KONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Pfade
base_data_path = os.path.join(script_dir, "../../../data")
processed_dir = os.path.join(base_data_path, "processed/prepared")
models_dir = os.path.join(script_dir, "../../../models")
images_dir = os.path.join(script_dir, "../../../images")
os.makedirs(images_dir, exist_ok=True)

print("=" * 70)
print("RL AGENT BACKTEST (EXP 5)")
print("=" * 70)

# 1. Daten laden
print("\n[1/4] Lade Test-Daten...")
test_df = pd.read_parquet(os.path.join(processed_dir, "test_prepared.parquet"))
print(f"   Test: {test_df.shape}")

# 2. Modell laden
print("\n[2/4] Lade trainiertes DQN Modell...")
model_path = os.path.join(models_dir, "dqn_final.zip")

if not os.path.exists(model_path):
    print(f"❌ Modell nicht gefunden: {model_path}")
    print("   Führe zuerst train_dqn.py aus!")
    exit(1)

model = DQN.load(model_path)
print(f"   ✅ Modell geladen: {model_path}")

# 3. Environment erstellen
print("\n[3/4] Erstelle Test Environment...")
test_env = make_trading_env(
    test_df,
    initial_balance=100000,
    fee_rate=0.001,
    max_steps=len(test_df)
)

# 4. Backtest
print("\n[4/4] Führe Backtest durch...")

obs, info = test_env.reset()  # gymnasium returns (obs, info)
done = False
step = 0
history = []

while not done:
    # Agent wählt Action
    action, _states = model.predict(obs, deterministic=True)
    
    # Execute action
    obs, reward, terminated, truncated, info = test_env.step(action)  # gymnasium returns 5 values
    done = terminated or truncated
    
    # Log
    history.append({
        'step': step,
        'balance': info['balance'],
        'position': info['position'],
        'reward': reward,
        'action': action
    })
    
    step += 1
    
    if step % 1000 == 0:
        print(f"   Step {step}/{len(test_df)}: Balance=${info['balance']:,.2f}")

# 5. Metriken
print("\n" + "=" * 70)
print("RL BACKTEST RESULTS")
print("=" * 70)

history_df = pd.DataFrame(history)

initial_balance = 100000
final_balance = history_df.iloc[-1]['balance']
total_return = (final_balance / initial_balance - 1) * 100

# Buy & Hold Comparison
buy_hold_return = (test_df.iloc[-1]['close'] / test_df.iloc[0]['close'] - 1) * 100

# Trade Statistics
actions = history_df['action'].value_counts()
num_trades = len(history_df[history_df['action'] != 0])  # Non-hold actions

print(f"Initial Balance:    ${initial_balance:,.2f}")
print(f"Final Balance:      ${final_balance:,.2f}")
print(f"Total Return:       {total_return:+.2f}%")
print(f"Buy & Hold Return:  {buy_hold_return:+.2f}%")
print(f"Outperformance:     {total_return - buy_hold_return:+.2f}%")
print("-" * 70)
print(f"Total Actions:      {len(history_df)}")
print(f"Hold Actions:       {actions.get(0, 0)} ({actions.get(0, 0)/len(history_df)*100:.1f}%)")
print(f"Buy Actions:        {actions.get(1, 0)} ({actions.get(1, 0)/len(history_df)*100:.1f}%)")
print(f"Sell Actions:       {actions.get(2, 0)} ({actions.get(2, 0)/len(history_df)*100:.1f}%)")
print("=" * 70)

# 6. Visualisierung
print("\n[6/6] Erstelle Plots...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Equity Curve
ax1 = axes[0]
ax1.plot(history_df['step'], history_df['balance'], label='RL Agent', linewidth=2)
ax1.axhline(initial_balance, color='gray', linestyle='--', label='Initial Balance')
ax1.set_title('Equity Curve: RL Agent (Exp 5)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Step')
ax1.set_ylabel('Balance ($)')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Position over Time
ax2 = axes[1]
ax2.plot(history_df['step'], history_df['position'], label='Position', linewidth=1.5)
ax2.fill_between(history_df['step'], 0, history_df['position'], alpha=0.3)
ax2.set_title('Position over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Step')
ax2.set_ylabel('Position (-1=Short, 0=Cash, 1=Long)')
ax2.set_ylim(-1.5, 1.5)
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Reward Distribution
ax3 = axes[2]
rewards = history_df['reward']
ax3.hist(rewards, bins=50, alpha=0.7, edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
ax3.set_title('Reward Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Reward')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(images_dir, "05_rl_backtest.png")
plt.savefig(plot_path, dpi=150)
print(f"   ✅ Plot: {plot_path}")

print("\n" + "=" * 70)
print("✅ RL BACKTEST ABGESCHLOSSEN")
print("=" * 70)
