"""
07_deployment/01_backtesting/run_rl_backtest.py
FIX: Nutzt ECHTE Preise für die Simulation, aber SKALIERTE Features für das Modell.
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
try:
    from trading_env import make_trading_env
except ImportError:
    # Fallback falls Datei lokal liegt
    from trading_env import make_trading_env

# --- KONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Pfade
base_data_path = os.path.join(script_dir, "../../../data")
processed_dir = os.path.join(base_data_path, "processed/prepared")
split_dir = "../../../data/processed/split" # Hier liegen die Rohdaten (test.parquet)
models_dir = os.path.join(base_data_path, "models")
images_dir = os.path.join(script_dir, "../../../images")
os.makedirs(images_dir, exist_ok=True)

print("=" * 70)
print("RL AGENT BACKTEST (REAL PRICE FIX)")
print("=" * 70)

# 1. Daten laden
print("\n[1/4] Lade Daten...")

# A) Skalierte Daten (für das Neuronale Netz)
test_df_scaled = pd.read_parquet(os.path.join(processed_dir, "test_prepared.parquet"))
print(f"   Modell-Input (Scaled): {test_df_scaled.shape}")

# B) Echte Daten (für den Kontostand)
# Wir laden die originale test.parquet aus dem Split-Ordner
raw_test_path = os.path.join(split_dir, "test.parquet")
if not os.path.exists(raw_test_path):
    print(f"❌ Rohdaten nicht gefunden: {raw_test_path}")
    exit(1)

test_df_raw = pd.read_parquet(raw_test_path)
print(f"   Simulation-Input (Raw): {test_df_raw.shape}")

# SICHERHEITS-CHECK: Sind die Dateien synchron?
if len(test_df_scaled) != len(test_df_raw):
    print("⚠️ ACHTUNG: Unterschiedliche Länge! Schneide auf das Minimum zu.")
    min_len = min(len(test_df_scaled), len(test_df_raw))
    test_df_scaled = test_df_scaled.iloc[:min_len]
    test_df_raw = test_df_raw.iloc[:min_len]

# C) MERGE: Wir injizieren den ECHTEN Preis in das DataFrame
# Das Environment nutzt 'close' für die Balance-Berechnung.
# Der Agent nutzt 'close' NICHT (da es in trading_env aus den Features entfernt wird).
# Daher können wir 'close' sicher überschreiben!
test_df_combined = test_df_scaled.copy()
test_df_combined['close'] = test_df_raw['close'].values # Überschreibe mit echten Dollar-Preisen

print("   ✅ Daten fusioniert: Features sind skaliert, Preis ist in $.")

# 2. Modell laden
print("\n[2/4] Lade trainiertes DQN Modell...")
model_path = os.path.join(models_dir, "dqn_final.zip")

if not os.path.exists(model_path):
    print(f"❌ Modell nicht gefunden: {model_path}")
    exit(1)

model = DQN.load(model_path)
print(f"   ✅ Modell geladen.")

# 3. Environment erstellen
print("\n[3/4] Erstelle Test Environment...")
test_env = make_trading_env(
    test_df_combined, # Hier ist jetzt der echte Preis drin!
    initial_balance=100000,
    fee_rate=0.001,
    max_steps=len(test_df_combined)
)

# 4. Backtest
print("\n[4/4] Führe Backtest durch...")

obs, info = test_env.reset()
done = False
step = 0
history = []

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

    history.append({
        'step': step,
        'balance': info['balance'],
        'position': info['position'],
        'reward': reward,
        'action': action,
        'price': test_df_combined.iloc[step]['close'] # Logge den Preis für Plot
    })

    step += 1
    if step % 10000 == 0:
        print(f"   Step {step}/{len(test_df_combined)}: Balance=${info['balance']:,.2f}")

# 5. Metriken
print("\n" + "=" * 70)
print("REALISTIC RL RESULTS")
print("=" * 70)

history_df = pd.DataFrame(history)
initial_balance = 100000
final_balance = history_df.iloc[-1]['balance']
total_return = (final_balance / initial_balance - 1) * 100

# Echter Buy & Hold Return
first_price = test_df_raw.iloc[0]['close']
last_price = test_df_raw.iloc[len(history_df)-1]['close']
buy_hold_return = (last_price / first_price - 1) * 100

print(f"Initial Balance:    ${initial_balance:,.2f}")
print(f"Final Balance:      ${final_balance:,.2f}")
print(f"Total Return:       {total_return:+.2f}%")
print(f"Buy & Hold Return:  {buy_hold_return:+.2f}%") # Das sollte jetzt realistisch sein (z.B. -50% oder +20%)
print(f"Outperformance:     {total_return - buy_hold_return:+.2f}%")
print("-" * 70)
actions = history_df['action'].value_counts()
print(f"Hold: {actions.get(0, 0)}")
print(f"Buy:  {actions.get(1, 0)}")
print(f"Sell: {actions.get(2, 0)}")
print("=" * 70)

# 6. Plots
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
# Equity
axes[0].plot(history_df['balance'], label='RL Agent')
axes[0].plot(history_df['price'] * (initial_balance/first_price), alpha=0.5, label='Buy & Hold (scaled)')
axes[0].set_title('Equity Curve (Real $$$)')
axes[0].legend()
# Position
axes[1].plot(history_df['position'], color='orange')
axes[1].set_title('Position')
# Rewards
axes[2].hist(history_df['reward'], bins=50)
axes[2].set_title('Rewards')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "05_rl_backtest_REAL.png"))
print("✅ Fertig.")