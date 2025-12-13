"""
07_deployment/01_backtesting/run_backtest.py

Führt eine historische Simulation (Backtest) auf den Test-Daten durch.
Beantwortet die Fragen:
- Wie viele Trades?
- Durchschnittlicher Return?
- Vergleich zu Buy & Hold?
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
# Wir müssen 3 Ebenen hoch: 01_backtesting -> 07_deployment -> scripts -> project
params_path = os.path.join(script_dir, "../../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Pfade zu den Daten
base_data_path = os.path.join(script_dir, "../../../data")
predictions_path = os.path.join(base_data_path, "models", "test_predictions.parquet")
raw_data_path = os.path.join(base_data_path, "processed", "merged_raw_data.parquet")
output_dir = os.path.join(script_dir, "../../../images")

# Strategie-Parameter
CONFIDENCE_THRESHOLD = 0.62  # Erst kaufen, wenn Modell > 53% sicher ist (Filtert Rauschen)
TRADING_FEE = 0.001          # 0.1% Gebühr pro Trade (Konservativ)

print("=" * 70)
print("BACKTESTING SIMULATION")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN & VORBEREITEN
# ==============================================================================
print("\n[1/4] Lade Daten...")

if not os.path.exists(predictions_path):
    print(f"❌ Fehler: Keine Vorhersagen gefunden unter {predictions_path}")
    print("   Bitte führe erst 'xgboost_model.py' aus!")
    exit(1)

# Lade Modell-Vorhersagen (Timestamp, Target, Prob)
preds = pd.read_parquet(predictions_path)

# Lade echte Preise (Close), um Returns zu berechnen
# Wir brauchen die Raw-Daten, weil 'prepare_for_modeling' die Preise entfernt hat.
raw = pd.read_parquet(raw_data_path)[['timestamp', 'close']]

# Merge auf Timestamp
df = pd.merge(preds, raw, on='timestamp', how='left')
df = df.sort_values('timestamp').reset_index(drop=True)

# Berechne den Return der NÄCHSTEN Kerze (das, was wir traden wollen)
df['market_return'] = df['close'].pct_change().shift(-1)
df.dropna(inplace=True)

print(f"   Zeitraum: {df['timestamp'].min()} bis {df['timestamp'].max()}")
print(f"   Datenpunkte: {len(df):,}")

# ==============================================================================
# 2. STRATEGIE-LOGIK
# ==============================================================================
print("\n[2/4] Simuliere Strategie...")

# Position: 1 = Long (Investiert), 0 = Cash (Nicht investiert)
# Wir kaufen nur, wenn die Wahrscheinlichkeit > Threshold ist
df['position'] = np.where(df['prob'] > CONFIDENCE_THRESHOLD, 1, 0)

# Brutto-Return der Strategie (Position * Markt-Bewegung)
df['strategy_gross'] = df['position'] * df['market_return']

# Gebühren berechnen:
# Wir zahlen Gebühr, wenn wir die Position ÄNDERN (Kaufen oder Verkaufen)
df['trades'] = df['position'].diff().abs().fillna(0)
df['fees'] = df['trades'] * TRADING_FEE

# Netto-Return
df['strategy_net'] = df['strategy_gross'] - df['fees']

# Kumulierte Performance (Equity Curve)
df['cum_market'] = (1 + df['market_return']).cumprod()
df['cum_strategy'] = (1 + df['strategy_net']).cumprod()

# ==============================================================================
# 3. KENNZAHLEN (METRIKEN)
# ==============================================================================
print("\n[3/4] Berechne KPIs (Key Performance Indicators)...")

total_trades = df['trades'].sum()
win_trades = df[(df['position'] == 1) & (df['market_return'] > 0)]
win_rate = len(win_trades) / (len(df[df['position'] == 1]) + 1e-9)

total_return_market = df['cum_market'].iloc[-1] - 1
total_return_strategy = df['cum_strategy'].iloc[-1] - 1

avg_return_per_trade = df[df['position'] == 1]['strategy_net'].mean()

print("-" * 50)
print(f"PERFORMANCE REPORT (Threshold: {CONFIDENCE_THRESHOLD})")
print("-" * 50)
print(f"Anzahl Trades:        {total_trades:.0f}")
print(f"Win Rate:             {win_rate:.2%}")
print(f"Avg Return per Trade: {avg_return_per_trade:.4%}")
print("-" * 50)
print(f"Total Return (Markt): {total_return_market:.2%}")
print(f"Total Return (Bot):   {total_return_strategy:.2%}")
print("-" * 50)

if total_return_strategy > total_return_market:
    print("🚀 ERGEBNIS: Der Bot hat den Markt geschlagen!")
else:
    print("⚠️ ERGEBNIS: Buy & Hold war besser (Gebühren oder Threshold anpassen).")

# ==============================================================================
# 4. PLOTTING
# ==============================================================================
print("\n[4/4] Erstelle Plot...")

plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['cum_market'], label='Buy & Hold (Bitcoin)', color='gray', alpha=0.6)
plt.plot(df['timestamp'], df['cum_strategy'], label='XGBoost Strategy', color='green', linewidth=1.5)

# Zeige nur Perioden, wo wir investiert sind (optional)
# plt.fill_between(df['timestamp'], df['cum_strategy'].min(), df['cum_strategy'].max(),
#                  where=df['position']==1, color='green', alpha=0.1, label="Active Trading")

plt.title(f'Backtest Ergebnis: Threshold > {CONFIDENCE_THRESHOLD} | Fees {TRADING_FEE:.1%}')
plt.ylabel('Kumulierter Return (Faktor)')
plt.xlabel('Zeit')
plt.legend()
plt.grid(True, alpha=0.3)

plot_path = os.path.join(output_dir, "07_backtest_result.png")
plt.savefig(plot_path)
print(f"   ✅ Plot gespeichert: {plot_path}")