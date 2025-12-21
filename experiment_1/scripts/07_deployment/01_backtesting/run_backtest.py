"""
07_deployment/01_backtesting/run_backtest.py

Führt eine historische Simulation (Backtest) auf den Test-Daten durch.
Beantwortet die Fragen:
- Wie viele Trades?
- Durchschnittlicher Return?
- Vergleich zu Buy & Hold?
- Win Rate, Sharpe Ratio, Max Drawdown
- Trade Distribution über Zeit
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
# Wir müssen 3 Ebenen hoch: 01_backtesting -> 07_deployment -> scripts -> experiment_1
params_path = os.path.join(script_dir, "../../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Pfade zu den Daten
base_data_path = os.path.join(script_dir, "../../../data")
predictions_path = os.path.join(base_data_path, "models", "test_predictions.parquet")
raw_data_path = os.path.join(base_data_path, "processed", "merged_raw_data.parquet")
output_dir = os.path.join(script_dir, "../../../images")
os.makedirs(output_dir, exist_ok=True)

# Strategie-Parameter
CONFIDENCE_THRESHOLD = 0.62  # Erst kaufen, wenn Modell > 62% sicher ist
TRADING_FEE = 0.001          # 0.1% Gebühr pro Trade

print("=" * 70)
print("BACKTESTING SIMULATION (ENHANCED)")
print("=" * 70)

# ==============================================================================
# 1. DATEN LADEN & VORBEREITEN
# ==============================================================================
print("\n[1/6] Lade Daten...")

if not os.path.exists(predictions_path):
    print(f"❌ Fehler: Keine Vorhersagen gefunden unter {predictions_path}")
    print("   Bitte führe erst 'xgboost_model.py' aus!")
    exit(1)

# Lade Modell-Vorhersagen (Timestamp, Target, Prob)
preds = pd.read_parquet(predictions_path)

# Lade echte Preise (Close), um Returns zu berechnen
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
print("\n[2/6] Simuliere Strategie...")

# Position: 1 = Long (Investiert), 0 = Cash (Nicht investiert)
df['position'] = np.where(df['prob'] > CONFIDENCE_THRESHOLD, 1, 0)

# Brutto-Return der Strategie (Position * Markt-Bewegung)
df['strategy_gross'] = df['position'] * df['market_return']

# Gebühren berechnen: Wir zahlen Gebühr, wenn wir die Position ÄNDERN
df['trades'] = df['position'].diff().abs().fillna(0)
df['fees'] = df['trades'] * TRADING_FEE

# Netto-Return
df['strategy_net'] = df['strategy_gross'] - df['fees']

# Kumulierte Performance (Equity Curve)
df['cum_market'] = (1 + df['market_return']).cumprod()
df['cum_strategy'] = (1 + df['strategy_net']).cumprod()

# ==============================================================================
# 3. DETAILLIERTE KENNZAHLEN
# ==============================================================================
print("\n[3/6] Berechne erweiterte KPIs...")

# Basis-Metriken
total_trades = int(df['trades'].sum())
active_periods = df[df['position'] == 1]
win_trades = active_periods[active_periods['market_return'] > 0]
lose_trades = active_periods[active_periods['market_return'] < 0]

win_rate = len(win_trades) / (len(active_periods) + 1e-9)
avg_win = win_trades['market_return'].mean() if len(win_trades) > 0 else 0
avg_loss = lose_trades['market_return'].mean() if len(lose_trades) > 0 else 0

# Returns
total_return_market = df['cum_market'].iloc[-1] - 1
total_return_strategy = df['cum_strategy'].iloc[-1] - 1

# Sharpe Ratio (annualisiert, angenommen 365*24*60 Minuten pro Jahr)
strategy_returns = df['strategy_net']
sharpe_ratio = (strategy_returns.mean() / (strategy_returns.std() + 1e-9)) * np.sqrt(365*24*60)

# Maximum Drawdown
cummax = df['cum_strategy'].cummax()
drawdown = (df['cum_strategy'] - cummax) / cummax
max_drawdown = drawdown.min()

# Profit Factor
total_wins = win_trades['market_return'].sum() if len(win_trades) > 0 else 0
total_losses = abs(lose_trades['market_return'].sum()) if len(lose_trades) > 0 else 1e-9
profit_factor = total_wins / total_losses

# Zeit im Markt
time_in_market = (df['position'].sum() / len(df)) * 100

# ==============================================================================
# 4. PERFORMANCE REPORT
# ==============================================================================
print("-" * 70)
print(f"PERFORMANCE REPORT (Threshold: {CONFIDENCE_THRESHOLD})")
print("-" * 70)
print(f"\n📊 TRADING STATISTICS:")
print(f"   Total Trades:         {total_trades}")
print(f"   Winning Trades:       {len(win_trades)}")
print(f"   Losing Trades:        {len(lose_trades)}")
print(f"   Win Rate:             {win_rate:.2%}")
print(f"   Avg Win:              {avg_win:.4%}")
print(f"   Avg Loss:             {avg_loss:.4%}")
print(f"   Profit Factor:        {profit_factor:.2f}")
print(f"   Time in Market:       {time_in_market:.1f}%")

print(f"\n💰 RETURNS:")
print(f"   Buy & Hold Return:    {total_return_market:+.2%}")
print(f"   Strategy Return:      {total_return_strategy:+.2%}")
print(f"   Difference:           {(total_return_strategy - total_return_market):+.2%}")

print(f"\n📉 RISK METRICS:")
print(f"   Sharpe Ratio:         {sharpe_ratio:.2f}")
print(f"   Max Drawdown:         {max_drawdown:.2%}")
print(f"   Total Fees Paid:      {df['fees'].sum():.4%}")

print("-" * 70)

if total_return_strategy > total_return_market:
    print("🚀 RESULT: Strategy outperformed Buy & Hold!")
else:
    print("⚠️  RESULT: Buy & Hold was better.")
    print("    → Root Cause: Model accuracy (52%) insufficient for profitable trading.")
    print("    → Recommendation: Longer time horizons or improved risk management needed.")

# ==============================================================================
# 5. VISUALISIERUNGEN
# ==============================================================================
print("\n[4/6] Erstelle Visualisierungen...")

# Plot 1: Equity Curve
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Subplot 1: Kumulierte Returns
axes[0].plot(df['timestamp'], df['cum_market'], label='Buy & Hold (Bitcoin)', 
             color='gray', alpha=0.7, linewidth=2)
axes[0].plot(df['timestamp'], df['cum_strategy'], label='XGBoost Strategy', 
             color='green', linewidth=2)
axes[0].set_title(f'Backtest Result: Threshold > {CONFIDENCE_THRESHOLD} | Fees {TRADING_FEE:.1%}', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Cumulative Return (Factor)', fontsize=11)
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)

# Subplot 2: Drawdown
axes[1].fill_between(df['timestamp'], drawdown * 100, 0, color='red', alpha=0.3)
axes[1].plot(df['timestamp'], drawdown * 100, color='darkred', linewidth=1)
axes[1].set_title('Strategy Drawdown', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Drawdown (%)', fontsize=11)
axes[1].grid(True, alpha=0.3)

# Subplot 3: Trading Activity (Position über Zeit)
axes[2].fill_between(df['timestamp'], df['position'], 0, color='green', alpha=0.3, label='Long Position')
axes[2].set_title('Trading Activity (Position over Time)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Position (1=Long, 0=Cash)', fontsize=11)
axes[2].set_xlabel('Time', fontsize=11)
axes[2].set_ylim(-0.1, 1.1)
axes[2].legend(loc='upper left', fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(output_dir, "07_backtest_result.png")
plt.savefig(plot_path, dpi=150)
print(f"   ✅ Plot 1 gespeichert: {plot_path}")

# Plot 2: Trade Distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Returns Distribution
axes[0, 0].hist(active_periods['market_return'] * 100, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_title('Distribution of Trade Returns', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Return (%)', fontsize=10)
axes[0, 0].set_ylabel('Frequency', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Win/Loss Distribution
win_loss_counts = [len(win_trades), len(lose_trades)]
axes[0, 1].bar(['Winning Trades', 'Losing Trades'], win_loss_counts, color=['green', 'red'], alpha=0.7)
axes[0, 1].set_title(f'Win/Loss Distribution (Win Rate: {win_rate:.1%})', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Count', fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Subplot 3: Monthly Trade Count
df['month'] = df['timestamp'].dt.to_period('M')
monthly_trades = df.groupby('month')['trades'].sum()
axes[1, 0].bar(range(len(monthly_trades)), monthly_trades.values, color='orange', alpha=0.7)
axes[1, 0].set_title('Trade Distribution over Time (Monthly)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Month', fontsize=10)
axes[1, 0].set_ylabel('Number of Trades', fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Subplot 4: Confidence Distribution
axes[1, 1].hist(df['prob'], bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold ({CONFIDENCE_THRESHOLD})')
axes[1, 1].set_title('Model Confidence Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Predicted Probability', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path2 = os.path.join(output_dir, "07_backtest_distribution.png")
plt.savefig(plot_path2, dpi=150)
print(f"   ✅ Plot 2 gespeichert: {plot_path2}")

# ==============================================================================
# 6. EXPORT RESULTS
# ==============================================================================
print("\n[5/6] Exportiere Ergebnisse...")

# CSV mit allen Metriken
results_summary = {
    'Metric': [
        'Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate',
        'Avg Win', 'Avg Loss', 'Profit Factor', 'Time in Market (%)',
        'Buy & Hold Return', 'Strategy Return', 'Difference',
        'Sharpe Ratio', 'Max Drawdown', 'Total Fees'
    ],
    'Value': [
        total_trades, len(win_trades), len(lose_trades), f"{win_rate:.2%}",
        f"{avg_win:.4%}", f"{avg_loss:.4%}", f"{profit_factor:.2f}", f"{time_in_market:.1f}",
        f"{total_return_market:+.2%}", f"{total_return_strategy:+.2%}", 
        f"{(total_return_strategy - total_return_market):+.2%}",
        f"{sharpe_ratio:.2f}", f"{max_drawdown:.2%}", f"{df['fees'].sum():.4%}"
    ]
}

results_df = pd.DataFrame(results_summary)
results_path = os.path.join(base_data_path, "models", "backtest_results.csv")
results_df.to_csv(results_path, index=False)
print(f"   ✅ Results CSV gespeichert: {results_path}")

print("\n[6/6] ✅ Backtest abgeschlossen!")
print("=" * 70)
