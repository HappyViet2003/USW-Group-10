"""
07_deployment/01_backtesting/run_backtest.py

COMPLETE PRICE FIX VERSION:
- Fixed "Always Long" Bug
- Fixed Feature Selection
- Fixed regime column reconstruction
- Fixed Buy & Hold calculation (use original prices!)
- Fixed Backtest simulation (use original prices!)
- Disabled Plots (wegen Errors)
- Added Debug Output
"""

import os
import yaml
import pandas as pd
import numpy as np
import joblib

# --- CONFIG ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(script_dir, "../../../conf/params.yaml")
params = yaml.safe_load(open(params_path))

# Fixed paths
processed_dir = os.path.join(script_dir, "../../../data/processed")
model_dir = os.path.join(script_dir, "../../../data/models")

# Backtest Parameters
INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.001  # 0.1% per trade

# NOCH STRENGER:
THRESHOLD_BULL = 0.65      # Sehr hoch
THRESHOLD_BEAR = 0.40      # Sehr niedrig
THRESHOLD_SIDEWAYS = 0.65  # Sehr hoch

print("=" * 70)
print("ADAPTIVE BACKTEST (COMPLETE PRICE FIX VERSION)")
print("=" * 70)

# 1. Load Data & Model
print("\n[1/4] Lade Daten und Modell...")

# Lade prepared Daten (für Features & Predictions)
df = pd.read_parquet(os.path.join(processed_dir, "prepared/test_prepared.parquet"))
df = df.sort_values('timestamp')

# Filter: 2024 Test Set
df = df[df['timestamp'] >= '2024-01-01']

# WICHTIG: Lade ORIGINAL Daten (für echte Preise!)
df_original = pd.read_parquet(os.path.join(processed_dir, "training_data.parquet"))
df_original = df_original[df_original['timestamp'] >= '2024-01-01']
df_original = df_original.sort_values('timestamp')

# Merge: Füge originale Preise hinzu
df['close_original'] = df_original['close'].values

print(f"   Test Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Test Samples: {len(df)}")
print(f"   Price Range: ${df['close_original'].min():,.2f} - ${df['close_original'].max():,.2f}")

# Load Model
model_path = os.path.join(model_dir, "ensemble_model.pkl")
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    exit(1)

model = joblib.load(model_path)
print(f"   ✅ Model loaded: {model_path}")

# 2. Generate Predictions
print("\n[2/4] Generiere Predictions...")

# FIXED: Don't exclude 'open', 'high', 'low', 'close', 'volume'
# But exclude regime columns AND close_original!
feature_cols = [col for col in df.columns if col not in [
    'timestamp', 'target', 'sample_weight',
    'regime_bull', 'regime_bear', 'regime_sideways',
    'close_original'  # ← Exclude original price!
]]

X = df[feature_cols]
df['pred_proba'] = model.predict_proba(X)[:, 1]


# WICHTIG: Rekonstruiere 'regime' Spalte RICHTIG!
def get_regime(row):
    # Check welche Regime-Spalte = 1 ist
    if row.get('regime_bear', 0) == 1:
        return 'bear'
    elif row.get('regime_sideways', 0) == 1:
        return 'sideways'
    elif row.get('regime_bull', 0) == 1:
        return 'bull'
    else:
        # Fallback: Wenn alle 0 sind (sollte nicht passieren!)
        return 'sideways'


df['regime'] = df.apply(get_regime, axis=1)

print(f"   ✅ Predictions generated for {len(df)} samples")
print(
    f"   Prediction Stats: min={df['pred_proba'].min():.3f}, max={df['pred_proba'].max():.3f}, mean={df['pred_proba'].mean():.3f}")

# Debug: Check Regime Distribution
print(f"\n   Regime Distribution:")
print(df['regime'].value_counts())

# 3. FIXED: Adaptive Strategy (Realistic)
print("\n[3/4] Führe Adaptive Backtest durch...")


def get_signal_fixed(row):
    """
    FIXED: Realistic Adaptive Strategy

    Uses ML predictions properly with regime-specific thresholds.
    NO "always long" in bull markets!
    """
    prob = row['pred_proba']
    regime = row['regime']

    if regime == 'bull':
        # Bull Market: Go long with moderate confidence
        if prob > THRESHOLD_BULL:
            return 1  # Long
        else:
            return 0  # No position

    elif regime == 'bear':
        # Bear Market: Short with moderate threshold
        if prob < THRESHOLD_BEAR:
            return -1  # Short
        else:
            return 0  # No position

    else:  # sideways
        # Sideways: Mean reversion strategy
        if prob > THRESHOLD_SIDEWAYS:
            return 1  # Long
        elif prob < (1 - THRESHOLD_SIDEWAYS):
            return -1  # Short
        else:
            return 0  # No position


# Apply strategy
df['signal'] = df.apply(get_signal_fixed, axis=1)

# Debug: Check signal distribution by regime
print("\n   DEBUG: Signal Distribution by Regime:")
for regime in ['bull', 'bear', 'sideways']:
    regime_df = df[df['regime'] == regime]
    if len(regime_df) > 0:
        signal_counts = regime_df['signal'].value_counts()
        print(f"   {regime.capitalize():10s}: {dict(signal_counts)}")
        print(
            f"                Pred Proba: min={regime_df['pred_proba'].min():.3f}, max={regime_df['pred_proba'].max():.3f}, mean={regime_df['pred_proba'].mean():.3f}")

# Backtest Simulation (USE ORIGINAL PRICES!)
capital = INITIAL_CAPITAL
position = 0
entry_price = 0
trades = []

for idx, row in df.iterrows():
    current_price = row['close_original']  # ← USE ORIGINAL PRICE!
    signal = row['signal']

    # Close existing position if signal changes
    if position != 0 and signal != position:
        exit_price = current_price
        pnl_pct = (exit_price / entry_price - 1) * position
        fee = FEE_RATE * 2  # Entry + Exit
        net_pnl = pnl_pct - fee
        capital *= (1 + net_pnl)

        trades.append({
            'exit_time': row['timestamp'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'fee': fee,
            'net_pnl': net_pnl,
            'capital': capital
        })

    # Open new position
    if signal != 0:
        position = signal
        entry_price = current_price
        trades.append({
            'entry_time': row['timestamp'],
            'entry_price': entry_price,
            'position': position,
            'regime': row['regime']
        })
    else:
        position = 0

# Close final position
if position != 0:
    exit_price = df.iloc[-1]['close_original']  # ← USE ORIGINAL PRICE!
    pnl_pct = (exit_price / entry_price - 1) * position
    fee = FEE_RATE * 2
    net_pnl = pnl_pct - fee
    capital *= (1 + net_pnl)

# 4. Metriken
print("\n[4/4] Berechne Metriken...")

trades_df = pd.DataFrame(trades)
total_return = (capital / INITIAL_CAPITAL - 1) * 100
num_trades = len(trades_df) // 2  # Entry + Exit = 1 Trade

winning_trades = trades_df[trades_df['net_pnl'] > 0]
win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0

# Buy & Hold Comparison (USE ORIGINAL PRICES!)
buy_hold_return = (df.iloc[-1]['close_original'] / df.iloc[0]['close_original'] - 1) * 100

print("\n" + "=" * 70)
print("ADAPTIVE BACKTEST RESULTS (COMPLETE PRICE FIX VERSION)")
print("=" * 70)
print(f"Initial Capital:    ${INITIAL_CAPITAL:,.2f}")
print(f"Final Capital:      ${capital:,.2f}")
print(f"Total Return:       {total_return:+.2f}%")
print(f"Buy & Hold Return:  {buy_hold_return:+.2f}%")
print(f"Outperformance:     {total_return - buy_hold_return:+.2f}%")
print("-" * 70)
print(f"Total Trades:       {num_trades}")
print(f"Win Rate:           {win_rate:.2f}%")
if 'net_pnl' in trades_df.columns:
    print(f"Avg Trade:          {trades_df['net_pnl'].mean() * 100:.2f}%")
print("=" * 70)

# Regime-spezifische Analyse
print("\nPerformance by Regime:")
for regime in ['bull', 'bear', 'sideways']:
    regime_trades = trades_df[trades_df['regime'] == regime]
    if len(regime_trades) > 0:
        regime_return = regime_trades['net_pnl'].sum() * 100
        print(f"  {regime.capitalize():10s}: {regime_return:+.2f}%")

# Signal Distribution
print("\nSignal Distribution:")
print(df['signal'].value_counts())

print("\n" + "=" * 70)
print("BACKTEST COMPLETE (COMPLETE PRICE FIX VERSION)")
print("Plots deaktiviert (wegen matplotlib errors)")
print("=" * 70)