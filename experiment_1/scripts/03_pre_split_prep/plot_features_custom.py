import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import yaml

# --- SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params = yaml.safe_load(open(os.path.join(script_dir, "../../conf/params.yaml")))
base_path = params['DATA_ACQUISITON']['DATA_PATH']
feat_path = os.path.join(script_dir, base_path, "processed/training_data.parquet")

print(f"Lade Daten von: {feat_path}")
df = pd.read_parquet(feat_path)

# Timestamp sicherstellen
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- ZEITRAUM FILTERN (NOVEMBER 2024) ---
# Wir nehmen einen vollen Monat ohne viele Feiertage für saubere Charts
start_date = "2024-11-01"
end_date = "2024-12-01"

mask = (df['timestamp'] >= start_date) & (df['timestamp'] < end_date)
subset = df.loc[mask].copy()

if len(subset) == 0:
    print("⚠️ Warnung: Keine Daten im gewählten Zeitraum gefunden! Prüfe das Jahr.")
    # Fallback: Letzte 20.000 Zeilen
    subset = df.iloc[-20000:].copy()
else:
    print(f"Zeige Zeitraum: {start_date} bis {end_date} ({len(subset)} Zeilen)")

# --- PLOTTING ---
fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

# 1. BITCOIN vs. NVIDIA
ax1 = axes[0]
ax1.plot(subset['timestamp'], subset['close'], label='Bitcoin ($)', color='black', linewidth=1.5)
ax1.set_ylabel('Bitcoin', color='black', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)

if 'nvda_close' in subset.columns:
    ax2 = ax1.twinx()
    ax2.plot(subset['timestamp'], subset['nvda_close'], label='Nvidia ($)', color='green', linestyle='--',
             linewidth=1.5)
    ax2.set_ylabel('Nvidia', color='green', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='green')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
else:
    ax1.legend(loc='upper left')

ax1.set_title('Market Driver: Bitcoin vs. AI-Hype (November 2024)', fontsize=12, fontweight='bold')

# 2. RATIO BTC/NVDA
if 'ratio_btc_nvda' in subset.columns:
    axes[1].plot(subset['timestamp'], subset['ratio_btc_nvda'], label='Ratio BTC/NVDA', color='tab:blue')
    axes[1].set_title('Relative Strength: Entkopplung Bitcoin vs. Nvidia', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Ratio')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

# 3. RSI
axes[2].plot(subset['timestamp'], subset['rsi_14_norm'], label='RSI 14 (Z-Norm)', color='purple', alpha=0.8)
axes[2].axhline(2, color='red', linestyle='--', alpha=0.5)
axes[2].axhline(-2, color='green', linestyle='--', alpha=0.5)
axes[2].set_title('Momentum (RSI)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

# 4. PRICE SPEED (SLOPE)
axes[3].plot(subset['timestamp'], subset['slope_close_norm'], label='Price Speed (Slope)', color='orange', alpha=0.8)
axes[3].fill_between(subset['timestamp'], subset['slope_close_norm'], 0, alpha=0.2, color='orange')
axes[3].set_title('Trend-Geschwindigkeit', fontsize=12, fontweight='bold')
axes[3].grid(True, alpha=0.3)

# X-Achse formatieren
date_fmt = mdates.DateFormatter('%d.%m')  # Nur Tag.Monat
axes[3].xaxis.set_major_formatter(date_fmt)
plt.gcf().autofmt_xdate()

plt.tight_layout()
save_p = os.path.join(script_dir, "../../images/03_feature_deepdive_nov.png")
plt.savefig(save_p)
print(f"✅ Plot gespeichert: {save_p}")