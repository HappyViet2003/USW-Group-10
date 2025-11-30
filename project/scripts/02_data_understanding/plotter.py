import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import yaml

# --- KONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
params = yaml.safe_load(open(os.path.join(script_dir, "../../conf/params.yaml")))

# Pfade zu den ROHDATEN (Schritt 2: Wir schauen uns die Quellen an)
base_data_path = params['DATA_ACQUISITON']['DATA_PATH']
crypto_dir = os.path.join(script_dir, base_data_path, "Bars_1m_crypto")
external_dir = os.path.join(script_dir, base_data_path, "external_data")
image_dir = os.path.join(script_dir, "../../images")
os.makedirs(image_dir, exist_ok=True)

# Welche Dateien wollen wir analysieren?
files_to_check = {
    'Bitcoin (Alpaca)': os.path.join(crypto_dir, "BTC_USD.parquet"),
    'Nvidia (Alpaca)': os.path.join(external_dir, "NVDA.parquet"),
    'Nasdaq Future (Yahoo)': os.path.join(external_dir, "NASDAQ_FUTURE.parquet"),
    'US Zinsen (Yahoo)': os.path.join(external_dir, "US_INTEREST_RATES.parquet"),
    'Geldmenge M2 (FRED)': os.path.join(external_dir, "M2.parquet")
}

# Container für Daten (für die Plots später)
data_store = {}

print("=" * 60)
print("🧐 DATA UNDERSTANDING: ANALYSE DER ROHDATEN")
print("=" * 60)

# ==============================================================================
# TEIL 1: STATISTISCHE ANALYSE DER EINZELDATEIEN
# ==============================================================================

for name, path in files_to_check.items():
    print(f"\n--- Analysiere: {name} ---")

    if not os.path.exists(path):
        print(f"❌ Datei nicht gefunden: {path}")
        continue

    # Laden
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Speichern für Plots
    data_store[name] = df

    # REPORTING (Das will der Dozent sehen)
    print(f"   Zeitraum:   {df['timestamp'].min()} bis {df['timestamp'].max()}")
    print(f"   Anzahl:     {len(df):,} Zeilen")
    print(f"   Frequenz:   {(df['timestamp'].iloc[1] - df['timestamp'].iloc[0])} (geschätzt)")

    # Statistik der 'close' Spalte (oder value)
    col = 'close' if 'close' in df.columns else df.columns[1]  # Fallback
    stats = df[col].describe()
    print(f"   Preis-Info: Min={stats['min']:.2f} | Max={stats['max']:.2f} | Mean={stats['mean']:.2f}")

    # Zeige Lücken (Missing Values / Gaps)
    # Wir schauen, ob zwischen zwei Zeilen mehr als erwartet Zeit vergeht
    time_diffs = df['timestamp'].diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(days=3)]
    if len(large_gaps) > 0:
        print(f"   ⚠️  Große Datenlücken (>3 Tage): {len(large_gaps)} mal gefunden (z.B. Wochenenden)")
    else:
        print("   ✅ Keine großen Lücken (>3 Tage)")

# ==============================================================================
# TEIL 2: VISUALISIERUNG DER ROHDATEN
# ==============================================================================
print("\n" + "=" * 60)
print("🎨 ERSTELLE PLOTS AUS ROHDATEN")
print("=" * 60)

plt.style.use('seaborn-v0_8-darkgrid')

# --- PLOT B: MAKRO-DATEN (Low Frequency) ---
# Zeigt die Daten, die wir seltener bekommen (M2, Zinsen)
print("   Erstelle Plot B: Makro-Indikatoren...")

fig, ax1 = plt.subplots(figsize=(12, 6))

# Zinsen (Links)
if 'US Zinsen (Yahoo)' in data_store:
    df_rates = data_store['US Zinsen (Yahoo)']
    ax1.plot(df_rates['timestamp'], df_rates['close'], color='red', label='US 10Y Yields (Täglich)', linewidth=2)
    ax1.set_ylabel("Zinsen (%)", color='red', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='red')

# M2 (Rechts)
if 'Geldmenge M2 (FRED)' in data_store:
    ax2 = ax1.twinx()
    df_m2 = data_store['Geldmenge M2 (FRED)']
    ax2.plot(df_m2['timestamp'], df_m2['close'], color='blue', label='M2 Geldmenge (Wöchentlich)', linewidth=2,
             linestyle='--')
    ax2.set_ylabel("M2 Supply (Mrd $)", color='blue', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')

# Legende kombinieren
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels() if 'ax2' in locals() else ([], [])
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title("Makro-Umfeld: Zinsen vs. Geldmenge (Unterschiedliche Frequenzen)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_b = os.path.join(image_dir, "02_raw_data_macro.png")
plt.savefig(save_b)
print(f"   ✅ Gespeichert: {save_b}")

print("\n🏁 Fertig! Kopiere den Text-Output (Statistiken) in deine 'Data Understanding' Folie.")