import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

# --- KONFIGURATION ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(script_dir, "../../conf/params.yaml")
    params = yaml.safe_load(open(params_path))

    # Datenpfade
    base_data_path = os.path.join(script_dir, params['DATA_ACQUISITON']['DATA_PATH'])
    crypto_path = os.path.join(base_data_path, "Bars_1m_crypto", "BTC_USD.parquet")
    external_path = os.path.join(base_data_path, "external_data")
except FileNotFoundError:
    print("Error: 'conf/params.yaml' not found. Please check your config.")
    exit()

# Bild-Speicherpfad
image_dir = os.path.join(script_dir, "../../images")
os.makedirs(image_dir, exist_ok=True)


# --- FUNKTIONEN ---
def load_and_merge_data():
    """Lädt BTC und die externen Assets (GLD, QQQ, UUP) und merged sie."""
    print("Lade Daten...")

    # 1. Bitcoin laden
    if not os.path.exists(crypto_path):
        raise FileNotFoundError(f"BTC Datei nicht gefunden: {crypto_path}")
    df_btc = pd.read_parquet(crypto_path)[['timestamp', 'close']]
    df_btc.rename(columns={'close': 'BTC'}, inplace=True)

    # 2. Externe Assets laden
    assets = {'QQQ': 'Nasdaq-100', 'GLD': 'Gold', 'UUP': 'USD-Index'}

    df_merged = df_btc.copy()

    for symbol, name in assets.items():
        path = os.path.join(external_path, f"{symbol}.parquet")
        if os.path.exists(path):
            df_ext = pd.read_parquet(path)[['timestamp', 'close']]
            df_ext.rename(columns={'close': name}, inplace=True)

            # Merge auf Timestamp (Left Join behält BTC Zeitstempel)
            df_merged = pd.merge(df_merged, df_ext, on='timestamp', how='left')
        else:
            print(f"⚠ Warnung: {symbol} nicht gefunden.")

    # Sortieren und Forward Fill für Lücken (Wochenenden bei Aktien)
    df_merged.sort_values('timestamp', inplace=True)
    df_merged.ffill(inplace=True)
    df_merged.dropna(inplace=True)  # Entfernt Zeilen ganz am Anfang ohne Daten

    return df_merged


def plot_normalized_history(df):
    """Zeigt alle Assets relativ zum Startpunkt (Start = 100%)."""
    print("Erstelle Plot 1: Normalisierter Verlauf...")

    # Normalisierung: (Preis / Startpreis) * 100
    df_norm = df.set_index('timestamp').copy()
    df_norm = (df_norm / df_norm.iloc[0]) * 100

    plt.figure(figsize=(12, 6))
    for column in df_norm.columns:
        # BTC etwas dicker, damit er auffällt
        linewidth = 2 if column == 'BTC' else 1
        alpha = 1 if column == 'BTC' else 0.7
        plt.plot(df_norm.index, df_norm[column], label=column, linewidth=linewidth, alpha=alpha)

    plt.title("Performance Vergleich: BTC vs. Makro-Assets (Start = 100)", fontsize=14)
    plt.xlabel("Datum")
    plt.ylabel("Relative Performance (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(image_dir, "02_performance_comparison.png")
    plt.savefig(save_path)
    print(f" Gespeichert unter: {save_path}")
    plt.close()


def plot_subplots(df):
    """Zeigt jedes Asset in einem eigenen Chart (für absolute Preise)."""
    print("Erstelle Plot 2: Einzelne Verläufe...")

    assets = [c for c in df.columns if c != 'timestamp']
    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 12), sharex=True)

    for i, asset in enumerate(assets):
        ax = axes[i]
        ax.plot(df['timestamp'], df[asset], color='tab:blue')
        ax.set_title(asset)
        ax.set_ylabel("Preis ($)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(image_dir, "02_asset_subplots.png")
    plt.savefig(save_path)
    print(f" Gespeichert unter: {save_path}")
    plt.close()


def plot_correlation_matrix(df):
    """Erstellt eine Heatmap der Korrelationen."""
    print("Erstelle Plot 3: Korrelations-Matrix...")

    # Wir berechnen die Korrelation der täglichen Returns (Veränderungen),
    # nicht der absoluten Preise! (Wichtig für Statistik)
    df_returns = df.set_index('timestamp').pct_change().dropna()

    corr_matrix = df_returns.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Korrelation der Returns (1 Minute)", fontsize=14)

    save_path = os.path.join(image_dir, "02_correlation_matrix.png")
    plt.savefig(save_path)
    print(f" Gespeichert unter: {save_path}")
    plt.close()


# --- MAIN ---
if __name__ == "__main__":
    data = load_and_merge_data()

    # Plots erstellen
    plot_normalized_history(data)
    plot_subplots(data)
    plot_correlation_matrix(data)

    print("\nFertig! Prüfe den Ordner 'experiment_1/images'.")