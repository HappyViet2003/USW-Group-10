import pandas as pd
import os

# Pfad zu deiner Parquet-Datei
# Tipp: In PyCharm Rechtsklick auf die Datei -> "Copy Path/Reference" -> "Absolute Path"
file_path_BTC = "../../data/Bars_1m_crypto/BTC_USD.parquet"
file_path_GLD = "../../data/external_data/GLD.parquet"
file_path_QQQ = "../../data/external_data/QQQ.parquet"
file_path_UUP = "../../data/external_data/UUP.parquet"
file_path_M2 = "../../data/external_data/M2.parquet"
file_path_US_Interest_Rate = "../../data/external_data/US_INTEREST_RATES.parquet"

def load_and_display_info(file_path):
    df = pd.read_parquet(file_path)

    print(f" Datei erfolgreich geladen: {file_path}")
    print(f"📊 Anzahl Zeilen: {len(df):,}")
    print(f" Erster Eintrag: {df['timestamp'].min()}")
    print(f" Letzter Eintrag: {df['timestamp'].max()}")

    print("\nSo sehen die ersten 5 Zeilen aus:")
    print(df.head().to_string(index=False))

    print("Alle Spaltennamen:")
    print(df.columns.tolist())
    print("\n" + "-"*50 + "\n")

# Falls du das Skript aus einem anderen Ordner startest, nutze den absoluten Pfad oder passe die ../ an.
#if not os.path.exists(file_path_BTC, file_path_GLD, file_path_QQQ, file_path_UUP):
if not all(os.path.exists(p) for p in (file_path_BTC, file_path_GLD, file_path_QQQ, file_path_UUP)):
    print(" Datei nicht gefunden! Prüfe den Pfad.")
else:
    load_and_display_info(file_path_BTC)
    load_and_display_info(file_path_GLD)
    load_and_display_info(file_path_QQQ)
    load_and_display_info(file_path_UUP)
    load_and_display_info(file_path_M2)
    load_and_display_info(file_path_US_Interest_Rate)
