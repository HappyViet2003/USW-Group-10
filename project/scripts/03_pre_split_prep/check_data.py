import pandas as pd
import os

# Pfad zu deiner Parquet-Datei
# Tipp: In PyCharm Rechtsklick auf die Datei -> "Copy Path/Reference" -> "Absolute Path"
file_path_merged = "../../data/processed/training_data_final.parquet"

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
    print(df.isnull().sum())

# Falls du das Skript aus einem anderen Ordner startest, nutze den absoluten Pfad oder passe die ../ an.
#if not os.path.exists(file_path_BTC, file_path_GLD, file_path_QQQ, file_path_UUP):
if not all(os.path.exists(p) for p in (file_path_merged,)):
    print(" Datei nicht gefunden! Prüfe den Pfad.")
else:
    load_and_display_info(file_path_merged)