import pandas as pd
import os

# Pfad zu deiner Parquet-Datei
# Tipp: In PyCharm Rechtsklick auf die Datei -> "Copy Path/Reference" -> "Absolute Path"
file_path = "../../data/Bars_1m_crypto/BTC_USD.parquet"

# Falls du das Skript aus einem anderen Ordner startest, nutze den absoluten Pfad oder passe die ../ an.
if not os.path.exists(file_path):
    print("❌ Datei nicht gefunden! Prüfe den Pfad.")
else:
    # Datei laden
    df = pd.read_parquet(file_path)

    print("✅ Datei erfolgreich geladen!")
    print(f"📊 Anzahl Zeilen: {len(df):,}")  # Zeigt z.B. 2,600,000
    print(f"📅 Erster Eintrag: {df['timestamp'].min()}")
    print(f"📅 Letzter Eintrag: {df['timestamp'].max()}")

    print("\nSo sehen die ersten 5 Zeilen aus:")
    print(df.head().to_string(index=False))

    # Füge das in check_data.py ein:
    print("Alle Spaltennamen:")
    print(df.columns.tolist())