import requests
import pandas as pd
import os


def fetch_fear_greed():
    print("   Lade Fear & Greed Index...")
    url = "https://api.alternative.me/fng/?limit=0&format=json"  # limit=0 holt die gesamte Historie

    try:
        r = requests.get(url)
        r.raise_for_status()  # Check auf HTTP-Fehler
        data = r.json()['data']

        df = pd.DataFrame(data)

        # --- BUGFIX: Erst in Zahlen umwandeln, dann in Datum ---
        # Die API sendet Strings ("160000"), Pandas will Zahlen für unit='s'
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Werte aufräumen
        df['value'] = df['value'].astype(int)
        df['value_classification'] = df['value_classification'].astype(str)
        df = df[['timestamp', 'value', 'value_classification']].rename(columns={'value': 'fear_greed'})


        # Zeitzone auf UTC setzen (wichtig für Merge!)
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

        # Speichern
        # Wir gehen vom Skript-Pfad aus, um den Speicherort sicher zu finden
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "../../data/external_data/FEAR_GREED.parquet")

        # Ordner erstellen falls nicht existent
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_parquet(output_path, index=False)
        print(f"✅ Fear & Greed Index gespeichert: {output_path} ({len(df)} Zeilen)")

    except Exception as e:
        print(f"❌ Fehler beim Laden von Fear & Greed: {e}")


if __name__ == "__main__":
    fetch_fear_greed()