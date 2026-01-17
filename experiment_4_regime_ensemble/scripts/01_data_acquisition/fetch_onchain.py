"""
01_data_acquisition/fetch_onchain.py

Holt fundamentale Blockchain-Daten von Blockchain.com (Public API).
Metriken:
1. Hashrate (Sicherheit/Mining Power)
2. Unique Addresses (Nutzer-Aktivität)
"""

import pandas as pd
import requests
import os
import time
import yaml

def fetch_blockchain_data(chart_name, file_name):
    print(f"   Fetching {chart_name}...")
    # Blockchain.com API URL
    url = f"https://api.blockchain.info/charts/{chart_name}?timespan=5years&rollingAverage=8hours&format=json"

    try:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"❌ Error {r.status_code}: {r.text}")
            return

        data = r.json()

        # Daten parsen
        values = data['values']
        df = pd.DataFrame(values)
        df.columns = ['timestamp', chart_name]

        # --- FIX: UTC=True ---
        # Wir setzen die Zeitzone direkt beim Erstellen auf UTC.
        # Damit passen sie perfekt zu den Alpaca-Daten im Merge-Schritt.
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

        # Duplikate entfernen und sortieren
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        # Speichern vorbereiten
        script_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(script_dir, "../../conf/params.yaml")
        params = yaml.safe_load(open(params_path))

        # Pfad: data/external_data (wie in deinem merge script definiert)
        base_path = params['DATA_ACQUISITON']['DATA_PATH']
        output_dir = os.path.join(script_dir, base_path, 'external_data')
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{file_name}.parquet")
        df.to_parquet(output_path)
        print(f"✅ Saved to {output_path} ({len(df)} rows)")

    except Exception as e:
        print(f"❌ Exception fetching {chart_name}: {e}")

if __name__ == "__main__":
    print("="*70)
    print("FETCHING ON-CHAIN DATA (UTC FIXED)")
    print("="*70)

    fetch_blockchain_data("hash-rate", "onchain_hashrate")
    fetch_blockchain_data("n-unique-addresses", "onchain_active_addresses")