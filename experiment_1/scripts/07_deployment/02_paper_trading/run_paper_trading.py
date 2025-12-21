"""
07_deployment/02_paper_trading/run_paper_trading.py

Demonstrator für den Live-Trading-Loop ("Paper Trading").
Simuliert den Prozess:
1. Daten abrufen (Live API)
2. Features berechnen (Pipeline)
3. Vorhersage treffen (XGBoost)
4. Order ausführen (Alpaca API)

Hinweis: Dies ist ein Simulations-Skript für die Präsentation/Abgabe.
"""

import time
import os
import json
import random
import pandas as pd
import xgboost as xgb
import yaml

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
# Modell laden
model_path = os.path.join(script_dir, "../../../data/models/xgboost_final.json")
scaler_path = os.path.join(script_dir, "../../../data/models/scaler.pkl")

print("=" * 70)
print("PAPER TRADING BOT (SIMULATION MODE)")
print("=" * 70)


# ==============================================================================
# MOCK KLASSEN (Simulieren die echten APIs)
# ==============================================================================

class MockAlpacaAPI:
    """Simuliert die Verbindung zum Broker."""

    def get_latest_bar(self, symbol):
        # Simuliert Live-Datenabruf
        print(f"   📡 API: Rufe Live-Daten für {symbol} ab...")
        price = 95000 + random.uniform(-100, 100)
        return {'timestamp': pd.Timestamp.now(), 'close': price, 'volume': 500}

    def submit_order(self, side, qty):
        print(f"   💸 ORDER EXECUTION: {side.upper()} {qty} BTC @ MARKET")


class FeatureEngineer:
    """Simuliert die Feature-Berechnung in Echtzeit."""

    def calculate_features(self, raw_data):
        print("   ⚙️  Processing: Berechne Indikatoren (RSI, VWAP, Beta)...")
        # Hier würde 'features.py' Logik stehen
        # Wir geben Dummy-Features zurück, die das Modell erwartet
        return pd.DataFrame([np.random.rand(60)], columns=[f'feat_{i}' for i in range(60)])

    # ==============================================================================


# MAIN TRADING LOOP
# ==============================================================================

def main():
    # 1. Setup
    print("[INIT] Lade Modell...")
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        print("   ✅ Modell geladen.")
    else:
        print("   ⚠️  Modell nicht gefunden (Simulation läuft trotzdem).")
        model = None

    broker = MockAlpacaAPI()
    engineer = FeatureEngineer()

    print("\n[START] Starte Trading Loop (Drücke Ctrl+C zum Beenden)...")

    try:
        # Endlosschleife (Simulation von 3 Runden)
        for i in range(1, 4):
            print(f"\n--- ⏱️  Tick {i} ---")

            # A. FETCH
            current_data = broker.get_latest_bar("BTC/USD")
            print(f"   Preis: ${current_data['close']:.2f}")

            # B. PROCESS
            # In der Realität würden wir hier die letzten 1440 Minuten laden
            features = engineer.calculate_features(current_data)

            # C. PREDICT
            # Simuliere eine Vorhersage
            probability = random.uniform(0.4, 0.7)
            print(f"   🤖 Model Prediction: {probability:.4f} (Prob Long)")

            # D. EXECUTE
            THRESHOLD = 0.55
            if probability > THRESHOLD:
                print("   ✅ SIGNAL: STRONG BUY detected.")
                broker.submit_order(side="buy", qty=0.1)
            elif probability < (1 - THRESHOLD):
                print("   🔻 SIGNAL: STRONG SELL detected.")
                broker.submit_order(side="sell", qty=0.1)
            else:
                print("   💤 SIGNAL: Neutral. Halte Position.")

            # Wartezeit simulieren (z.B. 1 Stunde)
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n[STOP] Bot beendet.")


if __name__ == "__main__":
    import numpy as np  # Import hier für die Mock-Daten

    main()