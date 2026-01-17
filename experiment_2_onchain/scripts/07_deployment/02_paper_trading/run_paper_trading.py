"""
07_deployment/02_paper_trading/run_paper_trading.py

Experiment 2:
Demonstrator für den Live-Trading-Loop inkl. On-Chain Check.
Simuliert den Prozess:
1. Daten abrufen (Live API + Blockchain.com)
2. Features berechnen (Pipeline)
3. Vorhersage treffen (XGBoost On-Chain Modell)
4. Order ausführen (Alpaca API)
"""

import time
import os
import json
import random
import pandas as pd
import xgboost as xgb
import yaml
import numpy as np

# ==============================================================================
# KONFIGURATION
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
# WICHTIG: Hier laden wir das NEUE Modell aus Experiment 2
model_path = os.path.join(script_dir, "../../../data/models/xgboost_onchain.json")
scaler_path = os.path.join(script_dir, "../../../data/models/scaler.pkl")

print("=" * 70)
print("PAPER TRADING BOT (EXP 2: ON-CHAIN ENABLED)")
print("=" * 70)

# ==============================================================================
# MOCK KLASSEN
# ==============================================================================

class MockAlpacaAPI:
    """Simuliert die Verbindung zum Broker."""
    def get_latest_bar(self, symbol):
        print(f"   📡 API: Rufe Live-Preise für {symbol} ab...")
        price = 95000 + random.uniform(-200, 200)
        return {'timestamp': pd.Timestamp.now(), 'close': price, 'volume': 500}

    def submit_order(self, side, qty):
        print(f"   💸 ORDER EXECUTION: {side.upper()} {qty} BTC @ MARKET")

class MockBlockchainAPI:
    """Simuliert den Abruf von On-Chain Daten."""
    def get_network_stats(self):
        print(f"   🔗 ON-CHAIN: Prüfe Hashrate & Active Addresses...")
        # Simuliert eine Änderung der Hashrate
        change = random.uniform(-0.05, 0.05)
        return {'hashrate_change_24h': change}

class FeatureEngineer:
    """Simuliert die Feature-Berechnung."""
    def calculate_features(self, market_data, onchain_data):
        print("   ⚙️  Processing: Merge Market Data + On-Chain Data...")
        print(f"       -> Hashrate Trend: {onchain_data['hashrate_change_24h']:.2%}")
        # Dummy Features für das Modell (muss zur Shape passen)
        # Das Modell erwartet ca 14 Features (laut deinem letzten Training)
        # Wir erzeugen hier einfach genug Zufallszahlen, damit XGBoost nicht meckert
        return pd.DataFrame([np.random.rand(30)], columns=[f'feat_{i}' for i in range(30)])

# ==============================================================================
# MAIN TRADING LOOP
# ==============================================================================

def main():
    # 1. Setup
    print("[INIT] Lade On-Chain Modell...")
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        print("   ✅ Modell 'xgboost_onchain.json' geladen.")
    else:
        print(f"   ⚠️  Modell nicht gefunden: {model_path}")
        print("       (Simulation läuft im Demo-Modus weiter)")
        model = None

    broker = MockAlpacaAPI()
    chain_api = MockBlockchainAPI()
    engineer = FeatureEngineer()

    print("\n[START] Starte Trading Loop (Drücke Ctrl+C zum Beenden)...")

    try:
        # Endlosschleife (Simulation von 3 Runden)
        for i in range(1, 4):
            print(f"\n--- ⏱️  Tick {i} ---")

            # A. FETCH
            market_data = broker.get_latest_bar("BTC/USD")
            onchain_data = chain_api.get_network_stats()

            print(f"   Preis: ${market_data['close']:.2f}")

            # B. PROCESS
            features = engineer.calculate_features(market_data, onchain_data)

            # C. PREDICT
            probability = random.uniform(0.4, 0.7)
            print(f"   🤖 Model Prediction: {probability:.4f} (Prob Long)")

            # D. EXECUTE
            THRESHOLD = 0.60 # Etwas konservativer für Exp 2
            if probability > THRESHOLD:
                print("   ✅ SIGNAL: BUY (Strong Fundamentals).")
                broker.submit_order(side="buy", qty=0.1)
            elif probability < (1 - THRESHOLD):
                print("   🔻 SIGNAL: SELL (Weak Fundamentals).")
                broker.submit_order(side="sell", qty=0.1)
            else:
                print("   💤 SIGNAL: Neutral. Warten auf besseres Setup.")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n[STOP] Bot beendet.")

if __name__ == "__main__":
    main()