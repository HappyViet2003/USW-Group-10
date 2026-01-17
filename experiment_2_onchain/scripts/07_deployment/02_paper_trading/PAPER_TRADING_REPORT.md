# Paper Trading Report: Experiment 2 (On-Chain)

**Datum:** 17.01.2026
**Status:** ✅ Simulation Erfolgreich
**Modell:** `xgboost_onchain.json`
**Features:** Market Data + On-Chain Fundamentals (Hashrate, Active Addresses)

---

## 1. System Health Check
Der Live-Demonstrator (`run_paper_trading.py`) bestätigt, dass alle Module korrekt interagieren:

- **Modell-Loading:** ✅ Erfolgreich (`xgboost_onchain.json` geladen).
- **Data Acquisition:** ✅ Simuliert (Mock API liefert Preise um $95k).
- **On-Chain Integration:** ✅ Simuliert (Hashrate-Daten werden in die Pipeline eingespeist).
- **Inference Speed:** < 0.1s pro Tick (Echtzeit-fähig).

---

## 2. Simulation Log (Auszug)

Der Bot wurde über 3 Ticks (Markt-Updates) beobachtet. Hierbei wurde geprüft, ob das Modell auf fundamentale Daten reagiert.

| Tick | Preis (BTC) | On-Chain Signal | Model Confidence | Entscheidung |
| :--- | :--- | :--- | :--- | :--- |
| **1** | $94,957 | Hashrate: -4.22% (Bearish) | 55.13% | 💤 **HOLD** (Neutral) |
| **2** | $95,174 | Hashrate: +3.09% (Bullish) | 59.49% | 💤 **HOLD** (Neutral) |
| **3** | $94,838 | Hashrate: -2.62% (Bearish) | 40.00% | 💤 **HOLD** (Neutral) |

**Beobachtung:**
Das Modell agiert sehr selektiv. Trotz eines Bullish-Signals der Hashrate in Tick 2 (+3.09%) blieb die Confidence (59.49%) knapp unter dem Trading-Threshold von 0.60. Dies zeigt, dass das Modell **nicht blind** jedem On-Chain-Signal folgt, sondern Bestätigung durch andere Faktoren (Preis, Volumen) sucht.

---

## 3. Analyse & Fazit

### Robustheit
Die Integration der On-Chain-Daten funktioniert technisch einwandfrei. Der Bot ist in der Lage, fundamentale Metriken (wie Hashrate-Einbrüche) in Echtzeit zu verarbeiten und in seine Entscheidung einzubeziehen.

### Deployment-Empfehlung: 🛑 NO GO
Basierend auf dem Backtest (siehe `backtest_results.csv` mit -27% Return) und diesem Paper-Trading-Lauf, ist das System **nicht bereit für Echtgeld-Einsatz**.

**Gründe:**
1.  **Gebühren-Falle:** Der erwartete Gewinn pro Trade ist kleiner als die Kosten (Spread + Fees).
2.  **Signal-Qualität:** Die Signale sind stabil, aber der "Edge" ist auf dem 1-Minuten-Zeitrahmen zu klein.

**Nächste Schritte:**
Übergang zu **Experiment 3 (LSTM)**, um komplexere zeitliche Muster zu erkennen und den Zeithorizont zu erweitern (Stunden statt Minuten), damit die Gewinne die Gebühren übersteigen.