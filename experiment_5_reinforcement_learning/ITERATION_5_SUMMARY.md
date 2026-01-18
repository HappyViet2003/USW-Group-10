# Iteration 5: Reinforcement Learning (DQN) Agent

**Status:** ✅ Abgeschlossen
**Datum:** 18.01.2026
**Verantwortlich:** USW-Group-10

---

## 1. Zielsetzung
Das Ziel von Experiment 5 war die Entwicklung eines autonomen Trading-Agenten mittels Deep Reinforcement Learning (DQN). Im Gegensatz zu den vorherigen Modellen (Exp 1-4), die versuchten, Preise vorherzusagen, sollte dieser Agent lernen, eine profitable Handelsstrategie (Policy) unter Berücksichtigung von Transaktionsgebühren und Risiken selbstständig zu entwickeln.

---

## 2. Methodik

### A. Setup
* **Algorithmus:** Deep Q-Network (DQN) mit MlpPolicy (Stable-Baselines3).
* **Environment:** Custom `BitcoinTradingEnv` (OpenAI Gym Interface).
* **Features:** Technische Indikatoren (RSI, MACD) + On-Chain Daten.
* **Reward Function:** Log-Returns mit expliziten Strafen für hohe Volatilität und Drawdowns ("Risk Aversion").

### B. Realismus-Check
Um Datenfehler (Look-Ahead Bias durch Skalierung) auszuschließen, wurde der Backtest in zwei Phasen getrennt:
1.  **Input:** Skalierte Features (Z-Scores) für das neuronale Netz.
2.  **Simulation:** Echte US-Dollar Preise für die Berechnung des Portfoliowerts.

---

## 3. Ergebnisse (Backtest)

| Metrik | Wert | Interpretation |
| :--- | :--- | :--- |
| **Startkapital** | $100,000.00 | - |
| **Endkapital** | **$100,729.27** | **Kapitalerhalt erfolgreich (+0.73%)**. |
| **Buy & Hold** | +155.85% | Starker Bullenmarkt im Testzeitraum. |
| **Outperformance** | -155.12% | Der Agent partizipierte nicht an der Rallye. |

### Verhaltensanalyse
* **Strategie:** "Extreme Risk Aversion".
* **Ablauf:** Der Agent erwirtschaftete in der ersten Phase kleine Gewinne (Balance stieg auf ~$102.300).
* **Der "Ausstieg":** Nach einer volatilen Phase (ca. Step 160.000) verkaufte der Agent seine Positionen und wechselte dauerhaft in **Cash**.
* **Grund:** Die "Risk Penalties" im Training waren vermutlich zu hoch gewichtet. Der Agent lernte, dass Nicht-Teilnahme (Cash) sicherer ist als die wilden Schwankungen des Krypto-Marktes.

---

## 4. Fazit & Learnings

1.  **Erfolg im Risikomanagement:** Im Gegensatz zu Experiment 4 (Verlust von -46%) hat der RL-Agent kein Geld verloren. Er hat bewiesen, dass er "überleben" kann.
2.  **Das "Safe Haven" Problem:** Der Agent fand ein lokales Optimum: *Geringer Gewinn + 100% Sicherheit > Hoher möglicher Gewinn + Risiko.* Für einen konservativen Fonds wäre das ideal, für einen Krypto-Trader ist es zu defensiv.
3.  **Ausblick:** Für zukünftige Iterationen müsste die Belohnungsfunktion angepasst werden ("Reward Shaping"), um das "Halten" während Aufwärtstrends stärker zu belohnen als das reine Vermeiden von Verlusten.

**Gesamtergebnis:** Experiment 5 liefert den stabilsten und sichersten Ansatz aller Experimente, auch wenn er in diesem Szenario die Marktchancen nicht aggressiv genug nutzte.