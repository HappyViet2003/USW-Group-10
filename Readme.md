# USW-Group-10: Bitcoin High-Frequency Trading Bot

## Members
- Viet Anh Hönemann (Matrikelnummer: S0587778)
- Julius Bollmann (Matrikelnummer: S0594551)


# Experiments
- [Exp_1.md](experiment_1/README.md)
- [Exp_2.md](experiment_2_onchain/ITERATION_2_SUMMARY.md)
- [Exp_3.md](experiment_3_lstm/ITERATION_3_SUMMARY.md)
- [Exp_4.md](experiment_4_regime_ensemble/ITERATION_4_SUMMARY.md)
- [Exp_5.md](experiment_5_reinforcement_learning/ITERATION_5_SUMMARY.md)
---

## 📖 Projektübersicht & Wissenschaftliche Erkenntnisse
Dieses Projekt dokumentiert eine wissenschaftliche Untersuchung zum Hochfrequenzhandel (HFT) von Bitcoin mittels Machine Learning. Anstatt nach unrealistischen "Black Box"-Gewinnen zu streben, folgten wir einem rigorosen, iterativen Prozess, um den **"Reality Gap"** zwischen Modellgenauigkeit und tatsächlicher Handelsprofitabilität zu untersuchen.

Wir haben **5 Hauptexperimente** durchgeführt, angefangen bei klassischem Supervised Learning (XGBoost) über Deep Learning (LSTM) bis hin zu Reinforcement Learning (DQN).

### Kern-Erkenntnisse (Key Findings)
1.  **Reality Gap:** Eine Vorhersagegenauigkeit von 52% reicht nicht aus, um die Handelsgebühren (0,1% pro Trade) zu decken.
2.  **Feature Engineering:** Das Hinzufügen von Hunderten technischen Indikatoren erzeugt oft nur "Rauschen". Weniger ist mehr ("Scorched Earth Policy").
3.  **Reinforcement Learning:** Der RL-Agent lernte **Kapitalerhalt** (Überleben) statt aggressivem Glücksspiel. In einem hochvolatilen Markt ist dies ein signifikantes Ergebnis.
4.  **Wissenschaftliche Integrität:** Wir dokumentieren explizit auch fehlgeschlagene Ansätze (z.B. Exp 3 & 4), um die Schwierigkeit der Domäne aufzuzeigen, anstatt Ergebnisse schönzurechnen ("Cherry-Picking").

---

## 🔬 Experimente & Ergebnisse

| Experiment | Methode | Fokus / Innovation | Ergebnis & Erkenntnis | Link |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | **XGBoost (Baseline)** | 80+ Technische Indikatoren | **Accuracy: 52,06%**<br>Backtest-Verlust: -65%. Bestätigte die extreme Schwierigkeit von 1h-Vorhersagen. | [Details ansehen](experiment_1/README.md) |
| **Exp 2** | **On-Chain Daten** | "Scorched Earth" (Keine Preis-Daten) | **Bestes Alpha.**<br>Nachweis eines statistischen Vorteils durch Blockchain-Fundamentaldaten, jedoch fraßen die Gebühren den Gewinn auf. | [Details ansehen](experiment_2_onchain/ITERATION_2_SUMMARY.md) |
| **Exp 3** | **Deep Learning (LSTM)** | Zeitreihen-Analyse | **Abgebrochen.**<br>Starkes Overfitting. Bewies, dass auf verrauschten 1-Min-Daten einfache Modelle besser generalisieren als komplexe Netze. | [Details ansehen](experiment_3_lstm/ITERATION_3_SUMMARY.md) |
| **Exp 4** | **Ensemble + Regime** | Marktphasen-Erkennung | **Scientific Honesty.**<br>Nach dem Fixen eines kritischen Look-Ahead-Bugs fiel die Performance. Wir berichten ehrlich über das negative Resultat. | [Details ansehen](experiment_4_regime_ensemble/ITERATION_4_SUMMARY.md) |
| **Exp 5** | **Reinforcement Learning** | DQN Agent (Selbstlernend) | **Kapitalerhalt.**<br>Der Agent erzielte +0,73% Gewinn und minimierte das Risiko ("Survival Mode"). Robustestes Verhalten aller Tests. | [Details ansehen](experiment_5_reinforcement_learning/ITERATION_5_SUMMARY.md) |

---

## 🛠 Tech Stack
- **Core:** Python 3.10+, Pandas, NumPy
- **ML/DL:** XGBoost, Scikit-Learn, TensorFlow/Keras
- **Reinforcement Learning:** Stable-Baselines3 (DQN), OpenAI Gym Custom Env
- **Beschleunigung:** CUDA / GPU Support für Training
- **Daten:** 1-Min OHLCV Daten + Blockchain.com On-Chain Metriken

---

## 📊 Fazit
Unsere Forschung zeigt, dass profitabler Hochfrequenzhandel mit Bitcoin mehr erfordert als reine Preisvorhersage. Während **Experiment 2** die beste Signalqualität lieferte ("Alpha"), zeigte **Experiment 5 (RL)** das robusteste Risikomanagement.

Das Projekt dient als transparente Dokumentation der Herausforderungen im Quant-Trading: **Gebühren, Rauschen (Noise) und Overfitting sind die Gegner.**