# Iteration 3: Deep Learning (LSTM) & Zeitreihen-Analyse

**Datum:** 17.01.2026
**Verantwortlich:** USW-Group-10
**Status:** ✅ Abgeschlossen (Ergebnis: Overfitting / Keine Generalisierung)

---

## 1. Wissenschaftliche Zielsetzung
Nachdem Iteration 2 (XGBoost) solide Ergebnisse (51.73%) lieferte, aber zeitliche Zusammenhänge ignorierte ("Snapshot-Analyse"), war das Ziel von Iteration 3:
1.  **Sequenz-Analyse:** Nutzung eines **Long Short-Term Memory (LSTM)** Netzwerks, um den *Verlauf* der letzten 60 Minuten als zusammenhängenden "Film" zu analysieren.
2.  **Mustererkennung:** Identifikation komplexer, nicht-linearer zeitlicher Muster (z.B. Trends, Umkehrformationen), die Feature-basierten Modellen entgehen.

---

## 2. Methodik & Setup

### A. Daten-Struktur (3D-Tensoren)
Anders als bei XGBoost (Tabellenform) wurden die Daten in 3D-Blöcke umgewandelt:
* **Shape:** `(Samples, Timesteps, Features)`
* **Timesteps:** 60 (Rückblick auf die letzte Stunde).
* **Features:** 14 (Identisch zu Exp 2, inkl. On-Chain-Daten).

### B. Modell-Architektur
Ein Deep Learning Modell mit Keras/TensorFlow:
* **Layer 1:** LSTM (64 Units, Return Sequences=True) – Extraktion feiner Zeit-Details.
* **Layer 2:** LSTM (32 Units) – Verdichtung der Information.
* **Regularisierung:** Dropout (0.2) nach jedem LSTM-Layer, um Overfitting zu reduzieren.
* **Output:** Sigmoid-Aktivierung (Wahrscheinlichkeit 0-1).

---

## 3. Ergebnisse

### Modell-Performance
Das Training zeigte eine klassische Diskrepanz zwischen Lernen und Verstehen:

| Metrik | Training (Epoch 4) | Test-Set (Unbekannte Daten) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **64.97%** | **49.84%** | Extremes Overfitting |
| **AUC** | 0.7093 | 0.5069 | Keine Generalisierung |

**Analyse:**
* Das Modell war in der Lage, den Trainingsdatensatz extrem gut auswendig zu lernen (Accuracy stieg schnell auf >60%).
* Auf dem Test-Set fiel die Leistung jedoch auf das Niveau von Raten (ca. 50%) zurück.
* **Ursache:** Das "Signal-zu-Rauschen"-Verhältnis (Signal-to-Noise Ratio) bei 1-Minuten-Bitcoin-Daten ist sehr schlecht. Das komplexe LSTM-Netzwerk hat begonnen, das zufällige Rauschen ("Noise") als Muster zu interpretieren, statt echte Marktdynamiken zu lernen.

---

## 4. Konklusion & Entscheidung

**Wissenschaftliche Erkenntnis:**
Komplexität ist im Finanzmarkt nicht gleichbedeutend mit Erfolg.
1.  **Feature-Engineering > Deep Learning:** Einfachere Modelle mit starken Features (XGBoost + On-Chain in Exp 2) sind robuster gegen Rauschen als komplexe neuronale Netze (LSTM in Exp 3).
2.  **Zeithorizont:** 1-Minuten-Daten sind für Deep Learning zu chaotisch. LSTMs würden vermutlich auf 4-Stunden- oder Tages-Daten besser funktionieren.

**Operative Entscheidung (Early Termination):**
Aufgrund der mangelnden Generalisierung (Test Accuracy < 50%) wurde das Experiment **vor dem Deployment abgebrochen**.
* **Kein Backtest:** Ein Handelssystem, das im Test schlechter als der Zufall ist, führt unter Berücksichtigung von Gebühren garantiert zu Verlusten.
* **Kein Paper Trading:** Ressourcenschonung durch Abbruch der Pipeline nach der Modellierung.

---

## 5. Fazit für das Gesamtprojekt
Der Vergleich zeigt deutlich: **Experiment 2 (XGBoost mit On-Chain Daten)** ist der Gewinner. Es liefert stabilen Mehrwert ("Alpha"), während Experiment 3 die Grenzen der Modellierung aufzeigt.