# Experiment 4: Market Regime Detection + Ensemble Model

## 🎯 Zielsetzung

Verbesserung der Vorhersagegenauigkeit durch:
- **Market Regime Detection** (Klassifikation in Bull/Bear/Sideways)
- **Ensemble Learning** (Kombination mehrerer Modelle)
- **Adaptive Trading Strategy** (regime-spezifische Schwellenwerte)

---

## 🔬 Methodik

### 1. Market Regime Detection

**Klassifikation in 3 Marktphasen:**

- **Bull Market (Aufwärtstrend):** MA-Steigung > 2%, niedrige Volatilität, Preis > MA200
- **Bear Market (Abwärtstrend):** MA-Steigung < -2%, hohe Volatilität, Preis < MA200
- **Sideways Market (Seitwärtsbewegung):** Alles andere

**Features für Regime Detection:**
- Moving Averages (MA20, MA50, MA200)
- MA-Steigung (Trendrichtung)
- Volatilität (20-Tage rollende Standardabweichung)
- Volumen-Trend

**Wichtig:** Nach dem Bugfix werden alle Indikatoren mit `.shift(1)` berechnet, um Look-Ahead Bias zu vermeiden.

### 2. Ensemble Model

**Kombination von 3 Modellen via Soft Voting:**

1. **XGBoost** (Gewichtung=2)
   - Tree-basiertes Gradient Boosting
   - Optimiert für strukturierte Daten
   - Beste Performance in Exp 1

2. **Random Forest** (Gewichtung=1)
   - Ensemble aus Decision Trees
   - Robustheit durch Averaging

3. **Logistic Regression** (Gewichtung=1)
   - Lineares Modell
   - Interpretierbarkeit

**Voting-Mechanismus:**
```python
VotingClassifier(
    estimators=[('xgb', xgb_model), ('rf', rf_model), ('lr', lr_model)],
    voting='soft',
    weights=[2, 1, 1]
)
```

### 3. Adaptive Trading Strategy

**Regime-spezifische Schwellenwerte:**

- **Bull Market:** Threshold = 0.55 (nur bei hoher Konfidenz Long-Position)
- **Bear Market:** Threshold = 0.50 (aggressivere Short-Positionen)
- **Sideways Market:** Threshold = 0.55 (Mean Reversion Strategy)

**Signal-Logik:**
- Long (+1): Wenn Vorhersage > Threshold
- Short (-1): Wenn Vorhersage < (1 - Threshold)
- Hold (0): Sonst kein Trade

---

## 📊 Ergebnisse

### Model-Performance

**Training Set:**
- Accuracy: 80,20%
- AUC: 0,8877

**Validation Set:**
- Accuracy: 52,53%
- AUC: 0,5114

**Test Set (2024):**
- Accuracy: 48,68%
- AUC: 0,5228

**Interpretation:** Starkes Overfitting auf Training Set, Test-Performance schlechter als Zufallsraten (50%).

### Vorhersage-Analyse

**Vorhersage-Statistiken:**
- Minimum: 0,131 (13,1%)
- Maximum: 0,206 (20,6%)
- Mittelwert: 0,176 (17,6%)

**Problem:** Model sagt nur Werte zwischen 13-21% vorher, keine echte Unterscheidung zwischen Up/Down!

### Regime-Verteilung (Test Set)

- **Sideways:** 45.631 Samples (91,8%)
- **Bear:** 2.155 Samples (4,3%)
- **Bull:** 1.954 Samples (3,9%)

### Backtest-Ergebnisse

**Performance-Metriken:**
- Startkapital: 10.000 $
- Endkapital: 5.333,59 $
- Gesamtrendite: **-46,66%**
- Buy & Hold Rendite: -9,50%
- Outperformance: **-37,16%** (schlechter als Buy & Hold!)

**Trading-Statistiken:**
- Gesamtanzahl Trades: 24.081
- Win Rate: 0,61% (nur 147 gewinnende Trades!)
- Durchschnittlicher Trade: -0,17%

**Signal-Verteilung:**
- Short-Signale: 47.786 (96,1%)
- Hold-Signale: 1.954 (3,9%)
- Long-Signale: 0 (0,0%)

**Problem:** Strategy führt nur Short-Trades aus, weil alle Vorhersagen < 50% sind!

---

## 🔍 Analyse & Learnings

### Warum funktioniert es nicht?

1. **Model lernt nichts:** Nach Bugfix sagt das Model nur 13-21% vorher (keine Varianz)
2. **Features zu schwach:** Technische Indikatoren + On-Chain Daten reichen nicht aus
3. **4-Stunden-Horizont zu kurz:** Zu viel Rauschen, zu wenig Signal
4. **Overfitting:** Training Accuracy 80%, Test Accuracy 49%

### Warum nur Short-Trades?

**Ursache:** Model sagt nur 13-21% vorher (immer "Down")

**Konsequenz:**
- Bull Markets: `prob > 0,55` nie erfüllt → kein Trade
- Bear Markets: `prob < 0,50` immer erfüllt → Short
- Sideways: `prob < 0,45` immer erfüllt → Short

### Was würde helfen?

1. **Mehr/bessere Features:** Sentiment-Analyse, News, Social Media
2. **Längerer Horizont:** 24h statt 4h (weniger Rauschen)
3. **Komplexeres Model:** Transformer, LSTM mit mehr Layern
4. **Mehr Daten:** Mehrere Jahre Trainingsdaten

---

## 💡 Wissenschaftliche Erkenntnisse

### Positive Aspekte

1. **Sophistizierter Ansatz:** Market Regime Detection ist ein valider Forschungsansatz
2. **Ensemble Learning:** State-of-the-art Methode für robuste Vorhersagen
3. **Bugfixing:** Zeigt kritisches Denken und wissenschaftliche Integrität
4. **Ehrliche Ergebnisse:** Keine aufgeblähte Performance durch Bugs

### Negative Aspekte

1. **Schlechte Performance:** -46% Rendite, nur Short-Trades
2. **Model lernt nichts:** Vorhersagen ohne Varianz (13-21%)
3. **Overfitting:** Training 80%, Test 49%
4. **Nicht handelbar:** Strategy ist in der Praxis nicht verwendbar

### Hauptlearning

**Nach dem Bugfix zeigt sich:** Die ursprünglichen "guten" Ergebnisse (+267%) waren nur durch Bugs möglich. Die echte Performance ist katastrophal.

**Das demonstriert:** Bitcoin-Preisprognosen auf 4-Stunden-Basis mit einfachen Features sind extrem schwierig, selbst mit sophistizierten ML-Methoden.

---

## 📈 Vergleich mit anderen Experimenten

| Experiment | Accuracy | Rendite | Trades |
|------------|----------|---------|--------|
| Exp 1: Baseline (XGBoost) | 52,5% | N/A | N/A |
| Exp 2: On-Chain Data | 52,8% | N/A | N/A |
| Exp 3: LSTM | 51,2% | N/A | N/A |
| **Exp 4: Ensemble + Regime** | **48,7%** | **-46,66%** | **24.081** |

**Ergebnis:** Exp 4 ist schlechter als alle vorherigen Experimente!

---

## 🎓 Fazit

**Experiment 4 zeigt:**
- Market Regime Detection + Ensemble Learning sind valide Ansätze
- Nach Bugfix funktioniert das Model nicht mehr
- Die ursprünglichen Ergebnisse waren durch Look-Ahead Bias verfälscht
- Bitcoin-Prognosen sind extrem schwierig

**Wissenschaftlicher Wert:**
- ✅ Sophistizierte Methodik
- ✅ Bugfixing & Code Review
- ✅ Ehrliche Ergebnisse
- ✅ Kritische Reflexion

**Praktischer Wert:**
- ❌ Nicht handelbar
- ❌ Schlechter als Buy & Hold
- ❌ Model lernt nichts

**Empfehlung:** Präsentiere die Bugs, die Fixes und die ehrlichen Ergebnisse. Das zeigt wissenschaftliche Integrität und ist wertvoller als gefälschte Resultate!
