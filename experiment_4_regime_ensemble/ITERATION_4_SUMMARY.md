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

- **Bull Market:** Threshold = 0.65 (nur bei sehr hoher Konfidenz Long-Position)
- **Bear Market:** Threshold = 0.40 (konservative Short-Positionen)
- **Sideways Market:** Threshold = 0.65 (Mean Reversion Strategy)

**Signal-Logik:**
- Long (+1): Wenn Vorhersage > Threshold
- Short (-1): Wenn Vorhersage < (1 - Threshold)
- Hold (0): Sonst kein Trade

**Wichtig:** Nach mehreren Iterationen wurden die Thresholds erhöht, um die Anzahl der Trades zu reduzieren und Trading Fees zu minimieren.

---

## 📊 Ergebnisse

### Model-Performance

**Training Set:**
- Accuracy: ~52-53%
- Nach Bugfix: Keine Überanpassung mehr

**Test Set (2024):**
- Accuracy: ~48-52%
- Vorhersage-Statistiken:
  - Minimum: 0.236 (23.6%)
  - Maximum: 0.682 (68.2%)
  - Mittelwert: 0.455 (45.5%)

**Interpretation:** Nach dem Bugfix zeigt das Model eine breitere Verteilung der Vorhersagen (24-68% statt 13-21%), aber immer noch keine starke Predictive Power.

### Regime-Verteilung (Test Set 2024)

**Problem:** Die Regime Detection ist fundamental fehlerhaft!

- **Sideways:** 313.811 Samples (99.9%)
- **Bear:** 208 Samples (0.07%)
- **Bull:** 1 Sample (0.0003%)

**Analyse:** Obwohl Bitcoin 2024 von $34k auf $108k stieg (+215%), erkennt das System fast nur Sideways Markets. Die Bedingungen für Bull/Bear Markets sind zu streng und werden fast nie erfüllt.

### Backtest-Ergebnisse (FINAL VERSION)

**Performance-Metriken:**
- Startkapital: $10.000
- Endkapital: **$0.00**
- Gesamtrendite: **-100.00%** (Totalverlust!)
- Buy & Hold Rendite: **+120.92%** (Bitcoin 2024: $34k → $76k)
- Outperformance: **-220.92%** (katastrophal schlechter als Buy & Hold!)

**Trading-Statistiken:**
- Gesamtanzahl Trades: **23.839**
- Win Rate: **1.21%** (nur 288 von 23.839 Trades gewinnen!)
- Durchschnittlicher Trade: **-0.20%**

**Signal-Verteilung:**
- Hold-Signale: 280.431 (89.3%)
- Short-Signale: 33.589 (10.7%)
- Long-Signale: 0 (0.0%)

**Kritische Analyse:**

1. **Zu viele Trades:** 23.839 Trades in 314.020 Samples = alle 13 Samples ein Trade!
2. **Trading Fees fressen alles auf:** 23.839 × 0.2% = **4.768% in Fees!**
3. **Katastrophale Win Rate:** 98.79% der Trades verlieren!
4. **Keine Long-Positionen:** Trotz 120% Bull Market nur Shorts und Holds!

---

## 🔍 Analyse & Learnings

### Warum -100% Totalverlust?

**Drei fundamentale Probleme:**

1. **Regime Detection ist kaputt**
   - 99.9% Sideways trotz starkem Bull Market
   - Die Bedingungen (MA-Steigung > 2%, niedrige Vol, Preis > MA200) sind zu streng
   - Fast nie werden Bull/Bear Markets erkannt

2. **Model hat keine Predictive Power**
   - Win Rate 1.21% = schlechter als Random Guessing!
   - Vorhersagen haben keine Korrelation mit echten Preisbewegungen

3. **Tod durch Trading Fees**
   - 23.839 Trades × 0.2% Fees = 4.768% Kapitalverlust durch Fees alleine
   - Jeder Trade verliert im Schnitt 0.20% + 0.2% Fees = 0.40%
   - Nach 250 Trades ist das Kapital weg (250 × 0.40% = 100%)

### Warum keine Long-Trades?

**Ursache:** Kombination aus fehlerhafter Regime Detection und zu hohen Thresholds

- **99.9% Sideways Markets** → Sideways-Strategy wird fast immer verwendet
- **Threshold 0.65** → Nur Long wenn Prediction > 65%
- **Mean Prediction 45.5%** → Fast nie über 65%
- **Resultat:** Keine Long-Trades, nur Shorts und Holds

### Was würde helfen?

**Kurzfristige Fixes (nicht mehr umsetzbar):**
1. **Regime Detection vereinfachen:** Nur MA-Steigung verwenden (ohne Volatilität & MA200)
2. **Thresholds drastisch erhöhen:** 0.70/0.30/0.70 → nur bei extremer Konfidenz traden
3. **Trading Fees erhöhen:** Realistische 0.5% statt 0.1%

**Langfristige Lösungen (würden Tage dauern):**
1. **Komplettes Redesign der Regime Detection**
2. **Mehr/bessere Features:** Sentiment, News, Social Media
3. **Längerer Horizont:** 24h statt 4h
4. **Komplexeres Model:** Transformer, LSTM mit mehr Layern

---

## 💡 Wissenschaftliche Erkenntnisse

### Positive Aspekte

1. **Sophistizierter Ansatz:** Market Regime Detection + Ensemble Learning sind state-of-the-art Methoden
2. **Umfangreiches Bugfixing:** Look-Ahead Bias, Always Long Bug, Skalierte Preise
3. **Iterative Verbesserung:** Mehrere Threshold-Anpassungen zur Reduzierung der Trades
4. **Wissenschaftliche Integrität:** Ehrliche Darstellung katastrophaler Ergebnisse

### Negative Aspekte

1. **Totalverlust:** -100% Rendite, schlechter als alle vorherigen Experimente
2. **Fundamentale Fehler:** Regime Detection erkennt 99.9% Sideways
3. **Keine Predictive Power:** Win Rate 1.21%
4. **Nicht handelbar:** Strategy ist in der Praxis völlig unbrauchbar

### Hauptlearning

**Die Ergebnisse demonstrieren:**

1. **Regime Detection ist extrem schwierig:** Einfache technische Indikatoren reichen nicht aus
2. **Trading Fees sind brutal:** Viele Trades = garantierter Verlust
3. **ML-basiertes Trading ist hart:** Selbst sophistizierte Methoden scheitern
4. **Wissenschaftliche Integrität > Fake Results:** Ehrliche negative Ergebnisse sind wertvoller als verfälschte positive

**Quote vom Dozenten:** *"Negative Ergebnisse sind OK, es geht um wissenschaftliche Korrektheit!"*

---

## 📈 Vergleich mit anderen Experimenten

| Experiment | Accuracy | Rendite | Trades | Status |
|------------|----------|---------|--------|--------|
| Exp 1: Baseline (XGBoost) | 52.5% | N/A | N/A | ✅ OK |
| Exp 2: On-Chain Data | 52.8% | N/A | N/A | ✅ OK |
| Exp 3: LSTM | 51.2% | N/A | N/A | ✅ OK |
| **Exp 4: Ensemble + Regime** | **48-52%** | **-100%** | **23.839** | ❌ **FAIL** |
| Exp 5: RL (DQN) | N/A | +0.73% | 0 | ✅ **BEST** |

**Ergebnis:** Exp 4 ist das schlechteste Experiment! Exp 5 (RL) ist das einzige profitable Experiment (+0.73%, kein Verlust).

---

## 🎓 Fazit

**Experiment 4 ist ein komplettes Scheitern, aber ein wertvolles Learning!**

### Was funktioniert nicht:

- ❌ Market Regime Detection mit technischen Indikatoren
- ❌ Adaptive Trading Strategy mit vielen Trades
- ❌ Ensemble Learning ohne gute Features
- ❌ 4-Stunden-Horizont für Bitcoin Trading

### Was wir gelernt haben:

- ✅ **Bugfixing ist essentiell:** Look-Ahead Bias, Always Long, Skalierte Preise
- ✅ **Trading Fees sind ein Killer:** Viele Trades = garantierter Verlust
- ✅ **Regime Detection ist schwer:** Einfache Regeln funktionieren nicht
- ✅ **Wissenschaftliche Integrität:** Ehrliche Ergebnisse > Fake Results

### Wissenschaftlicher Wert:

**Hoch!** Weil:
- Sophistizierte Methodik (Regime Detection, Ensemble, Adaptive Strategy)
- Umfangreiches Bugfixing und Code Review
- Iterative Verbesserungsversuche (Threshold-Anpassungen)
- Ehrliche Darstellung katastrophaler Ergebnisse
- Kritische Reflexion und Analyse

### Praktischer Wert:

**Null!** Weil:
- -100% Rendite (Totalverlust)
- Nicht handelbar in der Praxis
- Schlechter als Buy & Hold (-220% Outperformance)
- Schlechter als alle anderen Experimente

---

## 📝 Empfehlung für Präsentation

**Präsentiere Experiment 4 als "Negative Case Study":**

1. **Zeige die Bugs:** Look-Ahead Bias, Always Long, Skalierte Preise
2. **Zeige die Fixes:** Code-Änderungen, Threshold-Anpassungen
3. **Zeige die Ergebnisse:** -100%, 23.839 Trades, 1.21% Win Rate
4. **Erkläre die Probleme:** Regime Detection, Trading Fees, Predictive Power
5. **Zeige die Learnings:** Was funktioniert nicht, was haben wir gelernt

**Message:** *"Wir zeigen ehrliche Ergebnisse, nicht inflated Performance. Das ist wissenschaftliche Integrität!"*

**Vergleich mit Exp 5:** *"Experiment 5 (RL) zeigt, dass konservative Strategien (kein Trading) besser sind als aggressive Strategien (viele Trades)."*

---

## 🔗 Technische Details

### Bugs & Fixes

**Bug #1: Look-Ahead Bias im Regime Detection**
```python
# VORHER (Bug):
df['ma_slope'] = df['ma_20'].pct_change(20)  # Schaut in Zukunft!

# NACHHER (Fixed):
df['ma_20_lag'] = df['ma_20'].shift(1)  # Nur Vergangenheit
df['ma_slope'] = df['ma_20_lag'].pct_change(20)
```

**Bug #2: Always Long in Bull Markets**
```python
# VORHER (Bug):
if regime == 'bull':
    return 1  # Always Long! ❌

# NACHHER (Fixed):
if regime == 'bull':
    if prob > THRESHOLD_BULL:
        return 1  # Long
    else:
        return 0  # No position ✅
```

**Bug #3: Skalierte Preise im Backtest**
```python
# VORHER (Bug):
buy_hold_return = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100
# 'close' ist skaliert (Z-Score) → +1017% (FALSCH!)

# NACHHER (Fixed):
df['close_original'] = df_original['close'].values  # Original Preise laden
buy_hold_return = (df.iloc[-1]['close_original'] / df.iloc[0]['close_original'] - 1) * 100
# → +120.92% (RICHTIG!)
```

### Threshold-Iterationen

| Version | Bull | Bear | Sideways | Trades | Return |
|---------|------|------|----------|--------|--------|
| V1 | 0.55 | 0.50 | 0.55 | 91.182 | -100% |
| V2 | 0.60 | 0.45 | 0.60 | 39.897 | -100% |
| **V3 (Final)** | **0.65** | **0.40** | **0.65** | **23.839** | **-100%** |

**Ergebnis:** Höhere Thresholds reduzieren Trades, aber ändern nichts am Totalverlust!

---

**Status:** ❌ Gescheitert, aber wertvolles Learning!  
**Datum:** 18.01.2026  
**Verantwortlich:** USW-Group-10
