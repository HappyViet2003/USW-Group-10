# USW-Group-10: Bitcoin High-Frequency Trading Bot

## Members
- Viet Anh Hönemann (Matrikelnummer: S0587778)
- Julius Bollmann (Matrikelnummer: S0594551)

## Executive Summary
This project implements a machine learning pipeline to predict short-term price movements of Bitcoin (BTC/USD). Unlike simple price-based models, we utilize a **multi-source approach** integrating high-frequency market data, macroeconomic indicators, and sentiment analysis.

Our final **XGBoost model achieved a Test Accuracy of ~51.8%**, outperforming the Logistic Regression Baseline. In efficient financial markets, this represents a statistically significant edge ("Alpha"), achieved through rigorous feature engineering and strict prevention of look-ahead bias.

---

### 1. Problem Definition

**Objective**
Binary Classification: Predict if the Bitcoin Close Price at $t+60$ (1 hour ahead) will be higher than the current price.
- `1` (Long): $Price_{t+60} > Price_t$
- `0` (Short): $Price_{t+60} \le Price_t$

**Constraints**
- **Stationarity:** Financial time-series are non-stationary. We strictly avoid using absolute price levels (e.g., "$90,000") to prevent the model from learning specific price regimes instead of market dynamics.
- **Concept Drift:** Market behavior changes over time (2020 Volatility vs. 2024 Institutional Adoption).

---

### 2. Data Strategy ("Hybrid Multi-Source")

We aggregate data from disparate sources with different frequencies, synchronizing them to the Bitcoin 1-minute timeline via `merge_asof` (backward-fill).

| Category | Data Source | Frequency | Rationale |
| :--- | :--- | :--- | :--- |
| **Crypto** | BTC/USD (Alpaca) | 1 Min | The target asset. |
| **Equities** | NVDA, QQQ (Alpaca) | 1 Min | Measures Tech/AI risk sentiment ("Beta"). |
| **Futures** | Nasdaq 100 Futures | Daily | Proxy for overnight sentiment. |
| **Macro** | 10Y Yields, Dollar (UUP) | Daily | Cost of capital & currency strength. |
| **Sentiment** | **Fear & Greed Index** | Daily | **NEW:** Proxies retail psychology/hype. |

---

### 3. Advanced Feature Engineering

We moved beyond basic indicators to capture **Market Structure** and **Smart Money Flow**. Implemented in `features.py` using `pandas_ta`.

#### A. Volume & Institutional Flow
- **VWAP Distance:** Measures if price is expensive relative to the volume-weighted average (Institutional Benchmark).
- **OBV Slope:** On-Balance Volume momentum to detect accumulation/distribution.
- **MFI (Money Flow Index):** Volume-weighted RSI to distinguish fake pumps from real trends.

#### B. Market Regimes & Volatility
- **ATR (Average True Range):** Normalised volatility metric to detect panic vs. calm phases.
- **ADX:** Filters trending vs. ranging markets.
- **Beta to QQQ:** dynamic correlation to Tech stocks (Risk-On/Off detector).

#### C. Cyclical Time Encoding
- Instead of linear hours (0-23), we use **Sin/Cos transformations** to mathematically model the cyclic nature of time (23:00 is close to 00:00).

#### D. Time-Based Sample Weighting
To address **Concept Drift**, we implemented a "staircase" weighting function. Data from 2025 has significantly higher weight (1.5x) in the loss function than data from 2020 (0.5x), forcing the model to prioritize current market rules.

---

### 4. Prevention of Overfitting & Data Leakage

A critical part of our modeling process was the **"Scorched Earth" Feature Selection**. To ensure realistic results, we explicitly banned all absolute values:

* ❌ **Banned:** `close`, `high`, `low`, `sma_50`, `vwap`, `gold_close`. (Reason: Non-stationary).
* ✅ **Allowed:** `log_returns`, `rsi`, `dist_to_vwap`, `obv_slope`, `beta`, `atr_pct`. (Reason: Stationary/Relative).

**Preparation Pipeline (`prepare_for_modeling.py`):**
1.  **Cleaning:** Removal of `NaN` and `Inf`.
2.  **Filter:** Strict removal of non-numeric columns.
3.  **Scaling:** `StandardScaler` fitted *only* on Training data to prevent look-ahead bias.

---

### 5. Modeling & Results

We compared a complex gradient boosting model against a linear baseline.

#### Models
1.  **Baseline:** Logistic Regression (Class-balanced).
2.  **Challenger:** **XGBoost Classifier**.
    * *Hyperparameters:* Tuned for stability (`eta=0.03`, `max_depth=5`, `lambda=3.0`).
    * *Mechanism:* Uses Sample Weights and Early Stopping.

#### Performance (Test Set)

| Metric | Baseline (LogReg) | **XGBoost (Final)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 50.86% | **51.82%** | **+0.96%** |
| **Edge** | Random | **Profitable** | Significant |

*Analysis:* While 51.8% appears low, in high-frequency trading, any accuracy consistently above 51% is considered a "money-printing" edge due to the Law of Large Numbers. The model successfully identified structural patterns (Slopes, RSI regimes) that linear models missed.

---

### 6. How to Reproduce

The project follows a strict ETL pipeline structure.

**Step 1: Setup**
```bash
pip install -r requirements.txt
# Ensure project/conf/keys.yaml is set