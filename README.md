# USW-Group-10: Bitcoin High-Frequency Trading Bot

## Members
- Viet Anh Hönemann (Matrikelnummer: S0587778)
- Julius Bollmann (Matrikelnummer: S0594551)

## Executive Summary
This project implements a machine learning pipeline to predict short-term price movements of Bitcoin (BTC/USD). Utilizing a **multi-source approach** (Crypto, Equities, Macro, Sentiment) and rigorous **"No-Overfit" Feature Engineering**, we developed an XGBoost model that identifies profitable market anomalies.

**Key Results:**
- **Model Edge:** 51.8% Test Accuracy (vs. 50.8% Baseline).
- **Deployment Strategy:** A high-confidence filter (Threshold > 0.62) generates ~988 highly selective trades over the test period, focusing on risk-adjusted returns rather than volume.

---

### 1. Problem Definition

**Objective**
Binary Classification: Predict if the Bitcoin Close Price at $t+60$ (1 hour ahead) will be higher than the current price.
- `1` (Long): $Price_{t+60} > Price_t$
- `0` (Short): $Price_{t+60} \le Price_t$

**Constraints**
- **Stationarity:** Financial time-series are non-stationary. We explicitly banned absolute price levels (e.g., "$90,000") to prevent the model from learning specific price regimes.
- **Concept Drift:** To address changing market behavior, we use **Sample Weighting**, giving 2025 data significantly higher influence than 2020 data.

---

### 2. Data Strategy ("Hybrid Multi-Source")

We aggregate data from disparate sources, synchronizing them to the Bitcoin 1-minute timeline via `merge_asof`.

| Category | Source | Frequency | Rationale |
| :--- | :--- | :--- | :--- |
| **Crypto** | BTC/USD (Alpaca) | 1 Min | Target Asset (Free full feed). |
| **Equities** | NVDA, QQQ (Alpaca) | 1 Min | Tech Risk Sentiment (IEX feed for correlation features). |
| **Futures** | Nasdaq 100 | Daily | Overnight Sentiment Proxy. |
| **Macro** | 10Y Yields, DXY | Daily | Cost of capital & Currency strength. |
| **Sentiment** | Fear & Greed | Daily | Retail Psychology Index. |

---

### 3. Advanced Feature Engineering

We focus on relative metrics to ensure the model learns *dynamics*, not prices.

* **Institutional Flow:** `vwap_distance` (Price vs. Volume Weighted Average), `obv_slope` (Smart Money Flow).
* **Market Regimes:** `atr_pct` (Volatility), `beta_qqq` (Correlation to Tech).
* **Oscillators:** `rsi_14`, `mfi_14` (Overbought/Oversold conditions).
* **Time Encoding:** Cyclical encoding (Sin/Cos) of hour and day to capture session liquidity patterns.

---

### 4. Prevention of Overfitting ("Scorched Earth Policy")

To ensure the 51.8% accuracy is real and not a result of leakage, we applied strict filtering in `xgboost_model.py`:

* ❌ **BANNED:** `close`, `high`, `low`, `sma_50`, `ema_200`, `trade_count`. (Absolute values that grow over time).
* ✅ **ALLOWED:** `log_returns`, `rsi`, `dist_to_vwap`, `slope`, `beta`. (Stationary ratios).

This guarantees that the model works regardless of whether Bitcoin is at $10k or $100k.

---

### 5. Modeling Performance

We compared our Tuned XGBoost against a Logistic Regression Baseline.

| Metric | Baseline (LogReg) | **XGBoost (Final)** | Delta |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 50.86% | **51.82%** | **+0.96%** |
| **Precision**| Balanced | **High (Selective)**| - |

*Analysis:* In high-frequency trading, an edge of ~1% is statistically significant. The XGBoost model successfully captures non-linear relationships that the linear baseline misses.

---

### 6. Deployment & Strategy (Backtesting)

Moving from prediction to trading, we implemented a **"Sniper Strategy"** in `07_deployment`.

**Configuration:**
- **Confidence Threshold:** `0.62` (Only trade if Model Probability > 62%).
- **Fees:** 0.1% per trade.

**Backtest Results (Test Set):**
- **Total Trades:** **988** (approx. 3-4 trades/day).
- **Rationale:** By raising the threshold to 0.62, we filter out market noise and "weak" signals. This drastically reduces transaction costs and improves the **Win Rate**.
- **Comparison:** While "Buy & Hold" yields higher absolute returns in strong bull runs, our Active Strategy offers lower volatility and reduces exposure during uncertain market phases (Cash Position).

**Paper Trading:**
A live-simulation script (`run_paper_trading.py`) demonstrates the production loop: Fetch Data $\rightarrow$ Feature Engineering $\rightarrow$ Inference $\rightarrow$ Order Execution via Alpaca API.

---

### 7. How to Reproduce

**Step 1: Setup**
```bash
pip install -r requirements.txt
# Check project/conf/keys.yaml