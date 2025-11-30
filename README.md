# USW-Group-10

## Members
- Viet Anh Hönemann (Matrikelnummer: S0587778)
- Julius Bollmann (Matrikelnummer: S0594551)

## Trading Review: Bitcoin Short-Term Prediction

### 1. Problem Definition

**Objective**
The goal is to predict the short-term market trend of Bitcoin (BTC/USD) using a machine learning approach.
We model this as a **Binary Classification Problem**:
- **Target Variable (y):** Will the Bitcoin Close Price at time $t+60$ (in 1 hour) be higher than the current price at $t$?
  - `1` (Long/Buy): Price(t+60) > Price(t)
  - `0` (Short/Sell): Price(t+60) <= Price(t)

**Rationale**
Bitcoin is highly volatile and influenced by global liquidity and tech sentiment. A 60-minute prediction window balances signal-to-noise ratio better than ultra-high-frequency (1-minute) or long-term (daily) predictions.

---

### 2. Data Acquisition Strategy ("Hybrid Multi-Source Approach")

To capture the full complexity of the crypto market, we moved beyond a single data source. We implemented a **multi-frequency data pipeline** that integrates high-frequency crypto data with macroeconomic indicators.

| Asset Class | Symbol | Source | Frequency | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Target Asset** | `BTC/USD` | **Alpaca** | 1 Minute | The asset to trade (High precision required). |
| **Tech Sentiment** | `QQQ` (Nasdaq ETF) | **Alpaca** | 1 Minute | BTC correlates strongly with Risk-On Tech assets. |
| **Hype Factor** | `NVDA` (Nvidia) | **Alpaca** | 1 Minute | Captures specific AI-driven market euphoria. |
| **Futures** | `NQ=F` (Nasdaq 100) | **Yahoo** | Daily | Proxy for overnight/global sentiment when spot markets are closed. |
| **Currency** | `UUP` (USD ETF) | **Alpaca** | 1 Minute | Strong USD usually suppresses BTC prices (Inverse correlation). |
| **Safe Haven** | `GLD` (Gold ETF) | **Alpaca** | 1 Minute | Competing store of value asset. |
| **Interest Rates** | `^TNX` (10Y Yield) | **Yahoo** | Daily | Proxy for cost of capital (Liquidity indicator). |
| **Economy** | `M2` (Money Supply) | **FRED** | Weekly | Long-term inflation driver. |

**Handling Asynchronous Markets**
A major challenge was merging 24/7 Crypto data with traditional assets (Stocks/Futures) that have closing times and weekends.
- **Solution:** We utilize `pd.merge_asof` with a backward direction to map the *last known* macro data point to every Bitcoin minute.
- **Weekend Logic:** We purposefully keep weekend data (using Forward Fill for stocks) but added `day_of_week` features so the model can learn specific weekend behavior patterns.

---

### 3. Data Preparation & Feature Engineering

Our feature engineering pipeline transforms raw time-series data into stationary, machine-learning-ready features.

**A. Stationarity & Normalization**
- **Log-Returns:** We use `np.log(price / price.shift(1))` instead of simple percentage changes to ensure statistical properties (additivity, normal distribution).
- **Z-Score Normalization:** Features like Volume or RSI are Z-normalized (rolling window of 1440 minutes) to scale them to a standard range (roughly -3 to +3), making them comparable for the model.

**B. Trend & Momentum Features**
- **Slopes (Steigungen):** Instead of absolute Moving Averages (which drift over years), we calculate the *slope* (speed) of the price and EMAs to capture trend dynamics.
- **RSI (14) & Bollinger Bands:** Standard technical indicators to detect overbought/oversold conditions.

**C. Macro-Economic Features**
- **Relative Strength Ratios:** We calculate `BTC / NVDA` and `BTC / Nasdaq` ratios to detect decoupling events (e.g., Bitcoin pumping while Tech dumps).
- **Real Rate Impact:** Bitcoin price adjusted by the 10-Year Treasury Yield to measure "liquidity-adjusted" value.

**D. Data Cleaning & Weighting**
- **Outlier Detection:** We remove data points with a Z-Score > 10 (extreme flash crashes/API errors) to stabilize training.
- **Sample Weights:** To address **Concept Drift** (changing market regimes from 2020 to 2025), we implemented linear time-weighting. Recent data (2025) has 3x higher weight in the loss function than old data (2020).

---

