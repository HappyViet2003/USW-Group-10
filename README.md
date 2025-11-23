# USW-Group-10

## Members
- Viet Anh Hönemann (Matrikelnummer: S0587778)
- Julius Bollmann (Matrikelnummer: S0594551)

## Trading Review: Problem Definition And Data Acquisition

### 1. Problem Definition

**Target Variable**
The goal is to predict the market trend direction for Bitcoin (BTC/USD).
Specifically, we treat this as a **Binary Classification Problem**:
- **Target (y):** Will the Close Price at time $t+60$ (in 1 hour) be higher than the Close Price at time $t$ (now)?
  - `1` (Long/Buy): Price(t+60) > Price(t)
  - `0` (Short/Sell): Price(t+60) <= Price(t)

**Input Variables (Features)**
For every minute from 2020-01-01 until 2025-01-01, we utilize the following feature sets:

1.  **Raw Market Data (from Alpaca):**
    - `Timestamp` (UTC)
    - `Open`, `High`, `Low`, `Close` Prices
    - `Volume` (Number of coins traded)
    - `Trade Count` (Number of trades per minute)

2.  **Derived Technical Indicators (Feature Engineering):**
    - **Momentum:** RSI (Relative Strength Index, 14-period) to identify overbought/oversold conditions.
    - **Trend:** SMA (Simple Moving Average) and EMA (Exponential Moving Average) for 50 and 200 periods.
    - **Volatility:** Bollinger Bands (Upper/Lower bands) to measure market deviations.
    - **MACD:** Moving Average Convergence Divergence to spot trend reversals.

---

### 2. Data Acquisition

**Approach**
We implemented a robust data pipeline using the official **Alpaca Market Data API (v2)**. This ensures high-quality, split-adjusted data directly from a regulated broker.

**API Specification**
- **Provider:** Alpaca Markets
- **API Service:** `CryptoHistoricalDataClient` (Alpaca-py SDK)
- **Data Feed:** Global Crypto Feed (accessing multiple exchanges)
- **Asset Class:** Cryptocurrency (Bitcoin)

**Parameters**
We configured the data retrieval with the following parameters:
- **Symbol:** `BTC/USD`
- **Timeframe:** `1 Minute` (High-frequency granularity)
- **Date Range:** `2020-01-01` to `2025-01-01`
- **Adjustment:** Raw crypto data (no stock splits applicable)

**Storage Strategy**
To handle the large volume of high-frequency data, we avoid CSV files.
- **Format:** **Apache Parquet** (`.parquet`)
- **Reasoning:** Parquet provides superior compression (reducing file size by ~70%) and significantly faster read/write speeds for time-series analysis compared to CSV.

[project/scripts/01_data_acquisition/data_acquisition.py](project/scripts/01_data_acquisition/data_acquisition.py) contains the implementation of the data acquisition process.

Pulls **1-minute** BTC/USD data from Alpaca API for the specified date range and saves it as a Parquet file for efficient storage and retrieval.

Data preview:
<img src="project/images/01_data_acquisition.png" alt="drawing" width="800"/>