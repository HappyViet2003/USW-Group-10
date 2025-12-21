# USW-Group-10: Bitcoin High-Frequency Trading Bot

## Members
- Viet Anh Hönemann (Matrikelnummer: S0587778)
- Julius Bollmann (Matrikelnummer: S0594551)

## Executive Summary
This project implements a machine learning pipeline to predict short-term price movements of Bitcoin (BTC/USD). Utilizing a **multi-source approach** (Crypto, Equities, Macro, Sentiment) and rigorous **"No-Overfit" Feature Engineering**, we developed an XGBoost model that identifies market patterns.

**Key Results:**
- **Model Performance:** 52.08% Test Accuracy (vs. 50% random baseline)
- **Backtesting:** Strategy underperformed Buy & Hold (-65% vs. +100%) over test period
- **Key Insight:** Demonstrates the fundamental challenge of profitable short-term crypto trading - even statistically significant model improvements don't guarantee trading profits

---

## Table of Contents
1. [Problem Definition](#1-problem-definition)
2. [Data Strategy](#2-data-strategy-hybrid-multi-source)
3. [Feature Engineering](#3-advanced-feature-engineering)
4. [Overfitting Prevention](#4-prevention-of-overfitting-scorched-earth-policy)
5. [Model Performance](#5-modeling-performance)
6. [Deployment & Backtesting](#6-deployment--backtesting)
7. [Paper Trading](#7-paper-trading)
8. [Limitations & Next Steps](#8-limitations--next-steps)
9. [How to Reproduce](#9-how-to-reproduce)

---

### 1. Problem Definition

**Objective**  
Binary Classification: Predict if the Bitcoin Close Price at $t+60$ (1 hour ahead) will be higher than the current price.
- `1` (Long): $Price_{t+60} > Price_t$
- `0` (Short): $Price_{t+60} \le Price_t$

**Constraints**
- **Stationarity:** Financial time-series are non-stationary. We explicitly banned absolute price levels (e.g., "$90,000") to prevent the model from learning specific price regimes.
- **Concept Drift:** To address changing market behavior, we use **Sample Weighting**, giving recent data (2024-2025) significantly higher influence than older data (2020-2023).

---

### 2. Data Strategy ("Hybrid Multi-Source")

We aggregate data from disparate sources, synchronizing them to the Bitcoin 1-minute timeline via `merge_asof`.

| Category | Source | Frequency | Rationale |
| :--- | :--- | :--- | :--- |
| **Crypto** | BTC/USD (Alpaca) | 1 Min | Target Asset (Free full feed). |
| **Equities** | NVDA, QQQ (Alpaca) | 1 Min | Tech Risk Sentiment (IEX feed for correlation features). |
| **Futures** | Nasdaq 100 (NQ=F) | 24/7 | Overnight Sentiment Proxy (avoids weekend data gaps). |
| **Commodities** | Gold (GLD), USD (UUP) | Daily | Safe-haven assets & Currency strength. |
| **Macro** | 10Y Yields, M2 Money Supply | Daily | Cost of capital & Liquidity. |
| **Sentiment** | Fear & Greed Index | Daily | Retail Psychology Index. |

**Data Period:** January 2020 - January 2025 (5 years, ~2.6M data points)

---

### 3. Advanced Feature Engineering

We focus on relative metrics to ensure the model learns *dynamics*, not prices.

**Categories:**

* **Returns & Volatility:** `log_ret`, `log_ret_5m`, `log_ret_15m`, `log_ret_30m`, `hist_vol_30`, `hist_vol_60`
* **Volume Indicators:** `obv` (On-Balance Volume), `obv_slope`, `mfi_14` (Money Flow Index), `vwap`, `dist_vwap`
* **Momentum Oscillators:** `rsi_14`, `stoch_k`, `cci`, `willr`, `macd`, `macd_histogram`
* **Trend Indicators:** `adx`, `bb_width`, `bb_position`, `ichimoku` (distance to cloud, cloud thickness, TK cross)
* **Volume Analysis:** `volume_spike`, `volume_momentum`, `volume_trend`
* **Support/Resistance:** `local_high`, `local_low`, `dist_local_high`
* **Candlestick Patterns:** `body_size`, `upper_wick`, `lower_wick`, `wick_ratio`
* **Macro Correlations:** `ratio_btc_nq`, `beta_qqq`, `corr_btc_gld_60m`, `corr_btc_uup_60m`
* **Time Encoding:** Cyclical encoding (Sin/Cos) of hour and day to capture session liquidity patterns

**Total Features:** ~80+ engineered features, all normalized using rolling z-score (window=1440 minutes = 1 day)

---

### 4. Prevention of Overfitting ("Scorched Earth Policy")

To ensure the 52% accuracy is real and not a result of leakage, we applied strict filtering in `xgboost_model.py`:

* ❌ **BANNED:** `close`, `high`, `low`, `open`, `volume`, `sma_50`, `ema_200`, `obv`, `vwap`, `trade_count`, `m2_close`, all external raw prices
* ✅ **ALLOWED:** `log_returns`, `rsi`, `dist_to_vwap`, `slope`, `beta`, `ratios`, `correlations`, normalized indicators

This guarantees that the model works regardless of whether Bitcoin is at $10k or $100k.

**Additional Safeguards:**
- Rolling z-score normalization (no global statistics)
- StandardScaler fitted only on training data
- Time-series split (no shuffling)
- Sample weighting (linear from 0.5 to 1.5 over months)

---

### 5. Modeling Performance

We compared our Tuned XGBoost against a Logistic Regression Baseline.

| Metric | Baseline (LogReg) | **XGBoost (Final)** | Delta |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 50.86% | **52.08%** | **+1.22%** |
| **Precision** | 50.2% | **52.5%** | +2.3% |
| **Recall** | 51.0% | **51.8%** | +0.8% |
| **AUC** | 0.508 | **0.523** | +0.015 |

**Hyperparameters (Optimized):**
```python
{
    'max_depth': 3,              # Shallow trees (prevent overfitting)
    'eta': 0.01,                 # Slow learning rate
    'subsample': 0.7,            # 70% data per tree
    'colsample_bytree': 0.6,     # 60% features per tree
    'min_child_weight': 10,      # Robust leaf nodes
    'gamma': 0.2,                # Split threshold
    'lambda': 5.0,               # L2 regularization
    'alpha': 1.0,                # L1 regularization
}
```

**Analysis:** While 52% is only 2% above random chance, it is statistically significant over 200k+ test samples. However, as backtesting shows, this edge is insufficient for profitable trading after fees.

---

### 6. Deployment & Backtesting

Moving from prediction to trading, we implemented a **"High-Confidence Filter Strategy"** in `07_deployment/01_backtesting`.

**Strategy Configuration:**
- **Entry Signal:** Buy when Model Probability > 62% (filters weak signals)
- **Exit Signal:** Hold for 1 hour, then sell (or until next signal)
- **Position Sizing:** 100% allocation when signal active, 0% in cash otherwise
- **Trading Fees:** 0.1% per trade (realistic for crypto exchanges)

**Backtest Results (Test Set: Jan 2024 - Jan 2025):**

| Metric | Value |
|--------|-------|
| **Total Trades** | 988 trades (~3-4 per day) |
| **Winning Trades** | 487 (49.3%) |
| **Losing Trades** | 501 (50.7%) |
| **Win Rate** | 49.3% |
| **Avg Win** | +0.82% |
| **Avg Loss** | -0.89% |
| **Profit Factor** | 0.92 (losses > wins) |
| **Time in Market** | 42.3% |
| **Buy & Hold Return** | **+100.2%** |
| **Strategy Return** | **-65.4%** |
| **Sharpe Ratio** | -1.23 |
| **Max Drawdown** | -68.7% |
| **Total Fees Paid** | -2.1% |

**⚠️ Critical Finding:** The strategy significantly underperformed Buy & Hold.

**Root Cause Analysis:**

1. **Insufficient Edge:** 52% model accuracy translates to only 49.3% profitable trades after fees
2. **Asymmetric Losses:** Average loss (-0.89%) exceeds average win (+0.82%)
3. **Transaction Costs:** 988 trades × 0.1% fee = -2.1% drag on returns
4. **Market Regime:** Test period (2024) was a strong bull market - passive holding outperformed
5. **No Risk Management:** Strategy lacks stop-loss, take-profit, or position sizing

**Visualizations:**
- `project/images/07_backtest_result.png` - Equity curve comparison
- `project/images/07_backtest_distribution.png` - Trade distribution, win/loss analysis, monthly activity

**Key Insight:** This demonstrates a fundamental truth in quantitative trading: **predictive accuracy ≠ profitability**. Even statistically significant models can lose money due to transaction costs, market regime, and lack of risk management.

---

### 7. Paper Trading

**Setup:**  
We implemented a simulation framework (`07_deployment/02_paper_trading/run_paper_trading.py`) that demonstrates the production trading loop:

```
1. Fetch Live Data (Alpaca API) → 
2. Feature Engineering (RSI, MACD, Ichimoku, etc.) → 
3. Model Inference (XGBoost) → 
4. Order Execution (Alpaca Paper Trading API)
```

**Simulation Results:**  
Due to the negative backtest results, we did not deploy the strategy to live paper trading. The simulation script serves as a **proof-of-concept** for the deployment architecture.

**Time Frame:** N/A (not deployed due to unprofitable backtest)

**Comparison to Backtest:** Simulation uses same threshold (0.62) and fee structure (0.1%). Expected performance would mirror backtest results (-65%).

**Technical Implementation:**
- Mock API classes for demonstration
- Real-time feature calculation pipeline
- Order execution logic with confidence thresholds
- Error handling and logging

---

### 8. Limitations & Next Steps

**Current Limitations:**

1. **Profitability:** Backtest shows -65% return vs. +100% Buy & Hold
   - Root cause: 52% accuracy insufficient for 1-hour trading after fees
   - Model edge too small to overcome transaction costs

2. **Risk Management:** No stop-loss, take-profit, or dynamic position sizing
   - Strategy takes full losses on wrong predictions
   - No mechanism to cut losses early or lock in profits

3. **Time Horizon:** 1-hour predictions too short-term
   - High noise-to-signal ratio
   - Technical indicators (RSI, MACD) optimized for longer timeframes

4. **Market Regime Dependency:** Strategy performs poorly in bull markets
   - Tested during 2024 bull run (+100% BTC)
   - May perform better in sideways/bear markets (untested)

5. **Paper Trading:** Only simulation, no live validation
   - No real-world slippage, latency, or API issues tested

**Planned Improvements (Future Iterations):**

**Short-Term (1-2 weeks):**
1. **Longer Time Horizons:** Test 4-hour and 24-hour predictions
   - Less noise, better for technical indicators
   - Expected accuracy improvement: 53-55%

2. **Risk Management:**
   - Implement Stop-Loss (-2%) and Take-Profit (+3%)
   - Dynamic position sizing based on confidence (50% at 0.55, 100% at 0.65)
   - Trailing stop-loss for trend-following

3. **Strategy Optimization:**
   - Lower threshold to 0.55 (more trades, better diversification)
   - Test different holding periods (30min, 2h, 4h)
   - Add market regime filter (only trade in sideways markets)

**Medium-Term (1-2 months):**
4. **Ensemble Models:** Combine XGBoost with:
   - Random Forest (different bias-variance tradeoff)
   - LSTM (captures sequential patterns)
   - Logistic Regression (linear baseline)
   - Voting/Stacking for robustness

5. **Alternative Features:**
   - On-chain metrics (transaction volume, whale movements, exchange flows)
   - Order book imbalance (bid/ask pressure)
   - Social media sentiment (Twitter/Reddit volume & sentiment)
   - Google Trends ("Bitcoin" search volume)

6. **Live Paper Trading:** Deploy for 2-4 weeks
   - Validate real-world performance
   - Test API reliability, latency, slippage
   - Compare to backtest results

**Long-Term (3+ months):**
7. **Multi-Asset Portfolio:**
   - Trade BTC, ETH, SOL simultaneously
   - Diversification reduces strategy-specific risk
   - Correlation-based position sizing

8. **Reinforcement Learning:**
   - Train RL agent (PPO, DQN) for dynamic strategy
   - Learns optimal entry/exit timing
   - Adapts to changing market conditions

**Realistic Goal:** Achieve 53-55% accuracy with proper risk management for break-even or small positive returns (10-20% annually).

**Academic Contribution:** This project demonstrates the **reality gap** between ML model performance and trading profitability - a critical lesson for quantitative finance education.

---

### 9. How to Reproduce

**Step 1: Setup**
```bash
git clone https://github.com/HappyViet2003/USW-Group-10.git
cd USW-Group-10
pip install -r requirements.txt
```

**Step 2: Configure API Keys**  
Edit `project/conf/keys.yaml`:
```yaml
ALPACA_API_KEY: "your_key"
ALPACA_SECRET_KEY: "your_secret"
FRED_API_KEY: "your_fred_key"
```

**Step 3: Run Pipeline**
```bash
# 1. Data Acquisition (Alpaca, FRED, Yahoo Finance)
python experiment_1/scripts/01_data_acquisition/fetch_btc_data.py
python experiment_1/scripts/01_data_acquisition/fetch_external_data.py

# 2. Data Understanding (EDA plots)
python experiment_1/scripts/02_data_understanding/plot_correlations.py

# 3. Feature Engineering
python experiment_1/scripts/03_pre_split_prep/features.py

# 4. Train/Val/Test Split
python experiment_1/scripts/04_split_data/split_data.py

# 5. Data Preparation (Scaling, Feature Selection)
python experiment_1/scripts/05_preparation/prepare_for_modeling.py

# 6. Model Training
python experiment_1/scripts/06_modelling/xgboost_model.py

# 7. Backtesting
python experiment_1/scripts/07_deployment/01_backtesting/run_backtest.py

# 8. Paper Trading (Simulation)
python experiment_1/scripts/07_deployment/02_paper_trading/run_paper_trading.py
```

**Step 4: View Results**
- Model metrics: `project/data/models/feature_importance.csv`
- Backtest results: `project/data/models/backtest_results.csv`
- Visualizations: `project/images/`

---

## Project Structure
```
USW-Group-10/
├── project/
│   ├── conf/
│   │   ├── params.yaml          # Pipeline configuration
│   │   └── keys.yaml            # API keys (gitignored)
│   ├── data/
│   │   ├── raw/                 # Raw data from APIs
│   │   ├── processed/           # Engineered features
│   │   └── models/              # Trained models & results
│   ├── images/                  # Plots & visualizations
│   └── scripts/
│       ├── 01_data_acquisition/ # Data fetching
│       ├── 02_data_understanding/ # EDA
│       ├── 03_pre_split_prep/   # Feature engineering
│       ├── 04_split_data/       # Train/val/test split
│       ├── 05_preparation/      # Scaling & selection
│       ├── 06_modelling/        # Model training
│       └── 07_deployment/       # Backtesting & paper trading
├── README.md
└── requirements.txt
```

---

## References & Learning Resources

**Books:**
- *Python for Data Analysis* by Wes McKinney (Pandas fundamentals)
- *Advances in Financial Machine Learning* by Marcos López de Prado (Overfitting, backtesting)

**Papers:**
- *Machine Learning for Algorithmic Trading* (2nd Ed.) - Stefan Jansen
- *Quantitative Trading* - Ernest Chan

**Libraries:**
- XGBoost: https://xgboost.readthedocs.io/
- pandas_ta: https://github.com/twopirllc/pandas-ta
- Alpaca API: https://alpaca.markets/docs/

---

## License
MIT License - Free for educational use.

---

## Acknowledgments
- Prof. [Name] for guidance on ML best practices
- Alpaca Markets for free market data API
- FRED for macroeconomic data
- Alternative.me for Fear & Greed Index

---

**Last Updated:** December 13, 2025
