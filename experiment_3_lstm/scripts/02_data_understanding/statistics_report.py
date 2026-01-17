import pandas as pd
import os
import yaml

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
params = yaml.safe_load(open(os.path.join(script_dir, "../../conf/params.yaml")))
base_path = params['DATA_ACQUISITON']['DATA_PATH']

# 1. Rohdaten Stats (Data Understanding)
raw_path = os.path.join(script_dir, base_path, "processed/merged_raw_data.parquet")
if os.path.exists(raw_path):
    df_raw = pd.read_parquet(raw_path)
    print("=== DATA UNDERSTANDING: DESCRIPTIVE STATISTICS (RAW) ===")
    # Wähle wichtige Spalten
    cols = ['close', 'volume', 'qqq_close', 'nvda_close', 'rates_US_10Y_YIELD']
    # Zeige nur existierende Spalten
    cols = [c for c in cols if c in df_raw.columns]
    stats = df_raw[cols].describe().transpose()
    print(stats[['mean', 'std', 'min', '50%', 'max']])
    print("\n")

# 2. Feature Stats (Data Preparation)
feat_path = os.path.join(script_dir, base_path, "processed/training_data.parquet")
if os.path.exists(feat_path):
    df_feat = pd.read_parquet(feat_path)
    print("=== DATA PREPARATION: DESCRIPTIVE STATISTICS (FEATURES) ===")
    # Wähle deine neuen Profi-Features
    cols = ['log_ret', 'rsi_14_norm', 'slope_close_norm', 'ratio_btc_nvda', 'target']
    cols = [c for c in cols if c in df_feat.columns]
    stats = df_feat[cols].describe().transpose()
    print(stats[['mean', 'std', 'min', '50%', 'max']])