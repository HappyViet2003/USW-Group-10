"""
06_modelling/trading_env.py
FIX: Math-Safety + Info-Keys für Backtesting.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class BitcoinTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=100000, fee_rate=0.001, max_steps=None):
        super(BitcoinTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.max_steps = max_steps or len(df)

        # Features auswählen
        self.feature_cols = [c for c in df.columns if c not in
                            ['timestamp', 'target', 'sample_weight', 'regime',
                             'close', 'open', 'high', 'low', 'volume']]

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_cols),), dtype=np.float32
        )

        # Status-Variablen
        self.current_step = 0
        self.cash = initial_balance
        self.btc_held = 0.0
        self.total_value = initial_balance
        self.last_value = initial_balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_balance
        self.btc_held = 0.0
        self.total_value = self.initial_balance
        self.last_value = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        if self.current_step >= len(self.df):
            return np.zeros(len(self.feature_cols), dtype=np.float32)
        return self.df.loc[self.current_step, self.feature_cols].values.astype(np.float32)

    def _get_current_price(self):
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.loc[self.current_step, 'close']

    def step(self, action):
        current_price = self._get_current_price()

        # 1. Action Execution
        if action == 1:  # BUY
            if self.cash > 10: # Mindestbetrag
                amount = self.cash * (1 - self.fee_rate)
                self.btc_held = amount / current_price
                self.cash = 0.0
        elif action == 2:  # SELL
            if self.btc_held > 1e-8:
                self.cash = (self.btc_held * current_price) * (1 - self.fee_rate)
                self.btc_held = 0.0

        # 2. Portfolio Value
        current_value = self.cash + (self.btc_held * current_price)
        self.total_value = current_value

        # 3. Reward Calculation (Crash Safe)
        if current_value < self.initial_balance * 0.01:
            reward = -1.0
        else:
            try:
                reward = np.log(current_value / self.last_value)
            except:
                reward = -1.0

        self.last_value = current_value
        self.current_step += 1

        # Done Condition
        done = (self.current_step >= self.max_steps - 1) or (current_value < self.initial_balance * 0.1)

        # --- HIER WAR DER FEHLER: INFO DICT MUSS POSITION ENTHALTEN ---
        info = {
            'balance': self.total_value,
            'position': 1 if self.btc_held > 1e-8 else 0, # 1 = Investiert, 0 = Cash
            'btc_held': self.btc_held
        }

        return self._get_observation(), reward, done, False, info

def make_trading_env(df, **kwargs):
    return BitcoinTradingEnv(df, **kwargs)