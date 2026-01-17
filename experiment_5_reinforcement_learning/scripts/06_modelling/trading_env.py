"""
06_modelling/trading_env.py

Experiment 5: Custom Trading Environment für Reinforcement Learning
FIX: Saubere Trennung von Portfolio-Balance und RL-Reward.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class BitcoinTradingEnv(gym.Env):
    """
    Custom Trading Environment.
    Action 0: Hold
    Action 1: Buy (100% invest)
    Action 2: Sell (100% cash)
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=100000, fee_rate=0.001, max_steps=None):
        super(BitcoinTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.max_steps = max_steps or len(df)

        # Features (alles außer Meta-Daten)
        self.feature_cols = [c for c in df.columns if c not in
                            ['timestamp', 'target', 'sample_weight', 'regime',
                             'close', 'open', 'high', 'low', 'volume']]

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_cols),), dtype=np.float32
        )

        # Internal State
        self.current_step = 0
        self.cash = initial_balance
        self.btc_held = 0.0
        self.total_value = initial_balance
        self.last_value = initial_balance
        self.trade_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_balance
        self.btc_held = 0.0
        self.total_value = self.initial_balance
        self.last_value = self.initial_balance
        self.trade_history = []
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
        reward = 0

        # 1. EXECUTE ACTION (Real Money Logic)
        if action == 1:  # BUY
            if self.cash > 0: # Nur kaufen wenn wir Cash haben
                amount_to_buy = self.cash * (1 - self.fee_rate) # Gebühren abziehen
                self.btc_held = amount_to_buy / current_price
                self.cash = 0.0

        elif action == 2:  # SELL
            if self.btc_held > 0: # Nur verkaufen wenn wir BTC haben
                proceeds = (self.btc_held * current_price) * (1 - self.fee_rate)
                self.cash = proceeds
                self.btc_held = 0.0

        # 2. UPDATE PORTFOLIO VALUE (Mark-to-Market)
        current_value = self.cash + (self.btc_held * current_price)
        self.total_value = current_value

        # 3. CALCULATE REWARD (RL Logic)
        # Reward ist die prozentuale Änderung des Portfolios + Boni/Mali
        # Log-Return ist stabiler für Training: ln(V_t / V_t-1)
        portfolio_return = np.log(current_value / self.last_value) if self.last_value > 0 else 0

        reward = portfolio_return

        # Penalty für Inaktivität (Optional, damit er tradet)
        # if action == 0 and self.btc_held == 0:
        #    reward -= 0.0001

        self.last_value = current_value
        self.current_step += 1

        # Check Done
        done = (self.current_step >= self.max_steps - 1) or (current_value < self.initial_balance * 0.1)

        info = {
            'balance': self.total_value,
            'position': 1 if self.btc_held > 0 else 0,
            'btc_held': self.btc_held
        }

        return self._get_observation(), reward, done, False, info

def make_trading_env(df, **kwargs):
    """Factory function to create trading environment"""
    return BitcoinTradingEnv(df, **kwargs)