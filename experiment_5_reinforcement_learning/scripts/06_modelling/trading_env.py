"""
06_modelling/trading_env.py

Experiment 5: Custom Trading Environment für Reinforcement Learning
Implementiert Gym Environment für Bitcoin Trading mit DQN.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class BitcoinTradingEnv(gym.Env):
    """
    Custom Trading Environment für Reinforcement Learning.
    
    Action Space:
        0: Hold (keine Aktion)
        1: Buy (Long position)
        2: Sell (Short position oder Close Long)
    
    Observation Space:
        Vektor mit technischen Indikatoren (14 Features)
    
    Reward:
        Profit/Loss - Fees - Risk Penalty
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=100000, fee_rate=0.001, max_steps=None):
        super(BitcoinTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.max_steps = max_steps or len(df)
        
        # Feature columns (exclude target, timestamp, etc.)
        self.feature_cols = [c for c in df.columns if c not in 
                            ['timestamp', 'target', 'sample_weight', 'regime', 
                             'close', 'open', 'high', 'low', 'volume']]
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Feature vector
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.feature_cols),), 
            dtype=np.float32
        )
        
        # State
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0=Cash, 1=Long, -1=Short
        self.entry_price = 0
        self.total_profit = 0
        self.trade_history = []
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        self.trade_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation (feature vector)"""
        if self.current_step >= len(self.df):
            return np.zeros(len(self.feature_cols), dtype=np.float32)
        
        obs = self.df.loc[self.current_step, self.feature_cols].values
        return obs.astype(np.float32)
    
    def _get_current_price(self):
        """Get current close price"""
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.loc[self.current_step, 'close']
    
    def step(self, action):
        """
        Execute action and return (observation, reward, done, info)
        
        Actions:
            0: Hold
            1: Buy (Long)
            2: Sell (Short or Close Long)
        """
        current_price = self._get_current_price()
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            if self.position == 0:
                # Open Long
                self.position = 1
                self.entry_price = current_price
                reward -= self.fee_rate  # Entry fee
                
            elif self.position == -1:
                # Close Short, Open Long
                pnl = (self.entry_price - current_price) / self.entry_price
                reward += pnl - (self.fee_rate * 2)  # Close + Open fee
                self.position = 1
                self.entry_price = current_price
                
        elif action == 2:  # Sell
            if self.position == 0:
                # Open Short
                self.position = -1
                self.entry_price = current_price
                reward -= self.fee_rate  # Entry fee
                
            elif self.position == 1:
                # Close Long, Open Short
                pnl = (current_price - self.entry_price) / self.entry_price
                reward += pnl - (self.fee_rate * 2)  # Close + Open fee
                self.position = -1
                self.entry_price = current_price
        
        # Hold (action == 0): Calculate unrealized P&L
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.position
            reward += unrealized_pnl * 0.1  # Small reward for unrealized gains
        
        # Risk penalty (encourage risk management)
        if self.position != 0:
            price_change = abs(current_price - self.entry_price) / self.entry_price
            if price_change > 0.05:  # > 5% move
                reward -= 0.01  # Penalty for large exposure
        
        # Update balance
        self.balance *= (1 + reward)
        self.total_profit += reward
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = (self.current_step >= self.max_steps) or (self.balance < self.initial_balance * 0.5)
        
        # Get next observation
        obs = self._get_observation()
        
        # Info
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_profit': self.total_profit,
            'current_step': self.current_step
        }
        
        return obs, reward, done, False, info  # gymnasium: (obs, reward, terminated, truncated, info)
    
    def render(self, mode='human'):
        """Render environment state"""
        profit_pct = (self.balance / self.initial_balance - 1) * 100
        print(f"Step: {self.current_step}/{self.max_steps} | "
              f"Balance: ${self.balance:,.2f} | "
              f"Profit: {profit_pct:+.2f}% | "
              f"Position: {self.position}")
    
    def close(self):
        """Clean up"""
        pass


def make_trading_env(df, **kwargs):
    """Factory function to create trading environment"""
    return BitcoinTradingEnv(df, **kwargs)
