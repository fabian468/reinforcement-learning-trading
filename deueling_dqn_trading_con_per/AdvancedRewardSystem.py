# -*- coding: utf-8 -*-
"""
Created on Thu May 22 23:17:18 2025

@author: fabia
"""

"""
el problema con el drawndown es que cuando baja el total del balances 
un 10% la penalizacion es constantemente mayor 
la solucion seria o eliminar eso o darle un peso menor  o 
mandar el current equity actual y el peak equity actual calculado con el profit de solo esa accion
"""
import numpy as np
from collections import deque

class AdvancedRewardSystem:
    def __init__(self, initial_balance=100, risk_free_rate=0.02):
        self.initial_balance = initial_balance
        self.inv_initial_balance = 1.0 / initial_balance if initial_balance != 0 else 0
        self.risk_free_rate = risk_free_rate / 252  # Tasa diaria

        self.returns_buffer = deque(maxlen=50)
        self.equity_buffer = deque(maxlen=50)
        self.drawdown_buffer = deque(maxlen=30)

        self.previous_equity = initial_balance

        self.sumaRecompensaProfit = 0
        self.sumaRecompensaSharpe = 0
        self.sumaRecompensaDrawndown = 0
        self.sumaRecompensaConsistency = 0
        self.sumaRecompensaRiskAdjusted = 0
        self.sumaRecompensaMomentum = 0
        self.sumaRecompensaTradeQuality = 0

        self.weights = {
            'profit': 1.0,
            'sharpe': 0.3,
            'drawdown': 0.2,
            'consistency': 0.4,
            'risk_adjusted': 0.4,
            'momentum': 0.3,
            'trade_quality': 0.2
        }

        self.sum_map = {
            'profit': 'sumaRecompensaProfit',
            'sharpe': 'sumaRecompensaSharpe',
            'drawdown': 'sumaRecompensaDrawndown',
            'consistency': 'sumaRecompensaConsistency',
            'risk_adjusted': 'sumaRecompensaRiskAdjusted',
            'momentum': 'sumaRecompensaMomentum',
            'trade_quality': 'sumaRecompensaTradeQuality'
        }

    def calculate_reward(self, profit_dollars, current_equity, peak_equity, trade_returns_history, is_trade_closed=False):
        reward_components = {}

        reward_components['profit'] = profit_dollars

        returns_np = np.fromiter(self.returns_buffer, dtype=np.float32) if len(self.returns_buffer) > 0 else None

        reward_components['sharpe'] = self._calculate_sharpe_component(returns_np) if returns_np is not None and len(returns_np) >= 10 else 0
        reward_components['drawdown'] = self._calculate_drawdown_penalty(current_equity, peak_equity)
        reward_components['consistency'] = self._calculate_consistency_reward(returns_np) if returns_np is not None and len(returns_np) >= 5 else 0
        reward_components['risk_adjusted'] = self._calculate_risk_adjusted_reward(profit_dollars, returns_np) if returns_np is not None and len(returns_np) >= 5 else 0
        reward_components['momentum'] = self._calculate_momentum_reward(current_equity)
        reward_components['trade_quality'] = self._calculate_trade_quality_reward(profit_dollars, returns_np) if is_trade_closed and profit_dollars != 0 and returns_np is not None and len(returns_np) >= 5 else 0

        total_reward = 0.0
        for component, value in reward_components.items():
            weighted_value = self.weights.get(component, 0.0) * value
            setattr(self, self.sum_map[component], getattr(self, self.sum_map[component]) + weighted_value)
            total_reward += weighted_value

        self._update_buffers(profit_dollars, current_equity, current_equity / peak_equity if peak_equity > 0 else 1.0)

        return total_reward, reward_components

    def _calculate_sharpe_component(self, returns):
        std = np.std(returns)
        if std == 0:
            return 0
        excess_returns = returns - self.risk_free_rate
        sharpe = np.mean(excess_returns) / std
        return np.tanh(sharpe / 2)

    def _calculate_drawdown_penalty(self, current_equity, peak_equity):
        if peak_equity <= 0:
            return 0.0
        inv_peak = 1.0 / peak_equity
        drawdown_percentage = (peak_equity - current_equity) * inv_peak
        if drawdown_percentage > 0.05:
            penalty = -drawdown_percentage * 1.5
        elif drawdown_percentage > 0:
            penalty = -drawdown_percentage * 0.5
        else:
            penalty = 0.0
        return np.tanh(penalty) * 0.5

    def _calculate_consistency_reward(self, returns):
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.1 if mean_return >= 0 else -0.1
        cv = abs(std_return / mean_return) if mean_return != 0 else float('inf')
        score = 1 / (1 + cv)
        return -score if mean_return < 0 else score

    def _calculate_risk_adjusted_reward(self, profit_dollars, returns):
        std = np.std(returns)
        if std == 0:
            return 0
        return np.tanh(profit_dollars / (std * self.initial_balance))

    def _calculate_momentum_reward(self, current_equity):
        if len(self.equity_buffer) < 3:
            return 0
        eq = np.fromiter(self.equity_buffer, dtype=np.float32)
        recent_trend = np.mean(np.diff(eq[-3:]))
        return recent_trend * self.inv_initial_balance

    def _calculate_trade_quality_reward(self, profit_dollars, returns):
        recent = returns[-5:]
        avg_recent = np.mean(recent)
        if avg_recent == 0:
            return 0
        relative = profit_dollars / (abs(avg_recent) * self.initial_balance)
        return np.tanh(relative)

    def _update_buffers(self, profit_dollars, current_equity, drawdown_ratio):
        self.returns_buffer.append(profit_dollars * self.inv_initial_balance)
        self.equity_buffer.append(current_equity)
        self.drawdown_buffer.append(drawdown_ratio)

    def get_adaptive_weights(self, episode):
        adaptive_weights = self.weights.copy()
        if episode % 80 == 0:
            adaptive_weights['drawdown'] += 0.5
            adaptive_weights['consistency'] += 0.2
            adaptive_weights['profit'] += 0.5
        elif episode % 150 == 0:
            adaptive_weights['profit'] += 1.5
            adaptive_weights['sharpe'] += 1.3
            adaptive_weights['drawdown'] -= 1.1
        return adaptive_weights

    def reset_episode(self):
        self.returns_buffer.clear()
        self.equity_buffer.clear()
        self.drawdown_buffer.clear()


def calculate_advanced_reward(reward_system, profit_dollars, current_equity, peak_equity,
                              episode_returns, is_trade_closed=False, add_noise=True):
    
    reward, components = reward_system.calculate_reward(
        profit_dollars, current_equity, peak_equity,
        episode_returns, is_trade_closed
    )
    if add_noise:
        reward += np.random.normal(0, 0.01)
    return reward, components


