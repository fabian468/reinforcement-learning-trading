# -*- coding: utf-8 -*-
"""
Created on Thu May 22 23:17:18 2025

@author: fabia
"""

import numpy as np
from collections import deque
import math

class AdvancedRewardSystem:
    def __init__(self, initial_balance=100, risk_free_rate=0.02):
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate / 252  # Tasa diaria
        
        # Buffers para métricas móviles
        self.returns_buffer = deque(maxlen=100)
        self.equity_buffer = deque(maxlen=50)
        self.drawdown_buffer = deque(maxlen=30)
     
        # Pesos para componentes de recompensa
        self.weights = {
            'profit': 2.0,           # Base profit
            'sharpe': 0.5,          # Sharpe ratio component
            'drawdown': -1.0,       # Penalización por drawdown
            'consistency': 0.3,     # Consistencia de retornos
            'risk_adjusted': 0.4,   # Retorno ajustado por riesgo
            'momentum': 0.2,        # Momentum de equity
            'trade_quality': 0.3    # Calidad del trade pero al ser un valor de 0.3 disminuye el retorno de negativo o positivo
        }

    def calculate_reward(self, profit_dollars, current_equity, peak_equity, 
                        trade_returns_history, is_trade_closed=False):
        """
        Calcula recompensa multi-objetivo optimizada
        """
        reward_components = {}
        
        # 1. Componente base de profit (normalizado)
       
        #normalized_profit = profit_dollars / self.initial_balance
        normalized_profit= profit_dollars
        reward_components['profit'] = normalized_profit
        
        # 2. Componente de Sharpe Ratio (solo si hay suficientes datos)
        if len(self.returns_buffer) >= 10:
            sharpe_reward = self._calculate_sharpe_component()
            reward_components['sharpe'] = sharpe_reward
        else:
            reward_components['sharpe'] = 0
        
        # 3. Penalización por Drawdown (más sofisticada)
        drawdown_penalty = self._calculate_drawdown_penalty(current_equity, peak_equity)
        reward_components['drawdown'] = drawdown_penalty
        
        # 4. Componente de consistencia
        consistency_reward = self._calculate_consistency_reward()
        reward_components['consistency'] = consistency_reward
        
        # 5. Retorno ajustado por riesgo
        risk_adjusted_reward = self._calculate_risk_adjusted_reward(profit_dollars)
        reward_components['risk_adjusted'] = risk_adjusted_reward
        
        # 6. Momentum de equity
        momentum_reward = self._calculate_momentum_reward(current_equity)
        reward_components['momentum'] = momentum_reward
        
        # 7. Calidad del trade (solo si se cerró un trade)
        if is_trade_closed and profit_dollars != 0:
            trade_quality_reward = self._calculate_trade_quality_reward(profit_dollars)
            reward_components['trade_quality'] = trade_quality_reward
        else:
            reward_components['trade_quality'] = 0
        
        # Calcular recompensa final ponderada
        total_reward = sum(self.weights[component] * value 
                          for component, value in reward_components.items())
        
        # Actualizar buffers
        self._update_buffers(profit_dollars, current_equity, current_equity/peak_equity if peak_equity > 0 else 1)
        
        return total_reward, reward_components
    
    def _calculate_sharpe_component(self):
        """Calcula componente de recompensa basado en Sharpe ratio"""
        if len(self.returns_buffer) < 10:
            return 0
        
        returns = np.array(list(self.returns_buffer))
        if np.std(returns) == 0:
            return 0
        
        excess_returns = returns - self.risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(returns)
        
        # Normalizar Sharpe ratio (tanh para límites suaves)
        normalized_sharpe = np.tanh(sharpe / 2)  # Divide por 2 para escalar
        return normalized_sharpe
    
    def _calculate_drawdown_penalty(self, current_equity, peak_equity):
        """Penalización sofisticada por drawdown"""
        if peak_equity <= 0:
            return 0
        
        #current_drawdown=(100-88) / 100
        current_drawdown = (peak_equity - current_equity) / peak_equity
        
        # Penalización exponencial para drawdowns grandes
        if current_drawdown > 0:
            # Penalización más severa para drawdowns > 10%
            if current_drawdown > 0.1:
                penalty = np.exp(current_drawdown * 5) + 1
            else:
                penalty = current_drawdown * 2
        else:
            penalty = 0.1  # Pequeña recompensa por estar en peak
        
        return penalty
    
    def _calculate_consistency_reward(self):
        """Recompensa por consistencia en retornos"""
        if len(self.returns_buffer) < 5:
            return 0
        
        returns = np.array(list(self.returns_buffer))
        if len(returns) == 0:
            return 0
        
        # Calcular coeficiente de variación (menor es mejor)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.1 if mean_return >= 0 else -0.1
        
        cv = abs(std_return / mean_return) if mean_return != 0 else float('inf')
        
        # Recompensa inversa al coeficiente de variación
        consistency_score = 1 / (1 + cv)
        return consistency_score - 0.5  # Centrar en 0
    
    def _calculate_risk_adjusted_reward(self, profit_dollars):
        """Recompensa ajustada por el riesgo tomado"""
        if len(self.returns_buffer) < 5:
            return 0
        
        returns = np.array(list(self.returns_buffer))
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0
        
        # Return to risk ratio        
        risk_adjusted = profit_dollars / (volatility * self.initial_balance)
        return np.tanh(risk_adjusted)  # Normalizar
    
    def _calculate_momentum_reward(self, current_equity):
        """Recompensa basada en momentum de equity"""
        if len(self.equity_buffer) < 3:
            return 0
        
        equity_series = np.array(list(self.equity_buffer))
        
        # Calcular momentum simple (diferencias)
        if len(equity_series) >= 3:
            recent_trend = np.mean(np.diff(equity_series[-3:]))
            momentum_score = np.tanh(recent_trend / self.initial_balance)
            return momentum_score
        
        return 0
    
    def _calculate_trade_quality_reward(self, profit_dollars):
        """Evalúa la calidad del trade basado en contexto histórico"""
        if len(self.returns_buffer) < 5:
            return 0
        
        recent_returns = list(self.returns_buffer)[-5:]
        avg_recent_return = np.mean(recent_returns)
        
        #avg_recent_return= -0.50
        # Recompensa trades que superan el promedio reciente
        if avg_recent_return != 0:
            relative_performance = profit_dollars / (abs(avg_recent_return) * self.initial_balance)
            return np.tanh(relative_performance - 1)  # -1 para que sea relativo
        
        return 0
    
    def _update_buffers(self, profit_dollars, current_equity, drawdown_ratio):
        """Actualiza los buffers de métricas"""
        normalized_return = profit_dollars / self.initial_balance if self.initial_balance > 0 else 0
        self.returns_buffer.append(normalized_return)
        self.equity_buffer.append(current_equity)
        self.drawdown_buffer.append(drawdown_ratio)
    
    def get_adaptive_weights(self, episode, max_episodes):
        """Ajusta pesos dinámicamente durante el entrenamiento"""
        progress = episode / max_episodes
        
        # Al inicio, priorizar exploración y minimizar riesgo
        # Al final, priorizar profit y sharpe
        adaptive_weights = self.weights.copy()
        
        if progress < 0.3:  # Primeras 30% episodios
            adaptive_weights['drawdown'] *= 1.5  # Más conservador
            adaptive_weights['consistency'] *= 1.3
            adaptive_weights['profit'] *= 0.8
        elif progress > 0.7:  # Últimos 30% episodios
            adaptive_weights['profit'] *= 1.2
            adaptive_weights['sharpe'] *= 1.3
            adaptive_weights['drawdown'] *= 0.8
        
        return adaptive_weights
    
    def reset_episode(self):
        """Reinicia buffers para nuevo episodio"""
        self.returns_buffer.clear()
        self.equity_buffer.clear()
        self.drawdown_buffer.clear()


# Función auxiliar para integrar en tu código existente
def calculate_advanced_reward(reward_system, profit_dollars, current_equity, peak_equity,
                            episode_returns, is_trade_closed=False, add_noise=True):
    """
    Función para integrar fácilmente en tu código existente
    """
    # Calcular recompensa avanzada
    reward, components = reward_system.calculate_reward(
        profit_dollars, current_equity, peak_equity, 
        episode_returns, is_trade_closed
    )
    
    # Agregar ruido pequeño para exploración (opcional)
    if add_noise:
        noise = np.random.normal(0, 0.01)
        reward += noise
    
    return reward, components


# Ejemplo de uso en tu código principal:
