# -*- coding: utf-8 -*-
"""
AdvancedRewardSystem v2.0 — reescrito 2026-04-08

Problemas del v1 corregidos:
  1. trade_quality saturaba en ±0.95 (denominador normalizado ≈0.0003 del returns_buffer)
     → ahora usa ratio avg_win / avg_loss real en dólares
  2. drawdown acumulaba penalización en cada trade con DD persistente → dominaba el total
     → ahora penaliza solo el INCREMENTO de drawdown entre trades, no el nivel absoluto
  3. avg_loss crece con accuracy sin penalización específica por loser grande
     → nuevo componente loss_severity: penaliza cuando un loser supera el promedio histórico
  4. Sin incentivo a dejar correr winners
     → nuevo componente win_retention: premia cuando un winner supera el promedio histórico
  5. Ruido gaussiano innecesario (NoisyLinear ya provee exploración)
     → eliminado
  6. momentum usando solo 3 puntos de equity → muy ruidoso
     → reemplazado por win_retention con señal real de magnitud
  7. Step reward no crecía con tiempo en pérdida → reforzaba aguantar losers
     → get_step_reward() penaliza más cuanto más tiempo lleva el trade en rojo

Mapeo de atributos (compatibilidad con run_con_per.py sin cambios):
  sumaRecompensaRiskAdjusted → loss_severity
  sumaRecompensaMomentum     → win_retention
"""

import numpy as np
from collections import deque


class AdvancedRewardSystem:

    def __init__(self, initial_balance=1000, risk_free_rate=0.02, weights=None):
        self.initial_balance = initial_balance
        self._inv_balance = 1.0 / initial_balance if initial_balance != 0 else 0
        self.risk_free_rate = risk_free_rate / 252

        # Buffers de historial (solo trades cerrados del episodio actual)
        self.returns_buffer  = deque(maxlen=50)   # profit/balance normalizado
        self.equity_buffer   = deque(maxlen=50)   # equity en USD
        self.wins_buffer     = deque(maxlen=50)   # profit_dollars de trades ganadores
        self.losses_buffer   = deque(maxlen=50)   # profit_dollars de trades perdedores (valores negativos)

        self._prev_drawdown  = 0.0  # drawdown del cierre anterior (para calcular delta)
        self._prev_ratio     = 1.0  # ratio avg_win/avg_loss del trade anterior (para calcular delta)

        # Acumuladores por episodio (accedidos externamente por run_con_per.py)
        self.sumaRecompensaProfit        = 0.0
        self.sumaRecompensaSharpe        = 0.0
        self.sumaRecompensaDrawndown     = 0.0
        self.sumaRecompensaConsistency   = 0.0
        self.sumaRecompensaRiskAdjusted  = 0.0  # → loss_severity
        self.sumaRecompensaMomentum      = 0.0  # → win_retention
        self.sumaRecompensaTradeQuality  = 0.0

        default_weights = {
            'profit':        1.0,
            'sharpe':        0.3,
            'drawdown':      0.2,
            'consistency':   0.1,
            'risk_adjusted': 0.5,   # loss_severity
            'momentum':      0.2,   # win_retention
            'trade_quality': 0.3,
        }
        self.weights = weights if weights is not None else default_weights

        # Mapa de componente → atributo acumulador
        self._sum_map = {
            'profit':        'sumaRecompensaProfit',
            'sharpe':        'sumaRecompensaSharpe',
            'drawdown':      'sumaRecompensaDrawndown',
            'consistency':   'sumaRecompensaConsistency',
            'risk_adjusted': 'sumaRecompensaRiskAdjusted',
            'momentum':      'sumaRecompensaMomentum',
            'trade_quality': 'sumaRecompensaTradeQuality',
        }

    # ------------------------------------------------------------------
    # API principal — llamada en cada cierre de trade
    # ------------------------------------------------------------------

    def calculate_reward(self, profit_dollars, current_equity, peak_equity,
                         trade_returns_history=None, is_trade_closed=False,
                         episode_wins=None, episode_losses=None):
        """
        Calcula reward al cierre de un trade.

        Returns
        -------
        total_reward : float
        components   : dict  {nombre: valor_sin_ponderar}
        """
        components = {}

        # 1. Profit — expectancy real del episodio
        components['profit'] = self._calc_profit(episode_wins, episode_losses)

        # 2. Sharpe — rolling sobre últimos 50 trades cerrados
        returns_np = np.fromiter(self.returns_buffer, dtype=np.float32)
        components['sharpe'] = self._calc_sharpe(returns_np)

        # 3. Drawdown — solo penaliza INCREMENTO, no nivel absoluto
        components['drawdown'] = self._calc_drawdown_delta(current_equity, peak_equity)

        # 4. Consistency — coeficiente de variación de retornos
        components['consistency'] = self._calc_consistency(returns_np)

        # 5. Loss severity — penaliza losers peores que el promedio histórico
        #    (mapeado a sumaRecompensaRiskAdjusted para compatibilidad)
        components['risk_adjusted'] = self._calc_loss_severity(profit_dollars)

        # 6. Win retention — premia winners mejores que el promedio histórico
        #    (mapeado a sumaRecompensaMomentum para compatibilidad)
        components['momentum'] = self._calc_win_retention(profit_dollars)

        # 7. Trade quality — ratio avg_win / avg_loss real
        components['trade_quality'] = self._calc_trade_quality()

        # Acumular y ponderar
        total_reward = 0.0
        for name, value in components.items():
            weighted = self.weights.get(name, 0.0) * value
            setattr(self, self._sum_map[name],
                    getattr(self, self._sum_map[name]) + weighted)
            total_reward += weighted

        # Actualizar buffers DESPUÉS de calcular (no contamina el cálculo actual)
        self._update_buffers(profit_dollars, current_equity)

        return total_reward, components

    # ------------------------------------------------------------------
    # Step reward — llamado en cada paso mientras hay posición abierta
    # ------------------------------------------------------------------

    def get_step_reward(self, upnl, steps_in_position, balance):
        """
        Reward por paso mientras hay posición abierta.

        - upnl > 0: pequeña recompensa para mantener winners.
        - upnl < 0: penalización que crece con el tiempo en pérdida.
                    Presiona al agente a cerrar losers en lugar de esperar reversión.

        Parameters
        ----------
        upnl              : float  P&L no realizado en USD
        steps_in_position : int    cuántos pasos lleva abierta la posición
        balance           : float  balance inicial del episodio
        """
        base = float(np.tanh(upnl / (balance * 0.02))) * 0.01

        if upnl < 0:
            # time_factor sube de 1.0 a 2.0 en los primeros 200 steps en pérdida
            time_factor = 1.0 + min(steps_in_position * 0.005, 1.0)
            return base * time_factor
        return base

    # ------------------------------------------------------------------
    # Componentes internos
    # ------------------------------------------------------------------

    def _calc_profit(self, episode_wins, episode_losses):
        """
        Expectancy = accuracy * avg_win - (1-accuracy) * avg_loss
        Normalizada por 1% del balance.
        Requiere al menos un win Y un loss para dar señal.
        """
        if len(self.wins_buffer) == 0 or len(self.losses_buffer) == 0:
            return 0.0

        avg_win  = float(np.mean(self.wins_buffer))
        avg_loss = float(abs(np.mean(self.losses_buffer)))

        if episode_wins is not None and episode_losses is not None:
            total = episode_wins + episode_losses
            accuracy = episode_wins / total if total > 0 else 0.5
        else:
            total = len(self.wins_buffer) + len(self.losses_buffer)
            accuracy = len(self.wins_buffer) / total

        expectancy = (accuracy * avg_win) - ((1.0 - accuracy) * avg_loss)
        return float(np.tanh(expectancy / (self.initial_balance * 0.01)))

    def _calc_sharpe(self, returns_np):
        """Rolling Sharpe sobre últimos 50 trades. Mínimo 15 datos."""
        if len(returns_np) < 15:
            return 0.0
        std = float(np.std(returns_np))
        if std == 0:
            return 0.0
        excess = returns_np - self.risk_free_rate
        sharpe = float(np.mean(excess)) / std
        return float(np.tanh(sharpe / 2.0))

    def _calc_drawdown_delta(self, current_equity, peak_equity):
        """
        Penaliza solo el EMPEORAMIENTO del drawdown entre cierres de trades.
        Si el drawdown se mantiene o mejora → 0.
        Así no acumula penalización en períodos de drawdown persistente.
        """
        if peak_equity <= 0:
            return 0.0
        current_dd = (peak_equity - current_equity) / peak_equity
        delta = current_dd - self._prev_drawdown   # positivo = empeoró
        self._prev_drawdown = current_dd

        if delta <= 0 or current_dd < 0.03:
            return 0.0  # drawdown mejoró o es menor al 3% → sin penalización

        # Penalización proporcional al empeoramiento, con factor extra si DD es severo
        severity = 1.0 + max(current_dd - 0.10, 0) * 5  # DD >10% amplifica
        return float(np.tanh(-delta * 20 * severity))

    def _calc_consistency(self, returns_np):
        """Coeficiente de variación — premia retornos estables."""
        if len(returns_np) < 5:
            return 0.0
        mean_r = float(np.mean(returns_np))
        std_r  = float(np.std(returns_np))
        if std_r == 0:
            return 0.1 if mean_r >= 0 else -0.1
        cv = abs(std_r / mean_r) if mean_r != 0 else float('inf')
        score = 1.0 / (1.0 + cv)
        return score if mean_r >= 0 else -score

    def _calc_loss_severity(self, profit_dollars):
        """
        Penaliza cuando un loser es PEOR que el promedio histórico del buffer.
        Si el loser es igual o mejor que el promedio → 0 (sin señal positiva falsa).

        Corrección v2.1: la versión anterior premiaba losers "normales" (~50% de todos
        los losses son mejores que el promedio por definición) causando acumulación
        positiva de +34 en 1509 losses. Ahora solo penaliza, nunca premia.
        """
        if profit_dollars >= 0 or len(self.losses_buffer) == 0:
            return 0.0

        avg_loss = float(np.mean(self.losses_buffer))   # negativo
        diff = profit_dollars - avg_loss                 # negativo si peor que promedio
        if diff >= 0:
            return 0.0  # loser mejor que el promedio → sin señal
        return float(np.tanh(diff / (self.initial_balance * 0.005)))

    def _calc_win_retention(self, profit_dollars):
        """
        Premia cuando un winner supera el promedio histórico del buffer.
        Si el winner es igual o menor al promedio → 0 (sin señal negativa falsa).

        Corrección v2.1: la versión anterior penalizaba winners "normales" (~50% de todos
        los wins son menores que el promedio) causando acumulación negativa de -14.
        Ahora solo premia, nunca penaliza.
        """
        if profit_dollars <= 0 or len(self.wins_buffer) == 0:
            return 0.0

        avg_win = float(np.mean(self.wins_buffer))
        diff    = profit_dollars - avg_win           # positivo si mejor que promedio
        if diff <= 0:
            return 0.0  # winner menor al promedio → sin señal
        return float(np.tanh(diff / (self.initial_balance * 0.005)))

    def _calc_trade_quality(self):
        """
        Señaliza el CAMBIO en el ratio avg_win/avg_loss entre trades consecutivos,
        no el ratio absoluto.

        Corrección v2.1: el ratio absoluto (ej. 1.05) generaba tanh(0.10) en cada
        uno de los 2902 trades → acumulación de +66 con señal casi neutra.
        Con el delta, si el ratio se mantiene estable en 1.05, la señal es ~0.
        Solo dispara cuando el ratio MEJORA o EMPEORA respecto al trade anterior.

        Ejemplos:
          ratio sube de 1.0 a 1.3 → delta=+0.3 → tanh(6) ≈ +1.0 (señal fuerte positiva)
          ratio baja de 1.2 a 0.9 → delta=-0.3 → tanh(-6) ≈ -1.0 (señal fuerte negativa)
          ratio estable en 1.05  → delta≈0    → ~0 (sin acumulación)
        """
        if len(self.wins_buffer) < 5 or len(self.losses_buffer) < 5:
            return 0.0

        avg_win  = float(np.mean(self.wins_buffer))
        avg_loss = float(abs(np.mean(self.losses_buffer)))
        if avg_loss == 0:
            return 0.0

        ratio = avg_win / avg_loss
        delta = ratio - self._prev_ratio
        self._prev_ratio = ratio
        return float(np.tanh(delta * 20))

    # ------------------------------------------------------------------
    # Buffers y reset
    # ------------------------------------------------------------------

    def _update_buffers(self, profit_dollars, current_equity):
        self.returns_buffer.append(profit_dollars * self._inv_balance)
        self.equity_buffer.append(current_equity)
        if profit_dollars > 0:
            self.wins_buffer.append(profit_dollars)
        elif profit_dollars < 0:
            self.losses_buffer.append(profit_dollars)

    def reset_episode(self):
        """Limpia todos los buffers al inicio de cada episodio."""
        self.returns_buffer.clear()
        self.equity_buffer.clear()
        self.wins_buffer.clear()
        self.losses_buffer.clear()
        self._prev_drawdown = 0.0
        self._prev_ratio    = 1.0


# ------------------------------------------------------------------
# Función de compatibilidad — misma firma que v1
# ------------------------------------------------------------------

def calculate_advanced_reward(reward_system, profit_dollars, current_equity, peak_equity,
                              episode_returns, is_trade_closed=False, add_noise=False,
                              episode_wins=None, episode_losses=None):
    """
    Wrapper de compatibilidad con run_con_per.py.
    add_noise ignorado en v2 (eliminado por diseño).
    """
    reward, components = reward_system.calculate_reward(
        profit_dollars, current_equity, peak_equity,
        episode_returns, is_trade_closed, episode_wins, episode_losses
    )
    return reward, components
