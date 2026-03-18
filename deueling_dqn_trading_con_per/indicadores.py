# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:30:17 2025

@author: fabia
"""
import pandas as pd
import numpy as np

def rsi(data, period=14):
    """RSI usando el método de Wilder vectorizado (ewm con alpha=1/period)."""
    if len(data) < period + 1:
        return pd.Series([50] * len(data), index=data.index)

    delta = data['close'].diff(1)
    up   = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)

    avg_up   = up.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_up / avg_down.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    rsi_values = np.where(avg_down == 0, 100, rsi_values)
    rsi_values = np.where(avg_up   == 0,   0, rsi_values)

    return pd.Series(rsi_values, index=data.index).fillna(50)


def macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Versión más robusta que maneja mejor los datos limitados"""
    
    if len(data) < 2:
        zero_series = pd.Series([0] * len(data), index=data.index)
        return zero_series, zero_series
    
    # Si hay pocos datos, usar períodos más cortos
    actual_fast = min(fast_period, len(data) // 2)
    actual_slow = min(slow_period, len(data) - 1)
    actual_signal = min(signal_period, len(data) // 3)
    
    # Asegurar que fast < slow
    if actual_fast >= actual_slow:
        actual_fast = max(1, actual_slow - 1)
    
    # Calcular EMAs con períodos ajustados
    ema_fast = data['close'].ewm(span=actual_fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=actual_slow, adjust=False).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    if actual_signal > 0:
        signal_line = macd_line.ewm(span=actual_signal, adjust=False).mean()
    else:
        signal_line = macd_line.copy()
    
    # Rellenar NaN de forma inteligente
    # Usar el primer valor válido para rellenar hacia atrás
    macd_filled = macd_line.bfill().fillna(0)
    signal_filled = signal_line.bfill().fillna(0)
    
    return macd_filled, signal_filled

def add_ema200_distance(data, period=200):
    # Calcular EMA
    ema_200 = data['close'].ewm(span=period, adjust=False).mean()
    
    # Calcular distancia
    dist_ema200 = data['close'] - ema_200
    
    # Usar los métodos actualizados en lugar de fillna(method=...)
    return dist_ema200.bfill().fillna(0), ema_200.bfill().fillna(0)

