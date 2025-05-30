# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:30:17 2025

@author: fabia
"""
import pandas as pd
import numpy as np

def rsi(data, period=14):
    delta = data['close'].diff(1)
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    avg_up = up.rolling(window=period, min_periods=1).mean()
    avg_down = down.rolling(window=period, min_periods=1).mean()
    rs = avg_up / avg_down
    return pd.Series(np.where(avg_down == 0, 100, 100 - (100 / (1 + rs))), index=data.index)

def macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line

def add_ema200_distance(data):
    # Calcula la EMA de 200 periodos
    ema_200 = data['close'].ewm(span=200, adjust=False).mean()
    # Distancia entre el cierre y la EMA200
    
    dist_ema200 = data['close'] - ema_200
    # También puedes normalizar la distancia, si lo deseas
    # data['dist_ema200_pct'] = data['dist_ema200'] / data['ema200'] * 100

    return dist_ema200 , ema_200

