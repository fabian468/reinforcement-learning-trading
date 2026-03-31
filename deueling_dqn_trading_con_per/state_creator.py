# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:33:15 2025

@author: fabia
"""

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler


load_dotenv() 

from indicadores import  rsi , macd , add_ema200_distance

#timestep = 1 window_size = 15
#lo que pasa es que me rellena con 0 cuando no tiene suficientes datos para calcular el rsi y el macd 

def state_creator_ohcl_vectorized(data, timestep, window_size, scaler):
    starting_id = timestep - window_size + 1

    if starting_id < 0:
        padding = pd.DataFrame([data.iloc[0]] * -starting_id, columns=data.columns, index=data.index[: -starting_id])
        windowed_data = pd.concat([padding, data.iloc[0:timestep+1]])
    else:
        windowed_data = data.iloc[starting_id:timestep+1]

    features = windowed_data[['open', 'high', 'low', 'close', 'tick_volume']].values
    features_scaled = scaler.transform(features)

    state = features_scaled.flatten().tolist()

    # Hora codificada
    time_obj = pd.to_datetime(data.iloc[timestep]['time'])
    hour = time_obj.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    state.extend([hour_sin, hour_cos])

    return np.array(state).reshape(1, -1)


def create_all_states_ohcl(data, window_size, scaler, hora_int):
    """
    Versión vectorizada: crea todos los estados del fold de una vez.
    ~100x más rápido que el list comprehension con state_creator_ohcl_vectorized.

    Args:
        data       : DataFrame con open/high/low/close/tick_volume
        window_size: tamaño de la ventana temporal
        scaler     : StandardScaler ya ajustado al fold
        hora_int   : np.ndarray int con la hora (0-23) de cada fila, len == len(data)

    Returns:
        np.ndarray float32 de shape (len(data), window_size*5 + 2)
        Cada fila es el estado del timestep correspondiente.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    N = len(data)
    features = data[['open', 'high', 'low', 'close', 'tick_volume']].values
    features_scaled = scaler.transform(features).astype(np.float32)  # (N, 5)

    # Padding: repetir primera fila (window_size-1) veces para los primeros timesteps
    padding = np.tile(features_scaled[0:1], (window_size - 1, 1))
    padded = np.vstack([padding, features_scaled])  # (N + window_size - 1, 5)

    # Ventanas deslizantes: (N, 1, window_size, 5) → (N, window_size*5)
    windows = sliding_window_view(padded, (window_size, 5)).reshape(N, window_size * 5)

    # Hora codificada como seno/coseno
    h = hora_int.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * h / 24).reshape(-1, 1)
    hour_cos = np.cos(2 * np.pi * h / 24).reshape(-1, 1)

    return np.hstack([windows, hour_sin, hour_cos])  # (N, window_size*5 + 2)


# ==============================================================================
# NUEVA FUNCIÓN: Estado avanzado con OHLC + RSI + MACD + DÍA + HORA
# ==============================================================================

def create_all_states_advanced(data, window_size, scaler, hora_int):
    """
    Estado avanzado que incluye:
    - OHLC (5 features por timestep)
    - RSI (1 feature)
    - MACD (3 features: MACD line, Signal line, Histogram)
    - Día de la semana (2 features: sin/cos)
    - Hora del día (2 features: sin/cos)

    Args:
        data       : DataFrame con OHLCV + columna 'time'
        window_size: tamaño de la ventana temporal
        scaler     : StandardScaler ya ajustado para OHLCV
        hora_int   : np.ndarray int con la hora (0-23) de cada fila

    Returns:
        np.ndarray float32 de shape (N, features_totales)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from indicadores import rsi, macd

    N = len(data)

    # =========================================================================
    # 1. OHLCV normalizado (window_size * 5 features)
    # =========================================================================
    features_ohlc = data[['open', 'high', 'low', 'close', 'tick_volume']].values
    ohlc_scaled = scaler.transform(features_ohlc).astype(np.float32)

    # Padding para los primeros timesteps
    padding_ohlc = np.tile(ohlc_scaled[0:1], (window_size - 1, 1))
    padded_ohlc = np.vstack([padding_ohlc, ohlc_scaled])
    windows_ohlc = sliding_window_view(padded_ohlc, (window_size, 5)).reshape(N, window_size * 5)

    # =========================================================================
    # 2. RSI (1 feature: valor actual)
    # =========================================================================
    try:
        rsi_values = rsi(data, period=14).fillna(50).values  # 50 = neutral
    except:
        rsi_values = np.full(N, 50.0)
    rsi_normalized = (rsi_values / 100.0).astype(np.float32).reshape(-1, 1)  # Normalizar a [0, 1]

    # =========================================================================
    # 3. MACD (3 features: MACD line, Signal line, Histogram)
    # =========================================================================
    try:
        macd_line, signal_line = macd(data, fast_period=12, slow_period=26, signal_period=9)
        macd_line = macd_line.fillna(0).values
        signal_line = signal_line.fillna(0).values
        histogram = macd_line - signal_line
    except:
        macd_line = np.zeros(N)
        signal_line = np.zeros(N)
        histogram = np.zeros(N)

    # Normalizar MACD (asumiendo rango típico -2 a 2)
    macd_norm = (macd_line / 10.0).astype(np.float32).reshape(-1, 1)
    signal_norm = (signal_line / 10.0).astype(np.float32).reshape(-1, 1)
    hist_norm = (histogram / 10.0).astype(np.float32).reshape(-1, 1)

    # =========================================================================
    # 4. Día de la semana (2 features: sin/cos)
    # =========================================================================
    # Asumiendo que data.index es DatetimeIndex
    if hasattr(data.index, 'dayofweek'):
        day_of_week = data.index.dayofweek.values
    else:
        # Si no hay DatetimeIndex, intentar parsear desde alguna columna
        day_of_week = np.zeros(N)

    day_sin = np.sin(2 * np.pi * day_of_week / 7).astype(np.float32).reshape(-1, 1)
    day_cos = np.cos(2 * np.pi * day_of_week / 7).astype(np.float32).reshape(-1, 1)

    # =========================================================================
    # 5. Hora del día (2 features: sin/cos)
    # =========================================================================
    h = hora_int.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * h / 24).reshape(-1, 1)
    hour_cos = np.cos(2 * np.pi * h / 24).reshape(-1, 1)

    # =========================================================================
    # Combinar todos los features
    # =========================================================================
    # state = [windows_ohlc, rsi, macd_line, signal_line, histogram, day_sin, day_cos, hour_sin, hour_cos]
    state = np.hstack([
        windows_ohlc,       # window_size * 5
        rsi_normalized,      # 1
        macd_norm,           # 1
        signal_norm,         # 1
        hist_norm,           # 1
        day_sin,             # 1
        day_cos,             # 1
        hour_sin,            # 1
        hour_cos             # 1
    ])  # Total: window_size*5 + 8

    return state.astype(np.float32)


def get_state_size_advanced(window_size):
    """
    Calcula el tamaño del estado para la función create_all_states_advanced.

    Returns:
        int: Tamaño del estado
    """
    # OHLC * window + RSI + MACD_line + Signal + Histogram + dia_sin + dia_cos + hora_sin + hora_cos
    return window_size * 5 + 1 + 3 + 2 + 2



