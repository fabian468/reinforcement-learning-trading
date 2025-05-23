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

from indicadores import  rsi , macd

def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size + 1
    if starting_id >= 0:
        windowed_data = data.iloc[starting_id:timestep+1].copy()
    else:
        padding = pd.DataFrame([data.iloc[0]] * -starting_id, columns=data.columns, index=data.index[: -starting_id])
        windowed_data = pd.concat([padding, data.iloc[0:timestep+1]]).copy()

    state = []
    # Precio de cierre normalizado
    # Precio de cierre normalizado
    scaler_close = StandardScaler()
    scaled_close = scaler_close.fit_transform(windowed_data['close'].values.reshape(-1, 1))
    for i in range(window_size - 1):
        state.append(scaled_close[i+1][0] - scaled_close[i][0])
    
    # Volumen normalizado
    scaler_volume = StandardScaler()
    scaled_volume = scaler_volume.fit_transform(windowed_data['tick_volume'].values.reshape(-1, 1))
    for i in range(window_size - 1):
        state.append(scaled_volume[i+1][0] - scaled_volume[i][0])

    # RSI
    rsi_values = rsi(windowed_data, period=window_size).dropna().values
    if len(rsi_values) == window_size:
        state.extend(rsi_values / 100.0)
    else:
        state.extend([0.5] * window_size) # Padding si no hay suficientes datos

    # MACD
    macd_line, signal_line = macd(windowed_data)
    macd_diff = (macd_line - signal_line).dropna().values
    if len(macd_diff) >= window_size:
        state.extend(macd_diff[-window_size:] / 10.0) # Scaling aproximado
    else:
        state.extend([0.0] * window_size)

    return np.array(state).reshape(1, -1)

#lo que pasa es que me rellena con 0 cuando no tiene suficientes datos para calcular el rsi y el macd 
def state_creator_vectorized(data, timestep, window_size):
    starting_id = timestep - window_size + 1
    if starting_id < 0:
        padding = pd.DataFrame([data.iloc[0]] * -starting_id, columns=data.columns, index=data.index[: -starting_id])
        windowed_data = pd.concat([padding, data.iloc[0:timestep+1]])
    else:
        windowed_data = data.iloc[starting_id:timestep+1]

    close_scaled = StandardScaler().fit_transform(windowed_data['close'].values.reshape(-1, 1)).flatten()
    volume_scaled = StandardScaler().fit_transform(windowed_data['tick_volume'].values.reshape(-1, 1)).flatten()
    
    state = []
    state.extend(close_scaled[1:] - close_scaled[:-1])
    state.extend(volume_scaled[1:] - volume_scaled[:-1])

    rsi_values = rsi(windowed_data, period=window_size).dropna().values
    state.extend(rsi_values[-window_size:] / 100.0 if len(rsi_values) >= window_size else [0.5] * window_size)

    macd_line, signal_line = macd(windowed_data)
    macd_diff = (macd_line - signal_line).dropna().values
    state.extend(macd_diff[-window_size:] / 10.0 if len(macd_diff) >= window_size else [0.0] * window_size)

    return np.array(state).reshape(1, -1)