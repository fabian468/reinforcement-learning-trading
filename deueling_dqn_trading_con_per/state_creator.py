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

    time_str = data.iloc[timestep]['time']  # e.g. '01:00:00'
    hour = int(time_str.split(':')[0])  # extrae solo la hora como entero
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    state.extend([hour_sin, hour_cos])
    
    # EMA200 y distancia del close a la EMA200
    dist_ema200, ema_200 = add_ema200_distance(windowed_data)

   # Seleccionar y escalar los últimos valores correspondientes al window
    dist_ema_window = dist_ema200[-window_size:].fillna(0).values
    ema_200_window = ema_200[-window_size:].bfill().fillna(0).values
    
    dist_scaled = StandardScaler().fit_transform(dist_ema_window.reshape(-1, 1)).flatten()
    ema_scaled = StandardScaler().fit_transform(ema_200_window.reshape(-1, 1)).flatten()
    
    state.extend(dist_scaled)
    state.extend(ema_scaled)

    return np.array(state).reshape(1, -1)

#estoy agregando diferencia de close de ahora con el anterior 
#diferencia de volumen actual con el anterior
#el rsi
#el macd
#hora
#diferencia de ema200 con el close actual
#