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
    tiempo_rsi = 20
    tiempo_macd = 20
    tiempo_ema = 200
    
    # Para el state, usamos solo la ventana especificada
    starting_id = timestep - window_size + 1
    if starting_id < 0:
        padding = pd.DataFrame([data.iloc[0]] * -starting_id, columns=data.columns, index=data.index[: -starting_id])
        windowed_data = pd.concat([padding, data.iloc[0:timestep+1]])
    else:
        windowed_data = data.iloc[starting_id:timestep+1]

    # Escalado de precios y volumen para la ventana
    close_scaled = StandardScaler().fit_transform(windowed_data['close'].values.reshape(-1, 1)).flatten()
    volume_scaled = StandardScaler().fit_transform(windowed_data['tick_volume'].values.reshape(-1, 1)).flatten()
    
    state = []
    state.extend(close_scaled[1:] - close_scaled[:-1])
    state.extend(volume_scaled[1:] - volume_scaled[:-1])


    # RSI: usar datos desde (timestep - tiempo_rsi) hasta timestep
    if timestep >= tiempo_rsi - 1:  # Solo calcular si tenemos suficientes datos
        rsi_start = max(0, timestep - tiempo_rsi)
        rsi_data = data.iloc[rsi_start:timestep+1]
        
        try:
            rsi_values = rsi(rsi_data, period=tiempo_rsi).dropna().values
            if len(rsi_values) >= window_size:
                rsi_to_add = rsi_values[-window_size:] / 100.0
            elif len(rsi_values) > 0:
                padding_needed = window_size - len(rsi_values)
                rsi_to_add = np.concatenate([[0.5] * padding_needed, rsi_values / 100.0])
            else:
                rsi_to_add = [0.5] * window_size
                
        except Exception as e:
            if timestep >= 194:
                print(f"ERROR in RSI calculation: {e}")
            rsi_to_add = [0.5] * window_size
    else:
        rsi_to_add = [0.5] * window_size
    
    state.extend(rsi_to_add)
 

    # MACD: usar datos desde (timestep - tiempo_macd) hasta timestep
    if timestep >= tiempo_macd - 1:  # Solo calcular si tenemos suficientes datos
        macd_start = max(0, timestep - tiempo_macd + 1)
        macd_data = data.iloc[macd_start:timestep+1]
        
        try:
            macd_line, signal_line = macd(macd_data)
            macd_diff = (macd_line - signal_line).dropna().values
            
            if len(macd_diff) >= window_size:
                macd_to_add = macd_diff[-window_size:] / 10.0
            elif len(macd_diff) > 0:
                padding_needed = window_size - len(macd_diff)
                macd_to_add = np.concatenate([[0.0] * padding_needed, macd_diff / 10.0])
            else:
                macd_to_add = [0.0] * window_size
                
        except Exception as e:
            if timestep >= 194:
                print(f"ERROR in MACD calculation: {e}")
            macd_to_add = [0.0] * window_size
    else:
        macd_to_add = [0.0] * window_size

    state.extend(macd_to_add)
    
    # Información temporal
    time_str = data.iloc[timestep]['time']
    hour = int(time_str.split(':')[0])
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    state.extend([hour_sin, hour_cos])
    

    # EMA200: usar datos desde (timestep - tiempo_ema) hasta timestep
    if timestep >= tiempo_ema - 1:
        ema_start = max(0, timestep - tiempo_ema + 1)
        ema_data = data.iloc[ema_start:timestep+1]
    else:
        ema_data = data.iloc[0:timestep+1]  # Usar todos los datos disponibles
    
    try:
        dist_ema200, ema_200 = add_ema200_distance(ema_data)
        
        # Tomar los últimos valores correspondientes al window_size
        if len(dist_ema200) >= window_size:
            dist_ema_window = dist_ema200[-window_size:].fillna(0).values
            ema_200_window = ema_200[-window_size:].bfill().fillna(0).values
        else:
            available_dist = dist_ema200.fillna(0).values
            available_ema = ema_200.bfill().fillna(0).values
            
            padding_needed = window_size - len(available_dist)
            dist_ema_window = np.concatenate([np.zeros(padding_needed), available_dist])
            ema_200_window = np.concatenate([np.zeros(padding_needed), available_ema])
        
        # Escalar los valores de EMA
        dist_scaled = StandardScaler().fit_transform(dist_ema_window.reshape(-1, 1)).flatten()
        ema_scaled = StandardScaler().fit_transform(ema_200_window.reshape(-1, 1)).flatten()
        
    except Exception as e:
        if timestep >= 194:
            print(f"ERROR in EMA calculation: {e}")
        dist_scaled = np.zeros(window_size)
        ema_scaled = np.zeros(window_size)
    
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