# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:28:51 2025

@author: fabia
"""
import pandas as pd
import numpy as np

from utilsAgenteReinforcementLearning import sigmoid



def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size + 1
    if starting_id >= 0:
        windowed_data = data.iloc[starting_id:timestep+1]
    else:
        # Manejar el caso inicial rellenando con el primer valor
        padding = pd.DataFrame([data.iloc[0]] * -starting_id, columns=data.columns, index=data.index[: -starting_id])
        windowed_data = pd.concat([padding, data.iloc[0:timestep+1]])

    state = []
    # Incorporar cambios en el precio de cierre
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data['close'].iloc[i+1] - windowed_data['close'].iloc[i]))

    # Incorporar cambios en el volumen
    for i in range(window_size - 1):
        if windowed_data['tick_volume'].iloc[i] != 0:
            state.append(sigmoid(windowed_data['tick_volume'].iloc[i+1] / windowed_data['tick_volume'].iloc[i] - 1))
        else:
            state.append(0) # Manejar el caso de volumen cero

    # El tamaño del estado ahora será (window_size - 1) * número de características
    return np.array(state).reshape(1, -1)
