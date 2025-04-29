# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:48:30 2025

@author: fabia
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 02:38:58 2025

@author: fabia
"""

import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import sys
import MetaTrader5 as mt5
from datetime import datetime

@tf.keras.utils.register_keras_serializable()
def combine_value_and_advantage(inputs):
    value, advantage = inputs
    advantage_mean = tf.keras.backend.mean(advantage, axis=1, keepdims=True)
    return value + (advantage - advantage_mean)

class AI_Trader():
    def __init__(self, state_size, action_space=3, model_name="AITrader"):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=4000)  # Reduce memory size for efficiency
        self.inventory = []
        self.model_name = model_name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        # Añadido para Double DQN
        self.use_double_dqn = True  # Activa el Double DQN
        self.target_model_update = 100  # Actualizar el modelo objetivo cada 100 pasos
        self.step_counter = 0  # Contador para actualizar el modelo objetivo

        self.model = self.model_builder()

        # Modelo objetivo para Double DQN
        if self.use_double_dqn:
            self.target_model = self.model_builder()
            self.target_model.set_weights(self.model.get_weights())  # Inicializar con los mismos pesos

        # Para tracking de la evolución del entrenamiento
        self.profit_history = []
        self.epsilon_history = []
        self.trades_history = []  # Para guardar el número de operaciones
        self.loss_history = []  # Para guardar las pérdidas del entrenamiento

    def model_builder(self):
        """
        Implementa la arquitectura Dueling DQN usando capas Lambda para operaciones de Keras
        """
        # Capa de entrada
        input_layer = tf.keras.layers.Input(shape=(self.state_size,))

        # Capas compartidas
        x = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        # Stream de valor (estima el valor del estado)
        value_stream = tf.keras.layers.Dense(64, activation='relu')(x)
        value = tf.keras.layers.Dense(1)(value_stream)

        # Stream de ventaja (estima la ventaja de cada acción)
        advantage_stream = tf.keras.layers.Dense(64, activation='relu')(x)
        advantage = tf.keras.layers.Dense(self.action_space)(advantage_stream)

        # Combinar ambos streams usando Layer Lambda para operaciones de Keras
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        outputs = tf.keras.layers.Lambda(combine_value_and_advantage)([value, advantage])

        model = tf.keras.models.Model(inputs=input_layer, outputs=outputs)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        return model

    def trade(self, state):
        actions = self.model.predict(state, verbose=0)
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        # El backtesting no requiere entrenamiento
        pass

    def load_model(self, name):
        """Carga un modelo guardado y sus parámetros"""
        self.model = tf.keras.models.load_model(f"{name}.h5", custom_objects={'combine_value_and_advantage': combine_value_and_advantage}, compile=False)
        # Recompilar el modelo para evitar problemas de serialización
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        try:
            # Cargar target model si existe
            if self.use_double_dqn and tf.io.gfile.exists(f"{name}_target.h5"):
                self.target_model = tf.keras.models.load_model(f"{name}_target.h5", custom_objects={'combine_value_and_advantage': combine_value_and_advantage}, compile=False)
                self.target_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

            # Cargar parámetros (solo epsilon es relevante para la estrategia cargada)
            if tf.io.gfile.exists(f"{name}_params.txt"):
                with open(f"{name}_params.txt", "r") as f:
                    for line in f:
                        if ":" in line:
                            key, value = line.strip().split(":", 1)
                            if key == "epsilon":
                                self.epsilon = float(value)
                print(f"Modelo cargado desde {name}.h5 con epsilon = {self.epsilon}")
            else:
                print(f"Archivo de parámetros no encontrado, manteniendo valores por defecto")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            print(f"Manteniendo valores por defecto")

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def price_format(n):
    n = float(n)
    if n < 0:
        return "- {0:.6f}".format(abs(n))
    else:
        return "{0:.6f}".format(abs(n))

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

def initialize_mt5():
    """Inicializa la conexión con MetaTrader 5"""
    if not mt5.initialize():
        print("Error al inicializar MetaTrader 5")
        print(f"Error: {mt5.last_error()}")
        return False
    else:
        print("MetaTrader 5 inicializado correctamente")
        # Mostrar información sobre la versión de MetaTrader 5
        print(mt5.version())
        return True

def dataset_loader_mt5(symbol, desde, hasta, timeframe):
    """Carga datos históricos desde MetaTrader 5 incluyendo el volumen"""

    # Convertir parámetros de fecha a formato datetime
    desde_dt = datetime.strptime(desde, "%Y-%m-%d")
    hasta_dt = datetime.strptime(hasta, "%Y-%m-%d")

    # Mapeo de intervalos a constantes MT5
    timeframe_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }

    # Verificar que el intervalo es válido
    if timeframe not in timeframe_map:
        print(f"Intervalo {timeframe} no válido. Opciones disponibles: {list(timeframe_map.keys())}")
        return None

    # Verificar que el símbolo existe en MT5
    symbols = mt5.symbols_get()
    symbol_names = [s.name for s in symbols]
    if symbol not in symbol_names:
        print(f"El símbolo {symbol} no está disponible en MetaTrader 5")
        print("Símbolos disponibles:", symbol_names[:10], "...")
        return None

    # Obtener datos históricos
    rates = mt5.copy_rates_range(symbol, timeframe_map[timeframe], desde_dt, hasta_dt)

    if rates is None or len(rates) == 0:
        print(f"No se pudieron obtener datos para {symbol} en el período especificado")
        return None

    # Crear DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    print(f"Datos cargados: {len(df)} registros para {symbol} desde {desde} hasta {hasta}")

    # Devolver las columnas de cierre y volumen
    return df[['close', 'tick_volume']]

def plot_backtesting_results(profit_history, trades_history, data_index, buy_points, sell_points, symbol, timeframe):
    """Grafica los resultados del backtesting."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Gráfico superior: Beneficio acumulado
    ax1.plot(data_index, np.cumsum(profit_history), color='blue', label='Beneficio Acumulado')
    ax1.set_ylabel('Beneficio Acumulado')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Gráfico inferior: Número de operaciones acumuladas
    ax2.plot(data_index[1:], np.cumsum(trades_history), color='green', label='Operaciones Acumuladas')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Operaciones Acumuladas')
    ax2.grid(True)
    ax2.legend(loc='upper left')

    plt.title(f'Resultados del Backtesting - {symbol} ({timeframe})')
    plt.tight_layout()
    plt.savefig(f'backtesting_results_{symbol}_{timeframe}.png')
    plt.show()

    # Graficar la sesión de trading con puntos de compra y venta
    plt.figure(figsize=(14, 7))
    plt.plot(data_index, data['close'].values, label=f'{symbol} - {timeframe} (Precio)')

    # Puntos de compra en verde
    for point in buy_points:
        plt.scatter(point[0], point[1], color='green', s=100, marker='^')

    # Puntos de venta en rojo
    for point in sell_points:
        plt.scatter(point[0], point[1], color='red', s=100, marker='v')

    plt.title(f'Sesión de Trading (Backtesting) - {symbol} ({timeframe})')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'backtesting_trades_{symbol}_{timeframe}.png')
    plt.show()

def backtest(model, data, window_size):
    """Realiza el backtesting del modelo."""
    data_samples = len(data) - 1
    states = [state_creator(data, t, window_size) for t in range(data_samples)]
    inventory = []
    total_profit = 0
    trades_history = []
    profit_history = []
    buy_points = []
    sell_points = []

    for t in range(data_samples):
        state = states[t]
        action = model.trade(state)
        current_price = data['close'].iloc[t].item()
        timestamp = data.index[t]

        trade_profit = 0

        if action == 1:  # Comprar
            inventory.append(current_price)
            trades_history.append(1) # Registrar una operación
            print(f"Backtest: Comprando en {timestamp}, Precio: {price_format(current_price)}")
            buy_points.append((timestamp, current_price))
        elif action == 2 and inventory:  # Vender
            buy_price = inventory.pop(0)
            profit = current_price - buy_price
            total_profit += profit
            trade_profit = profit
            trades_history.append(1) # Registrar una operación
            print(f"Backtest: Vendiendo en {timestamp}, Precio: {price_format(current_price)}, Beneficio: {price_format(profit)}")
            sell_points.append((timestamp, current_price))
        else:
            trades_history.append(0) # No operación

        profit_history.append(trade_profit)

    print("########################")
    print(f"BENEFICIO TOTAL DEL BACKTESTING: {price_format(total_profit)}")
    print(f"OPERACIONES TOTALES EN BACKTESTING: {sum(trades_history)}")
    print("########################")

    return profit_history, trades_history, data.index, buy_points, sell_points

def main():
    # Inicializar MT5
    if not initialize_mt5():
        print("No se pudo inicializar MetaTrader 5. Finalizando programa.")
        sys.exit(1)

    # Parámetros para cargar el modelo y los datos de backtesting
    symbol = "EURUSD"  # Símbolo como aparece en MT5
    desde_backtest = "2024-01-01"  # Fecha de inicio para el backtesting (después de los datos de entrenamiento)
    hasta_backtest = "2025-04-23"  # Fecha de fin para el backtesting (fecha actual o futura)
    intervalo = "4h"  # Intervalo de tiempo del modelo entrenado
    modelo_para_backtest = "ai_trader_dueling_dqn_40_4h" # Cambia al mejor modelo entrenado que quieras probar
    window_size = 11
    state_size = (window_size - 1) * 2

    # Cargar datos para el backtesting
    backtest_data = dataset_loader_mt5(symbol, desde_backtest, hasta_backtest, intervalo)

    if backtest_data is None or len(backtest_data) < window_size:
        print("No hay suficientes datos para realizar el backtesting.")
        mt5.shutdown()
        sys.exit(1)

    # Crear una instancia del AI Trader (solo para cargar el modelo)
    trader = AI_Trader(state_size)

    # Cargar el modelo entrenado
    try:
        trader.load_model(modelo_para_backtest)
        print(f"Modelo {modelo_para_backtest} cargado exitosamente para el backtesting.")
    except Exception as e:
        print(f"Error al cargar el modelo {modelo_para_backtest}: {str(e)}")
        mt5.shutdown()
        sys.exit(1)

    # Realizar el backtesting
    profit_history, trades_history, data_index, buy_points, sell_points = backtest(trader, backtest_data, window_size)

    # Graficar los resultados del backtesting
    plot_backtesting_results(profit_history, trades_history, data_index, buy_points, sell_points, symbol, intervalo)

    # Cerrar la conexión con MT5 al finalizar
    mt5.shutdown()
    print("Conexión con MetaTrader 5 cerrada.")

if __name__ == "__main__":
    main()