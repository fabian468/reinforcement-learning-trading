# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 02:38:58 2025

@author: fabia
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 00:57:48 2025

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

class AI_Trader():
    def __init__(self, state_size, action_space=3, model_name="AITrader"):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=1500)  # Reduce memory size for efficiency
        self.inventory = []
        self.model_name = model_name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        self.model = self.model_builder()

        # Para tracking de la evolución del entrenamiento
        self.profit_history = []
        self.epsilon_history = []
        self.trades_history = []  # Para guardar el número de operaciones

    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_shape=(self.state_size,)))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        # Asegurarse de que state tenga la forma correcta (1, state_size)
        if len(state.shape) == 3 and state.shape[1] == 1:
            state = state.reshape(state.shape[0], state.shape[2])

        actions = self.model.predict(state, verbose=0)
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        batch = list(self.memory)[-batch_size:]

        # Preparar datos asegurando la forma correcta
        states = np.array([item[0] for item in batch])
        if len(states.shape) == 3 and states.shape[1] == 1:
            states = states.reshape(states.shape[0], states.shape[2])

        next_states = np.array([item[3] for item in batch])
        if len(next_states.shape) == 3 and next_states.shape[1] == 1:
            next_states = next_states.reshape(next_states.shape[0], next_states.shape[2])

        rewards = np.array([item[2] for item in batch])
        actions = np.array([item[1] for item in batch])

        target = self.model.predict(states, verbose=0)
        next_target = self.model.predict(next_states, verbose=0)

        for i in range(len(batch)):
            if batch[i][4]:  # If done, no future reward
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_target[i])

        self.model.fit(states, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

    def save_model(self, name):
        """Guarda el modelo y el valor actual de epsilon"""
        self.model.save(f"{name}.h5")
        with open(f"{name}_epsilon.txt", "w") as f:
            f.write(str(self.epsilon))
        print(f"Modelo guardado como {name}.h5 y epsilon como {name}_epsilon.txt")

    def load_model(self, name):
        """Carga un modelo guardado y su valor de epsilon"""
        self.model = tf.keras.models.load_model(f"{name}.h5", compile=False)
        # Recompilar el modelo para evitar problemas de serialización
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        try:
            with open(f"{name}_epsilon.txt", "r") as f:
                self.epsilon = float(f.read())
            print(f"Modelo cargado desde {name}.h5 y epsilon = {self.epsilon}")
        except FileNotFoundError:
            print(f"Archivo epsilon no encontrado, manteniendo epsilon = {self.epsilon}")

    def plot_training_evolution(self):
        """Grafica la evolución del entrenamiento"""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Configurar el eje principal para el beneficio total
        color = 'tab:blue'
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Beneficio Total', color=color)
        ax1.plot(range(1, len(self.profit_history) + 1), self.profit_history, color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Crear un segundo eje para epsilon
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Valor de Epsilon', color=color)
        ax2.plot(range(1, len(self.epsilon_history) + 1), self.epsilon_history, color=color, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

        # Crear un tercer eje para el número de operaciones
        ax3 = ax1.twinx()
        # Offset the right spine of ax3
        ax3.spines['right'].set_position(('outward', 60))
        color = 'tab:green'
        ax3.set_ylabel('Número de Operaciones', color=color)
        ax3.plot(range(1, len(self.trades_history) + 1), self.trades_history, color=color, linestyle='-.')
        ax3.tick_params(axis='y', labelcolor=color)

        plt.title('Evolución del Entrenamiento del AI Trader')
        fig.tight_layout()
        plt.savefig('training_evolution.png')
        plt.show()

        # También guardar los datos en un CSV para análisis posterior
        evolution_data = pd.DataFrame({
            'Episode': range(1, len(self.profit_history) + 1),
            'Profit': self.profit_history,
            'Epsilon': self.epsilon_history,
            'Trades': self.trades_history
        })
        evolution_data.to_csv('training_evolution.csv', index=False)
        print("Datos de evolución guardados en 'training_evolution.csv'")

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

def plot_trading_session(data, buy_points, sell_points, symbol, timeframe):
    """Grafica la sesión de trading con puntos de compra y venta"""
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['close'].values, label=f'{symbol} - {timeframe} (Precio)')

    # Graficar el volumen en un segundo eje
    ax2 = plt.gca().twinx()
    ax2.bar(data.index, data['tick_volume'].values, color='gray', alpha=0.3, label='Volumen')
    ax2.set_ylabel('Volumen', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Puntos de compra en verde
    for point in buy_points:
        plt.scatter(point[0], point[1], color='green', s=100, marker='^')

    # Puntos de venta en rojo
    for point in sell_points:
        plt.scatter(point[0], point[1], color='red', s=100, marker='v')

    plt.title(f'Sesión de Trading - {symbol} ({timeframe})')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'trading_session_{symbol}_{timeframe}.png')
    plt.show()

def main():
    # Inicializar MT5
    if not initialize_mt5():
        print("No se pudo inicializar MetaTrader 5. Finalizando programa.")
        sys.exit(1)

    # Parámetros
    symbol = "EURUSD"  # Símbolo como aparece en MT5
    desde = "2020-01-01"
    hasta = "2024-01-01"
    intervalo = "4h"  # Usar "1m", "5m", "15m", "30m", "1h", "4h" o "1d"

    # Cargar datos
    data = dataset_loader_mt5(symbol, desde, hasta, intervalo)

    if data is None or len(data) < 50:  # Verificar que tenemos suficientes datos
        print("No hay suficientes datos para entrenar el modelo")
        mt5.shutdown()
        sys.exit(1)

    # El estado tendrá tamaño (window_size - 1) * número de características (precio y volumen)
    window_size = 11
    state_size = (window_size - 1) * 2
    episodes = 25
    batch_size = 32
    data_samples = len(data) - 1

    # Configuración para cargar modelo existente
    cargar_modelo = False  # Cambiar a True para cargar un modelo existente
    modelo_existente = "ai_trader_10_4h"  # Nombre del modelo a cargar (sin extensión)

    trader = AI_Trader(state_size)

    # Cargar modelo existente si se especifica
    if cargar_modelo:
        try:
            trader.load_model(modelo_existente)
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar el modelo {modelo_existente}: {str(e)}")
            print("Deteniendo la ejecución...")
            mt5.shutdown()
            sys.exit(1)

    # Precalcular todos los estados
    print("Calculando estados...")
    states = [state_creator(data, t, window_size) for t in range(data_samples)]
    print(f"Estados calculados: {len(states)}")

    # Para graficar la sesión de trading
    buy_points = []  # Lista de tuplas (fecha, precio)
    sell_points = []  # Lista de tuplas (fecha, precio)

    for episode in range(1, episodes + 1):
        print("Episodio: {}/{}".format(episode, episodes))

        state = states[0]
        total_profit = 0
        trader.inventory = []
        trades_count = 0  # Contador de operaciones en este episodio

        # Limpiar puntos de compra/venta para cada episodio si queremos visualizar solo el último
        if episode == episodes:  # Solo guardar puntos para el último episodio
            buy_points = []
            sell_points = []

        for t in range(data_samples):
            action = trader.trade(state)
            next_state = states[t + 1] if t + 1 < data_samples else state
            reward = 0
            current_price = data['close'].iloc[t].item()
            timestamp = data.index[t]  # Obtener la fecha/hora del índice

            if action == 1:  # Comprar
                trader.inventory.append(current_price)
                trades_count += 1
                print(f"Episodio: {episode}, AI Trader compró: {price_format(current_price)}")

                # Guardar punto de compra para graficar (solo último episodio)
                if episode == episodes:
                    buy_points.append((timestamp, current_price))

            elif action == 2 and len(trader.inventory) > 0:  # Vender
                buy_price = trader.inventory.pop(0)
                reward = max(current_price - buy_price, 0)
                total_profit += current_price - buy_price
                trades_count += 1
                print(f"AI Trader vendió: {price_format(current_price)}, Beneficio: {price_format(current_price - buy_price)}")

                # Guardar punto de venta para graficar (solo último episodio)
                if episode == episodes:
                    sell_points.append((timestamp, current_price))

            done = (t == data_samples - 1)
            trader.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("########################")
                print(f"BENEFICIO TOTAL: {price_format(total_profit)}")
                print(f"OPERACIONES REALIZADAS: {trades_count}")
                print("########################")

                # Guardar datos para graficar evolución
                trader.profit_history.append(total_profit)
                trader.epsilon_history.append(trader.epsilon)
                trader.trades_history.append(trades_count)

        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

        # Guardar el modelo cada 25 episodios para no sobrecargar el proceso
        if episode % 25 == 0:
            trader.save_model(f"ai_trader_vol_{episode}_{intervalo}")

    # Graficar evolución del entrenamiento
    trader.plot_training_evolution()

    # Graficar la sesión de trading del último episodio
    if buy_points or sell_points:
        plot_trading_session(data, buy_points, sell_points, symbol, intervalo)

    # Cerrar la conexión con MT5 al finalizar
    mt5.shutdown()
    print("Conexión con MetaTrader 5 cerrada")

if __name__ == "__main__":
    main()