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
import os

@tf.keras.utils.register_keras_serializable()
def combine_value_and_advantage(inputs):
    value, advantage = inputs
    advantage_mean = tf.keras.backend.mean(advantage, axis=1, keepdims=True)
    return value + (advantage - advantage_mean)

class AI_Trader():
    def __init__(self, state_size, action_space=3, model_name="AITrader"):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=1000)  # Reduce memory size for efficiency
        self.inventory = []
        self.model_name = model_name
        #### MONITORIZACIÓN DE SOBREAJUSTE ####
        self.reward_noise_std = 0.001 # Desviación estándar del ruido en las recompensas
        self.random_market_event_probability = 0.01 # Probabilidad de un evento aleatorio
        self.spread = 0.0001 # Spread simulado (en unidades de precio)
        self.commission_per_trade = 0.00005 # Comisión por operación (como fracción del precio)

        self.peak_equity = 1.0  # Inicializamos la equity máxima alcanzada
        self.current_equity = 1.0 # Inicializamos la equity actual (puede ser un valor base)
        self.drawdown_history = [] # Para guardar el drawdown máximo en cada episodio

        self.gamma = 0.95
        self.epsilon = 0.5
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
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        # Asegurarse de que state tenga la forma correcta (1, state_size)
        if len(state.shape) == 3 and state.shape[1] == 1:
            state = state.reshape(state.shape[0], state.shape[2])

        actions = self.model.predict(state, verbose=0)
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        batch = list(self.memory)[-batch_size:]
        self.step_counter += 1

        # Preparar datos asegurando la forma correcta
        states = np.array([item[0] for item in batch])
        if len(states.shape) == 3 and states.shape[1] == 1:
            states = states.reshape(states.shape[0], states.shape[2])

        next_states = np.array([item[3] for item in batch])
        if len(next_states.shape) == 3 and next_states.shape[1] == 1:
            next_states = next_states.reshape(next_states.shape[0], next_states.shape[2])

        rewards = np.array([item[2] for item in batch])
        actions = np.array([item[1] for item in batch])
        dones = np.array([item[4] for item in batch])

        # Calcular Q-values objetivo
        if self.use_double_dqn:
            # Double DQN: usar modelo principal para seleccionar acciones,
            # pero modelo objetivo para evaluar estas acciones
            main_q_values = self.model.predict(next_states, verbose=0)
            target_q_values = self.target_model.predict(next_states, verbose=0)

            # Seleccionar acciones del modelo principal para el siguiente estado
            best_actions = np.argmax(main_q_values, axis=1)

            # Evaluar esas acciones usando el modelo objetivo
            next_q_values = np.array([target_q_values[i, best_actions[i]]
                                      for i in range(len(batch))])

            # Calcular Q-values objetivo
            target = self.model.predict(states, verbose=0)
            for i in range(len(batch)):
                if dones[i]:
                    target[i, actions[i]] = rewards[i]
                else:
                    target[i, actions[i]] = rewards[i] + self.gamma * next_q_values[i]
        else:
            # DQN estándar
            target = self.model.predict(states, verbose=0)
            next_target = self.model.predict(next_states, verbose=0)

            for i in range(len(batch)):
                if batch[i][4]:  # If done, no future reward
                    target[i][actions[i]] = rewards[i]
                else:
                    target[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_target[i])

        # Entrenar el modelo y guardar pérdida
        history = self.model.fit(states, target, epochs=1, verbose=0)
        self.loss_history.append(history.history['loss'][0])

        # Actualizar modelo objetivo si se usa Double DQN
        if self.use_double_dqn and self.step_counter % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())
            print(f"Modelo objetivo actualizado en el paso {self.step_counter}")

        # Actualizar epsilon
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

    def save_model(self, name):
        """Guarda el modelo y el valor actual de epsilon"""
        self.model.save(f"{name}.h5")
        if self.use_double_dqn:
            self.target_model.save(f"{name}_target.h5")

        with open(f"{name}_params.txt", "w") as f:
            f.write(f"epsilon:{self.epsilon}\n")
            f.write(f"step_counter:{self.step_counter}\n")
            f.write(f"use_double_dqn:{self.use_double_dqn}\n")
            #### MONITORIZACIÓN DE SOBREAJUSTE ####
            f.write(f"reward_noise_std:{self.reward_noise_std}\n")
            f.write(f"random_market_event_probability:{self.random_market_event_probability}\n")
            f.write(f"spread:{self.spread}\n")
            f.write(f"commission_per_trade:{self.commission_per_trade}\n")

        print(f"Modelo guardado como {name}.h5 y parámetros en {name}_params.txt")

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

            # Cargar parámetros
            if tf.io.gfile.exists(f"{name}_params.txt"):
                with open(f"{name}_params.txt", "r") as f:
                    for line in f:
                        if ":" in line:
                            key, value = line.strip().split(":", 1)
                            if key == "epsilon":
                                self.epsilon = float(value)
                            elif key == "step_counter":
                                self.step_counter = int(value)
                            elif key == "use_double_dqn":
                                self.use_double_dqn = value.lower() == "true"
                            #### MONITORIZACIÓN DE SOBREAJUSTE ####
                            elif key == "reward_noise_std":
                                self.reward_noise_std = float(value)
                            elif key == "random_market_event_probability":
                                self.random_market_event_probability = float(value)
                            elif key == "spread":
                                self.spread = float(value)
                            elif key == "commission_per_trade":
                                self.commission_per_trade = float(value)
                print(f"Modelo cargado desde {name}.h5 con epsilon = {self.epsilon} y parámetros de sobreajuste.")
            else:
                print("Archivo de parámetros no encontrado, manteniendo valores por defecto para epsilon y sobreajuste.")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            print("Manteniendo valores por defecto para epsilon y sobreajuste.")

    def plot_training_evolution(self):
        """Grafica la evolución del entrenamiento, garantizando que todas las listas tengan la misma longitud."""
    
        # Asegurar que las listas tengan la misma longitud
        min_length = min(len(self.profit_history), len(self.epsilon_history), len(self.trades_history))
        adjusted_profit_history = self.profit_history[:min_length]
        adjusted_epsilon_history = self.epsilon_history[:min_length]
        adjusted_trades_history = self.trades_history[:min_length]
        adjusted_loss_history = (self.loss_history[:min_length] 
                                 if len(self.loss_history) >= min_length 
                                 else self.loss_history + [None] * (min_length - len(self.loss_history)))
    
        # Crear gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
        # Gráfico superior: Beneficio total y número de operaciones
        color = 'tab:blue'
        ax1.set_ylabel('Beneficio Total', color=color)
        ax1.plot(range(1, min_length + 1), adjusted_profit_history, color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.6)
    
        # Número de operaciones en el mismo gráfico superior
        ax1_2 = ax1.twinx()
        color = 'tab:green'
        ax1_2.set_ylabel('Número de Operaciones', color=color)
        ax1_2.plot(range(1, min_length + 1), adjusted_trades_history, color=color, linestyle='-.')
        ax1_2.tick_params(axis='y', labelcolor=color)
        ax1.set_title('Rendimiento del Trading')
    
        # Gráfico inferior: Epsilon y Loss
        color = 'tab:red'
        ax2.set_xlabel('Episodio')
        ax2.set_ylabel('Valor de Epsilon', color=color)
        ax2.plot(range(1, min_length + 1), adjusted_epsilon_history, color=color, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(True, linestyle='--', alpha=0.6)
    
        # Loss en el mismo gráfico inferior
        if adjusted_loss_history:
            ax2_2 = ax2.twinx()
            color = 'tab:purple'
            ax2_2.set_ylabel('Loss', color=color)
            ax2_2.plot(range(1, min_length + 1), adjusted_loss_history, color=color, alpha=0.7)
            ax2_2.tick_params(axis='y', labelcolor=color)
    
        ax2.set_title('Parámetros de Entrenamiento')
    
        plt.tight_layout()
        plt.savefig('resultados/training_evolution_detailed.png')
        plt.show()
    
        # Guardar los datos en CSV
        evolution_data = pd.DataFrame({
            'Episode': range(1, min_length + 1),
            'Profit': adjusted_profit_history,
            'Epsilon': adjusted_epsilon_history,
            'Trades': adjusted_trades_history,
            'Loss': adjusted_loss_history
        })
        evolution_data.to_csv('resultados/training_evolution_detailed.csv', index=False)
        print("Datos de evolución guardados en 'training_evolution_detailed.csv'")

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
    rates = mt5.copy_rates_range(symbol,timeframe_map[timeframe], desde_dt, hasta_dt)

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
    plt.savefig(f'resultados/trading_session_{symbol}_{timeframe}.png')
    plt.show()

def main():
    # Inicializar MT5
    if not initialize_mt5():
        print("No se pudo inicializar MetaTrader 5. Finalizando programa.")
        sys.exit(1)

    os.makedirs('resultados', exist_ok=True)

    # Parámetros
    symbol = "EURUSD"  # Símbolo como aparece en MT5
    desde = "2023-12-01"
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
    episodes = 2
    batch_size = 32
    data_samples = len(data) - 1

    # Configuración para cargar modelo existente
    cargar_modelo = False  # Cambiar a True para cargar un modelo existente
    modelo_existente = "ai_trader_dueling_dqn_40_4h"  # Nombre del modelo a cargar (sin extensión)

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
    max_drawdown = 0
    current_peak = float('-inf')

    for episode in range(1, episodes + 1):
        print("Episodio: {}/{}".format(episode, episodes))

        state = states[0]
        total_profit = 0
        trader.inventory = []
        trades_count = 0  # Contador de operaciones en este episodio
        episode_drawdown = 0
        current_equity = 0 # Reiniciar la equity al inicio de cada episodio (simulado)
        initial_equity = 10000 # Equity inicial simulada
        current_equity = initial_equity
        peak_equity = initial_equity

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

            #### RECOMPENSA MODIFICADA CON CONCIENCIA DE RIESGO Y RUIDO ####
            profit_so_far_on_position = 0
            if len(trader.inventory) > 0:
                profit_so_far_on_position = current_price - trader.inventory[0]
                reward += 0.001 * sigmoid(profit_so_far_on_position) - 0.0005

            # Simulación de spread al comprar
            buy_price = current_price + trader.spread / 2 if action == 1 else current_price
            # Simulación de spread al vender
            sell_price = current_price - trader.spread / 2 if action == 2 and len(trader.inventory) > 0 else current_price

            if action == 1:  # Comprar
                trader.inventory.append(buy_price)
                trades_count += 1
                # Simulación de comisión al comprar
                current_equity -= buy_price * (1 + trader.commission_per_trade)
                print(f"Episodio: {episode}, AI Trader compró: {price_format(buy_price)}, epsilon{trader.epsilon} ")
                if episode == episodes:
                    buy_points.append((timestamp, buy_price))

            elif action == 2 and len(trader.inventory) > 0:  # Vender
                original_buy_price = trader.inventory.pop(0)
                profit = sell_price - original_buy_price
                reward = profit  # La recompensa principal es el beneficio
                total_profit += profit
                trades_count += 1
                # Simulación de comisión al vender
                current_equity += sell_price * (1 - trader.commission_per_trade)
                print(f"AI Trader vendió: {price_format(sell_price)}, Beneficio: {price_format(profit)} ,  epsilon{trader.epsilon}")
                if episode == episodes:
                    sell_points.append((timestamp, sell_price))

            #### MONITORIZACIÓN DE SOBREAJUSTE - Eventos Aleatorios de Mercado ####
            if random.random() < trader.random_market_event_probability:
                # Simular un evento aleatorio afectando el precio (ejemplo: salto pequeño)
                price_change = np.random.normal(0, 0.0005)
                current_price += price_change
                print(f"Evento de mercado aleatorio: precio cambió en {price_format(price_change)}")

            #### CONCIENCIA DE RIESGO (DRAWDOWN) - Seguimiento de Equity y Drawdown ####
            if current_equity > peak_equity:
                peak_equity = current_equity
            drawdown = (peak_equity - current_equity) / peak_equity if peak_equity != 0 else 0
            episode_drawdown = max(episode_drawdown, drawdown)

            #### CONCIENCIA DE RIESGO (DRAWDOWN) - Incorporación en la Recompensa ####
            reward -= 0.1 * sigmoid(drawdown * 10) - 0.05

            #### MONITORIZACIÓN DE SOBREAJUSTE - Ruido en las Recompensas ####
            reward += np.random.normal(0, trader.reward_noise_std)

            done = (t == data_samples - 1)
            trader.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("########################")
                print(f"BENEFICIO TOTAL: {price_format(total_profit)}")
                print(f"OPERACIONES REALIZADAS: {trades_count}")
                print(f"DRAWDOWN MÁXIMO: {price_format(episode_drawdown * 100)}%")
                print("########################")

                # Guardar datos para graficar evolución
                trader.profit_history.append(total_profit)
                trader.epsilon_history.append(trader.epsilon)
                trader.trades_history.append(trades_count)
                trader.drawdown_history.append(episode_drawdown)

            if len(trader.memory) > batch_size:
                trader.batch_train(batch_size)

        # Guardar el modelo cada 20 episodios para no sobrecargar el proceso
        if episode % 2 == 0:
            trader.save_model(f"resultados/ai_trader_dueling_dqn_{episode}_{intervalo}")

    # Guardar el modelo final
    #trader.save_model(f"ai_trader_dueling_dqn_final_{intervalo}")

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