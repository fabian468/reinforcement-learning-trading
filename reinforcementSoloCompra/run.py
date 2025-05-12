# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 00:25:00 2025
@author: fabia
"""

import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from dotenv import load_dotenv
import os
from sklearn.preprocessing import MinMaxScaler

import cProfile
import pstats

load_dotenv() 


@tf.keras.utils.register_keras_serializable()
def combine_value_and_advantage(inputs):
    value, advantage = inputs
    advantage_mean = tf.keras.backend.mean(advantage, axis=1, keepdims=True)
    return value + (advantage - advantage_mean)

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

class AI_Trader():
    def __init__(self, 
                 state_size, 
                 action_space=3, 
                 model_name="AITrader" ,
                 random_market_event_probability = 0.01,
                 spread= 0.20 , 
                 commission_per_trade= 0.07,
                 gamma = 0.95,
                 epsilon = 0.9,
                 epsilon_final = 0.1,
                 epsilon_decay = 0.9999,
                 use_double_dqn = True,
                 target_model_update = 100,
                 learning_rate  = 0.001
                 ):  
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=5000)
        self.inventory = []
        self.model_name = model_name
        self.reward_noise_std = 0.001
        self.random_market_event_probability = random_market_event_probability
        self.spread = spread
        self.commission_per_trade = commission_per_trade
        self.learning_rate = learning_rate

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.use_double_dqn = use_double_dqn
        self.target_model_update = target_model_update
        self.step_counter = 0
        self.total_rewards = 0

        self.model = self.model_builder()

        if self.use_double_dqn:
            self.target_model = self.model_builder()
            self.target_model.set_weights(self.model.get_weights())

        self.profit_history = []
        self.rewards_history = []
        self.epsilon_history = []
        self.trades_history = []
        self.loss_history = []
        self.drawdown_history = []
        self.sharpe_ratios = []
        self.accuracy_history = []
        self.avg_win_history = []
        self.avg_loss_history = []

    def model_builder(self):
        input_layer = tf.keras.layers.Input(shape=(self.state_size,))
        
        # Capa densa con Batch Normalization
        x = tf.keras.layers.Dense(32)(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization
        x = tf.keras.layers.ReLU()(x)  # Activación ReLU
    
        # Capa adicional
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization
        x = tf.keras.layers.ReLU()(x)  # Activación ReLU
    
        # Otra capa densa con Batch Normalization
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization
        x = tf.keras.layers.ReLU()(x)  # Activación ReLU
        
        # Valor (value stream)
        value_stream = tf.keras.layers.Dense(128)(x)
        value_stream = tf.keras.layers.BatchNormalization()(value_stream)  # Batch Normalization
        value_stream = tf.keras.layers.ReLU()(value_stream)  # Activación ReLU
        value = tf.keras.layers.Dense(1)(value_stream)
        
        # Ventaja (advantage stream)
        advantage_stream = tf.keras.layers.Dense(128)(x)
        advantage_stream = tf.keras.layers.BatchNormalization()(advantage_stream)  # Batch Normalization
        advantage_stream = tf.keras.layers.ReLU()(advantage_stream)  # Activación ReLU
        advantage = tf.keras.layers.Dense(self.action_space)(advantage_stream)
        
        # Combinamos valor y ventaja
        outputs = tf.keras.layers.Lambda(combine_value_and_advantage)([value, advantage])
    
        # Definimos el modelo
        model = tf.keras.models.Model(inputs=input_layer, outputs=outputs)
    
        # Compilamos el modelo
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        
        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
           
        actions = self.model.predict(state, verbose=0)
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        self.step_counter += 1

        states = np.array([item[0] for item in batch])
        if states.ndim > 2:
            states = np.squeeze(states, axis=1)
        next_states = np.array([item[3] for item in batch])
        if next_states.ndim > 2:
            next_states = np.squeeze(next_states, axis=1)
        rewards = np.array([item[2] for item in batch])
        actions = np.array([item[1] for item in batch])
        dones = np.array([item[4] for item in batch])

        if self.use_double_dqn:
            main_q_values = self.model.predict(next_states, verbose=0)
            target_q_values = self.target_model.predict(next_states, verbose=0)
            best_actions = np.argmax(main_q_values, axis=1)
            next_q_values = np.array([target_q_values[i, best_actions[i]] for i in range(batch_size)])

            targets = self.model.predict(states, verbose=0)
            for i in range(batch_size):
                if dones[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    targets[i, actions[i]] = rewards[i] + self.gamma * next_q_values[i]
        else:
            targets = self.model.predict(states, verbose=0)
            next_targets = self.model.predict(next_states, verbose=0)
            for i in range(batch_size):
                if dones[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    targets[i, actions[i]] = rewards[i] + self.gamma * np.amax(next_targets[i])

        history = self.model.fit(states, targets, epochs=1, verbose=0)
        self.loss_history.append(history.history['loss'][0])

        if self.use_double_dqn and self.step_counter % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())
            print(f"Modelo objetivo actualizado en el paso {self.step_counter}")

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
            
            
    def save_model(self, name):
        self.model.save(f"{name}.h5")
        if self.use_double_dqn:
            self.target_model.save(f"{name}_target.h5")
        with open(f"{name}_params.txt", "w") as f:
            f.write(f"epsilon:{self.epsilon}\n")
            f.write(f"step_counter:{self.step_counter}\n")
            f.write(f"use_double_dqn:{self.use_double_dqn}\n")
            f.write(f"reward_noise_std:{self.reward_noise_std}\n")
            f.write(f"random_market_event_probability:{self.random_market_event_probability}\n")
            f.write(f"spread:{self.spread}\n")
            f.write(f"commission_per_trade:{self.commission_per_trade}\n")
        print(f"Modelo guardado como {name}.h5 y parámetros en {name}_params.txt")

    def load_model(self, name):
        try:
            self.model = tf.keras.models.load_model(f"{name}.h5", custom_objects={'combine_value_and_advantage': combine_value_and_advantage}, compile=False)
            self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            if self.use_double_dqn and tf.io.gfile.exists(f"{name}_target.h5"):
                self.target_model = tf.keras.models.load_model(f"{name}_target.h5", custom_objects={'combine_value_and_advantage': combine_value_and_advantage}, compile=False)
                self.target_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
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
                            elif key == "reward_noise_std":
                                self.reward_noise_std = float(value)
                            elif key == "random_market_event_probability":
                                self.random_market_event_probability = float(value)
                            elif key == "spread":
                                self.spread = float(value)
                            elif key == "commission_per_trade":
                                self.commission_per_trade = float(value)
                print(f"Modelo cargado desde {name}.h5 con epsilon = {self.epsilon} y parámetros.")
            else:
                print("Archivo de parámetros no encontrado, manteniendo valores por defecto.")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            print("Manteniendo valores por defecto.")

    def plot_training_metrics(self, save_path='resultados_cv' ):
        min_length = min(len(self.profit_history), len(self.rewards_history),len(self.epsilon_history), len(self.trades_history), len(self.loss_history), len(self.drawdown_history), len(self.sharpe_ratios), len(self.accuracy_history), len(self.avg_win_history), len(self.avg_loss_history))

        episodes = range(1, min_length + 1)

        fig, axs = plt.subplots(4, 2, figsize=(15, 16))  # 4 filas, 2 columnas
        fig.suptitle('Métricas de Entrenamiento', fontsize=16)

        axs[0, 0].plot(episodes, self.profit_history[:min_length], label='Beneficio Total')
        axs[0, 0].set_ylabel('Beneficio')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[0, 1].plot(episodes, self.epsilon_history[:min_length], label='Epsilon', color='red')
        axs[0, 1].set_ylabel('Epsilon')
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        axs[1, 0].plot(episodes, self.loss_history[:min_length], label='Loss', color='purple')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].set_xlabel('Episodio')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        axs[1, 1].plot(episodes, self.drawdown_history[:min_length], label='Drawdown Máximo', color='orange')
        axs[1, 1].set_ylabel('Drawdown')
        axs[1, 1].set_xlabel('Episodio')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        axs[2, 0].plot(episodes, self.sharpe_ratios[:min_length], label='Ratio de Sharpe', color='green')
        axs[2, 0].set_ylabel('Ratio de Sharpe')
        axs[2, 0].set_xlabel('Episodio')
        axs[2, 0].grid(True)
        axs[2, 0].legend()

        axs[2, 1].plot(episodes, self.accuracy_history[:min_length], label='Frecuencia de Aciertos', color='brown')
        axs[2, 1].set_ylabel('Frecuencia de Aciertos')
        axs[2, 1].set_xlabel('Episodio')
        axs[2, 1].grid(True)
        axs[2, 1].legend()
        
        axs[3, 0].plot(episodes, self.rewards_history[:min_length], label='Recompensa por Episodio', color='blue')
        axs[3, 0].set_ylabel('Recompensa')
        axs[3, 0].set_xlabel('Episodio')
        axs[3, 0].grid(True)
        axs[3, 0].legend()
     
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_path, 'training_metrics.png'))
        plt.show()

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def price_format(n):
    n = float(n)
    return "- {0:.6f}".format(abs(n)) if n < 0 else "{0:.6f}".format(abs(n))

def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size + 1
    if starting_id >= 0:
        windowed_data = data.iloc[starting_id:timestep+1].copy()
    else:
        padding = pd.DataFrame([data.iloc[0]] * -starting_id, columns=data.columns, index=data.index[: -starting_id])
        windowed_data = pd.concat([padding, data.iloc[0:timestep+1]]).copy()

    state = []
    # Precio de cierre normalizado
    scaler_close = MinMaxScaler()
    scaled_close = scaler_close.fit_transform(windowed_data['close'].values.reshape(-1, 1))
    for i in range(window_size - 1):
        state.append(scaled_close[i+1][0] - scaled_close[i][0])

    # Volumen normalizado
    scaler_volume = MinMaxScaler()
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

def state_creator_vectorized(data, timestep, window_size):
    starting_id = timestep - window_size + 1
    if starting_id < 0:
        padding = pd.DataFrame([data.iloc[0]] * -starting_id, columns=data.columns, index=data.index[: -starting_id])
        windowed_data = pd.concat([padding, data.iloc[0:timestep+1]])
    else:
        windowed_data = data.iloc[starting_id:timestep+1]

    close_scaled = MinMaxScaler().fit_transform(windowed_data['close'].values.reshape(-1, 1)).flatten()
    volume_scaled = MinMaxScaler().fit_transform(windowed_data['tick_volume'].values.reshape(-1, 1)).flatten()

    state = []
    state.extend(close_scaled[1:] - close_scaled[:-1])
    state.extend(volume_scaled[1:] - volume_scaled[:-1])

    rsi_values = rsi(windowed_data, period=window_size).dropna().values
    state.extend(rsi_values[-window_size:] / 100.0 if len(rsi_values) >= window_size else [0.5] * window_size)

    macd_line, signal_line = macd(windowed_data)
    macd_diff = (macd_line - signal_line).dropna().values
    state.extend(macd_diff[-window_size:] / 10.0 if len(macd_diff) >= window_size else [0.0] * window_size)

    return np.array(state).reshape(1, -1)

def dataset_loader_csv(csv_path):
    try:
        df = pd.read_csv(f"data/{csv_path}", sep='\t')  # Usa tabulador como separador
        df.columns = [col.strip('<>').lower() for col in df.columns]  # Limpia < > y normaliza nombres

        # Verifica si existen las columnas necesarias
        if 'date' not in df.columns:
            print("Error: La columna 'date' es obligatoria.")
            return None

        # Combina fecha y hora si existe la columna 'time'
        if 'time' in df.columns and df['time'].notnull().all():
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str))
        else:
            print("Advertencia: No se encontró columna 'time' o tiene valores nulos. Se usará solo 'date'.")
            df['datetime'] = pd.to_datetime(df['date'])

        df.set_index('datetime', inplace=True)

        # Renombra columna de volumen si existe
        if 'tickvol' in df.columns:
            df.rename(columns={'tickvol': 'tick_volume'}, inplace=True)

        print(f"Datos cargados desde: {csv_path}")

        # Selecciona columnas disponibles
        if 'time' in df.columns and 'close' in df.columns and 'spread' in df.columns and 'tick_volume' in df.columns:
            return df[['time', 'close', 'tick_volume' , 'spread']]
        elif 'close' in df.columns and 'tick_volume' in df.columns:
            return df[['close', 'tick_volume']]
        elif 'close' in df.columns:
            print("Advertencia: La columna 'tick_volume' no se encontró. Se usará tickvol=0.")
            df['tickvol'] = 0
            return df[['close', 'tickvol']]
        else:
            print("Error: El archivo CSV debe contener al menos la columna 'close'.")
            return None

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {csv_path}")
        return None
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return None

    
def plot_trading_session(data, buy_points, sell_points, symbol, timeframe, save_path='resultados_cv'):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['close'].values, label=f'{symbol} - {timeframe} (Precio)')
    ax2 = plt.gca().twinx()
    ax2.bar(data.index, data['tick_volume'].values, color='gray', alpha=0.3, label='Volumen')
    ax2.set_ylabel('Volumen', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    for point in buy_points:
        plt.scatter(point[0], point[1], color='green', s=100, marker='^')
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
    plt.savefig(os.path.join(save_path, f'trading_session_{symbol}_{timeframe}.png'))
    plt.show()

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0.0


#como puedo mejorar la convergencia el 

#comienzo del codigo
def main():
    
    nombre_csv = "prueba.csv"
       
    cargar_modelo = False
    modelo_existente = "resultados_cv/Silver_Solo_Compra_ai_trader_dueling_dqn_3_daily"
    
    
    symbol = "GOLD"
    desde = "2018-01-01"
    hasta = "2024-01-01"
    intervalo = "daily"
    
    
    es_indice = False
    es_forex = False
    es_metal = True
    tick_value = 5  
    pip_multiplier = 10000  # Para el Nasdaq (2 decimales)
    
  
    
    episodes = 30
    batch_size = 128
    epsilon_decay = 0.999
    gamma = 0.95
    cada_cuanto_actualizar = 80
    learning_rate = 0.00001
    window_size = 25
    estados_vectorizados = True


    balance_first = 1000 # dinero inicial
    lot_size = 0.01   
    commission_per_trade = 0
    test_size_ratio = 0.2  # 20% para prueba
    penalty_factor = 2.0
    
    # Para calcular el valor del pip:
    # El volumen del contrato es 20 USD por cada 0.01 lote (tu tamaño de lote) por cada pip.
    pip_value = 20 * lot_size  # 20 USD * 0.01 = 0.20 USD por pip
    
    pip_value_eur_usd = 10 * lot_size
    
    # Creación de carpeta para guardar los resultados
    resultados_dir = 'resultados_cv'
    os.makedirs(resultados_dir, exist_ok=True)

    data = dataset_loader_csv(nombre_csv)
        
    # Calcular los datos para el entramiento y para el test el X e Y
    train_size = int(len(data) * (1 - test_size_ratio))
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()
    
    if 'time' in data.columns and data['time'].notnull().all():
        hora = data['time']
    
    state_size = (window_size - 1) * 2 + window_size + window_size

    # Carga el modelo y ve si cargar uno o crear uno nuevo
    trader = AI_Trader(state_size,
                       epsilon_decay= epsilon_decay,
                       learning_rate= learning_rate,
                       commission_per_trade= commission_per_trade,
                       gamma = gamma,
                       target_model_update = cada_cuanto_actualizar
                       )


    if cargar_modelo:
        try:
            trader.load_model(modelo_existente)
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar el modelo {modelo_existente}: {str(e)}")

    n_folds = 2
    fold_size = len(train_data) // n_folds
    all_fold_metrics = []
    

    # Comienza el entrenamiento del primero fold
    for fold in range(n_folds):
        print(f"\n{'='*30} Fold {fold + 1}/{n_folds} {'='*30}")
        start = fold * fold_size
        end = (fold + 1) * fold_size
        fold_data = train_data.iloc[start:end].copy()
        data_samples = len(fold_data) - 1
        if(estados_vectorizados):
            states = np.array([state_creator_vectorized(fold_data, t, window_size) for t in range(data_samples)]) # Usa la función vectorizada
        else:
            states = [state_creator(fold_data, t, window_size) for t in range(data_samples)]

        # Crea las estadísticas para guardar
        trader.profit_history = []
        trader.epsilon_history = []
        trader.trades_history = []
        trader.loss_history = []
        trader.drawdown_history = []
        trader.sharpe_ratios = []
        trader.accuracy_history = []
        trader.avg_win_history = []
        trader.avg_loss_history = []
        trader.epsilon = 1.0
        trader.step_counter = 0
        trader.total_rewards = 0
        trader.memory.clear()
        

        # Comienza los episodios
        for episode in range(1, episodes + 1):
            print(f"Episodio: {episode}/{episodes}")
            # Crea las estadísticas del episodio
            state = states[0]
            total_profit_pips = 0
            trader.inventory = []
            trades_count = 0
            wins = 0
            losses = 0
            previus_profit = 0.01
            profit_dollars = 0
            profit_dollars_total = 0
            winning_profits_pips = []
            losing_profits_pips = []
            peak_equity = balance_first
            current_equity = balance_first
            drawdown_history_episode = []
            episode_returns_pips = []
            buy_points = []
            sell_points = []

            # Bucle que recorre cada estado en los datos que descargue de mt5
            for t in range(data_samples):
                # La ia toma una decision
                action = trader.trade(state)
                # Siguiente estado de la ia
                next_state = states[t + 1] if t + 1 < data_samples else state
                # Comenzando la recompensa
                reward = 0
                # Precio actual
                current_price = fold_data['close'].iloc[t].item()
                spread = fold_data['spread'].iloc[t].item() * lot_size
                # Indice del precio en el estado actual
                timestamp = fold_data.index[t]

                # Coloca el precio de compra actual con
                buy_price = current_price +  spread / 2 if action == 1 else current_price
                sell_price = current_price - spread / 2 if action == 2 and len(trader.inventory) > 0 else current_price

                # Si la accion de la ia es igua a 1 compra
                if action == 1 and not trader.inventory :  # Comprar
                    # Agrega el precio de compra al inventario
                    trader.inventory.append(buy_price)
                    # Suma al contador de trades
                    trades_count += 1
                    
                    # Calcula cuanto dinero tiene (considerando el lote)
                    # if(current_equity <= 50):
                        # reward = -10
                        # current_equity +=1000
                        #1.22724 * 0.01 * 100000 * (1 + 0.7)
                        #print(current_equity)

                    if episode == episodes and fold == n_folds -1 : buy_points.append((timestamp, buy_price))

                # Si la accion de la ia es igual a 2 y hay un trade abierto vende esa accion
                elif action == 2 and len(trader.inventory) > 0:  # Vender
                    # Toma el precio que compro el activo
                    original_buy_price = trader.inventory.pop(0)
                    # Calcula el profit que obtuvo de la venta de activo (en pips)
                    if(es_indice):
                        profit_pips = (sell_price - original_buy_price) 
                        ticks = profit_pips / 0.25   
                        profit_dollars = ticks * tick_value * (lot_size / 1.0)
                    elif(es_forex):
                        profit_pips = (sell_price - original_buy_price) * pip_multiplier
                        # Calcula la ganancia/pérdida en dólares
                        profit_dollars = profit_pips * pip_value_eur_usd
                    elif(es_metal):
                        profit_pips = (sell_price - original_buy_price) 
                        # Calcula la ganancia/pérdida en dólares
                        profit_dollars = profit_pips * pip_value_eur_usd
                        
                    if profit_pips > 0:
                         delta_profit = profit_pips - previus_profit
                         if(delta_profit > 0):
                             reward = delta_profit  # Recompensa proporcional a la mejora
                         else:
                             reward= -0.5
                    else:
                         reward =  profit_pips
                         
                    #reward = profit_pips
                    # Coloco el profit en la variable (en pips)
                    total_profit_pips += profit_pips
                    # Usando la variable pip_value que ya definiste
                    profit_dollars_total += profit_dollars
                    
                    # Suma al contador de trades
                    trades_count += 1
                    if (es_indice):
                        # Calcula cuanto dinero tiene (considerando el lote)
                        current_equity += profit_dollars * (1 - trader.commission_per_trade) # Asumiendo indices es la base
                    else:
                        current_equity += profit_dollars  * (1 - trader.commission_per_trade) 
                    
                    # Agrega el retorno al retorno del episodio cada profit (en pips)
                    episode_returns_pips.append(profit_pips)
                    # Verifica si el profit salio ganador agrefa uno a wins y agrega el profit a winning_profits (en pips)
                    if profit_pips > 0:
                        wins += 1
                        winning_profits_pips.append(profit_pips)
                    # De lo contrario si sale perdedor se agrega uno a lose y agrega el profit a losing_profits (en pips)
                    else:
                        losses += 1
                        losing_profits_pips.append(profit_pips)
                    if episode == episodes and fold == n_folds -1 : sell_points.append((timestamp, sell_price))
                
                    # Puedes imprimir o guardar la ganancia en dólares si lo deseas
                   #print("")
                    if profit_pips > 0:
                        previus_profit = profit_pips
                    print(f"episodio: {episode} ,recompensa:{reward} , por eleccion de ia  Venta a {sell_price:.5f}, Compra a {original_buy_price:.5f}, Profit (pips): {profit_pips:.2f}, Profit (USD): {profit_dollars:.2f}, total de dinero actual {current_equity}")
                    print("")
                    print(f"suma de recompensa {trader.total_rewards}")
                    print("")
                    
                    
                elif 'time' in data.columns and data['time'].notnull().all() and int(hora[t].split(":")[0]) == 23 and len(trader.inventory) > 0:
                    # Toma el precio que compro el activo
                    original_buy_price = trader.inventory.pop(0)
                    # Calcula el profit que obtuvo de la venta de activo (en pips)
                    if(es_indice):
                        profit_pips = (sell_price - original_buy_price) 
                        ticks = profit_pips / 0.25   
                        profit_dollars = ticks * tick_value * (lot_size / 1.0)
                    elif(es_forex):
                        profit_pips = (sell_price - original_buy_price) * pip_multiplier
                        # Calcula la ganancia/pérdida en dólares
                        profit_dollars = profit_pips * pip_value_eur_usd
                    elif(es_metal):
                        profit_pips = (sell_price - original_buy_price) 
                        # Calcula la ganancia/pérdida en dólares
                        profit_dollars = profit_pips * pip_value_eur_usd
                    # vender forzadamente por cierre de jornada, por ejemplo
                    
    
                    
                    #print(delta_profit)
                    
                    if profit_pips > 0:
                        delta_profit = profit_pips - previus_profit
                        if(delta_profit > 0):
                            reward = delta_profit  # Recompensa proporcional a la mejora
                        else:
                            reward= -0.5
                    else:
                        reward =  profit_pips
                    # La recompensa es el profit ya sea positivo o negativo (en la unidad de precio)
                    #reward = profit_pips
                    # Coloco el profit en la variable (en pips)
                    total_profit_pips += profit_pips
                    # Usando la variable pip_value que ya definiste
                    profit_dollars_total +=  profit_dollars
                    
                    # Suma al contador de trades
                    trades_count += 1
                    if (es_indice):
                        # Calcula cuanto dinero tiene (considerando el lote)
                        current_equity += profit_dollars * (1 - trader.commission_per_trade) # Asumiendo indices es la base
                    else:
                
                        current_equity += profit_dollars  * (1 - trader.commission_per_trade) # Asumiendo EUR es la base
                        
                    # Agrega el retorno al retorno del episodio cada profit (en pips)
                    episode_returns_pips.append(profit_pips)
                    # Verifica si el profit salio ganador agrefa uno a wins y agrega el profit a winning_profits (en pips)
                    if profit_pips > 0:
                        wins += 1
                        winning_profits_pips.append(profit_pips)
                    # De lo contrario si sale perdedor se agrega uno a lose y agrega el profit a losing_profits (en pips)
                    else:
                        losses += 1
                        losing_profits_pips.append(profit_pips)
                    if episode == episodes and fold == n_folds -1 : sell_points.append((timestamp, sell_price))
                
                    # Puedes imprimir o guardar la ganancia en dólares si lo deseas
                   #print("")
                    if profit_pips > 0:
                        previus_profit = profit_pips
                    print(f"episodio: {episode} ,recompensa:{reward}, Venta a {sell_price:.5f}, Compra a {original_buy_price:.5f}, Profit (pips): {profit_pips:.2f}, Profit (USD): {profit_dollars:.2f}, total de dinero actual {current_equity}")
    
                # Seguimos en el bucle de episodes
                # Si el dinero actual es mayor al que empezo se actualiza peak_equity despues del episodio
                # para calcular el drawdown
                if current_equity > peak_equity:
                    peak_equity = current_equity
                # Calculo de drawdown como ($1070 - $1050) / $1070 ≈ 0.0187 o 1.87%.
                drawdown = (peak_equity - current_equity) / peak_equity if peak_equity != 0 else 0
                # Se agrega al array para ver cual es el max drawdown
                drawdown_history_episode.append(drawdown)
                # Recompensa para minimizar el drawdown
                reward -= 0.1 * sigmoid(drawdown * 10) - 0.05
                
                
                # Pequeño ruido aleatorio para mejorar el aprendizaje
                reward += np.random.normal(0, trader.reward_noise_std)
                
                trader.total_rewards += reward
                # Ver si termino el episodio
                done = (t == data_samples - 1)
                trader.memory.append((state, action, reward, next_state, done))
                state = next_state

                if len(trader.memory) > batch_size:
                    #entrenamiento !!!!! 
                    trader.batch_train(batch_size)

            sharpe = calculate_sharpe_ratio(np.array(episode_returns_pips))
            accuracy = wins / trades_count if trades_count > 0 else 0
            avg_win = np.mean(winning_profits_pips) if winning_profits_pips else 0
            avg_loss = np.mean(losing_profits_pips) if losing_profits_pips else 0
            max_drawdown = max(drawdown_history_episode) if drawdown_history_episode else 0
            print("")
            print(f"Fin Episodio {episode}: Beneficio (pips)={total_profit_pips},Beneficio (usd)={price_format(profit_dollars_total)}, Trades={trades_count}, Sharpe={sharpe:.2f}, Drawdown={max_drawdown:.2%}, Accuracy={accuracy:.2%}")
            print(f"total de recompensas: {trader.total_rewards} ")
            print("")
            trader.profit_history.append(total_profit_pips)
            trader.epsilon_history.append(trader.epsilon)
            trader.trades_history.append(trades_count)
            trader.rewards_history.append(trader.total_rewards)
            trader.loss_history.append(np.mean(trader.loss_history[-10:]) if trader.loss_history else 0)
            trader.drawdown_history.append(max_drawdown)
            trader.sharpe_ratios.append(sharpe)
            trader.accuracy_history.append(accuracy)
            trader.avg_win_history.append(avg_win)
            trader.avg_loss_history.append(avg_loss)

        trader.plot_training_metrics(save_path=resultados_dir )
        trader.save_model(os.path.join(resultados_dir, f"Silver_Solo_Compra_ai_trader_dueling_dqn_{fold + 1}_{intervalo}"))
        if buy_points or sell_points:
            plot_trading_session(fold_data, buy_points, sell_points, symbol, intervalo, save_path=resultados_dir)

        fold_metrics = {
            'fold': fold + 1,
            'final_profit_pips': total_profit_pips,
            'total_trades': trades_count,
            'final_sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'final_accuracy': accuracy,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss
        }
        all_fold_metrics.append(fold_metrics)

    print("\n{'='*30} Resultados de Validación Cruzada {'='*30}")
    metrics_df = pd.DataFrame(all_fold_metrics)
    print(metrics_df)
    print("\nPromedio de Métricas:")
    print(metrics_df.mean(numeric_only=True))

    # Evaluación final en el conjunto de prueba (opcional)
    print("\n{'='*30} Evaluación en Conjunto de Prueba {'='*30}")
    if len(test_data) > window_size:
        test_samples = len(test_data) - 1
        if(estados_vectorizados):
            test_states = np.array([state_creator_vectorized(test_data, t, window_size) for t in range(test_samples)]) # Usa la función vectorizada
        else:
            test_states = [state_creator(test_data, t, window_size) for t in range(test_samples)]
        test_profit_pips = 0
        test_inventory = []
        test_trades = 0
        test_profit_dollars = 0
        test_profit_dollars_total = 0
        test_returns_pips = []
        test_peak_equity = balance_first
        test_current_equity = balance_first
        test_drawdown_history = []
        test_buy_points = []
        test_sell_points = []
        wins_test = 0
        losses_test = 0
        winning_profits_test_pips = []
        losing_profits_test_pips = []
        
        if 'time' in test_data.columns and test_data['time'].notnull().all():
            hora_test = test_data['time']
        

        for t in range(test_samples):
            test_action = trader.trade(test_states[t])
            current_price = test_data['close'].iloc[t].item()
            spread = test_data['spread'].iloc[t].item() * lot_size
            timestamp = test_data.index[t]
            
            buy_price_test = current_price + spread / 2 if test_action == 1 else current_price
            sell_price_test = current_price - spread / 2 if test_action == 2 and len(test_inventory) > 0 else current_price

            if test_action == 1 and not test_inventory:
                test_inventory.append(buy_price_test)
                test_trades += 1
                #if(not es_indice):
                   # test_current_equity -= buy_price_test * lot_size * 100000 * (1 + trader.commission_per_trade)
                test_buy_points.append((timestamp, buy_price_test))
                
                
            elif test_action == 2 and len(test_inventory) > 0:
                original_buy_price_test = test_inventory.pop(0)
                
                if (es_indice):
                    profit_test_pips = (sell_price_test - original_buy_price_test) 
                    ticks = profit_pips / 0.25   
                    test_profit_dollars = ticks * tick_value * (lot_size / 1.0)
                elif(es_forex): 
                    profit_test_pips = (sell_price_test - original_buy_price_test) * pip_multiplier
                    test_profit_dollars = profit_test_pips * pip_value_eur_usd
                elif(es_metal):
                    profit_test_pips = (sell_price_test - original_buy_price_test) 
                    test_profit_dollars = profit_test_pips * pip_value_eur_usd
                                    
                test_profit_pips += profit_test_pips
                test_trades += 1
                test_profit_dollars_total +=test_profit_dollars
                if (es_indice):
                    # Calcula cuanto dinero tiene (considerando el lote)
                    test_current_equity += test_profit_dollars * (1 - trader.commission_per_trade)
                else:
                    test_current_equity += test_profit_dollars * (1 - trader.commission_per_trade)
                
                test_returns_pips.append(profit_test_pips)
                if profit_test_pips > 0:
                    wins_test += 1
                    winning_profits_test_pips.append(profit_test_pips)
                else:
                    losses_test += 1
                    losing_profits_test_pips.append(profit_test_pips)
                
                test_sell_points.append((timestamp, sell_price_test))
            
            elif 'time' in test_data.columns and test_data['time'].notnull().all() and int(hora_test[t].split(":")[0]) == 23 and len(trader.inventory) > 0:
                original_buy_price_test = test_inventory.pop(0)
                
                if (es_indice):
                    profit_test_pips = (sell_price_test - original_buy_price_test) 
                    ticks = profit_pips / 0.25   
                    test_profit_dollars = ticks * tick_value * (lot_size / 1.0)
                elif(es_forex): 
                    profit_test_pips = (sell_price_test - original_buy_price_test) * pip_multiplier
                    test_profit_dollars = profit_test_pips * pip_value_eur_usd
                elif(es_metal):
                    profit_test_pips = (sell_price_test - original_buy_price_test) 
                    test_profit_dollars = profit_test_pips * pip_value_eur_usd
                                    
                test_profit_pips += profit_test_pips
                test_trades += 1
                test_profit_dollars_total +=test_profit_dollars
                if (es_indice):
                    # Calcula cuanto dinero tiene (considerando el lote)
                    test_current_equity += test_profit_dollars * (1 - trader.commission_per_trade)
                else:
                    test_current_equity += test_profit_dollars * (1 - trader.commission_per_trade)
                
                test_returns_pips.append(profit_test_pips)
                if profit_test_pips > 0:
                    wins_test += 1
                    winning_profits_test_pips.append(profit_test_pips)
                else:
                    losses_test += 1
                    losing_profits_test_pips.append(profit_test_pips)
                
                test_sell_points.append((timestamp, sell_price_test))


            if test_current_equity > test_peak_equity:
                test_peak_equity = test_current_equity
            test_drawdown = (test_peak_equity - test_current_equity) / test_peak_equity if test_peak_equity != 0 else 0
            test_drawdown_history.append(test_drawdown)

        test_sharpe = calculate_sharpe_ratio(np.array(test_returns_pips))
        test_accuracy = wins_test / test_trades if test_trades > 0 else 0
        test_max_drawdown = max(test_drawdown_history) if test_drawdown_history else 0
        avg_win_test = np.mean(winning_profits_test_pips) if winning_profits_test_pips else 0
        avg_loss_test = np.mean(losing_profits_test_pips) if losing_profits_test_pips else 0

        print(f"Resultados en Prueba: Beneficio (pips)={price_format(test_profit_pips)},Beneficio (dollar) = {test_profit_dollars_total} Trades={test_trades}, Sharpe={test_sharpe:.2f}, Drawdown={test_max_drawdown:.2%}, Accuracy={test_accuracy:.2%}")
        plot_trading_session(test_data, test_buy_points, test_sell_points, symbol, intervalo, save_path=resultados_dir)

def run_main():
    main()

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        run_main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(20) # Muestra las 20 funciones más lentas
    
    
    
    
    
    
    
    
    
    
    
    