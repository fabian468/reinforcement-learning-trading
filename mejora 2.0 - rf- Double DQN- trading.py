
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
import sys
import MetaTrader5 as mt5
from datetime import datetime
import os
from sklearn.preprocessing import MinMaxScaler

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
    def __init__(self, state_size, action_space=3, model_name="AITrader"):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=500)
        self.inventory = []
        self.model_name = model_name
        self.reward_noise_std = 0.001
        self.random_market_event_probability = 0.01
        self.spread = 0.0001
        self.commission_per_trade = 0.00005
        self.sharpe_lookback = 20 # Número de trades para calcular el Sharpe Ratio

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.999

        self.use_double_dqn = True
        self.target_model_update = 100
        self.step_counter = 0

        self.model = self.model_builder()

        if self.use_double_dqn:
            self.target_model = self.model_builder()
            self.target_model.set_weights(self.model.get_weights())

        self.profit_history = []
        self.epsilon_history = []
        self.trades_history = []
        self.loss_history = []
        self.drawdown_history = []
        self.sharpe_ratios = []
        self.accuracy_history = []
        self.avg_win_history = []
        self.avg_loss_history = []
        self.episode_profits = [] # Para calcular el Sharpe Ratio por episodio

    def model_builder(self):
        input_layer = tf.keras.layers.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        value_stream = tf.keras.layers.Dense(64, activation='relu')(x)
        value = tf.keras.layers.Dense(1)(value_stream)

        advantage_stream = tf.keras.layers.Dense(64, activation='relu')(x)
        advantage = tf.keras.layers.Dense(self.action_space)(advantage_stream)

        outputs = tf.keras.layers.Lambda(combine_value_and_advantage)([value, advantage])

        model = tf.keras.models.Model(inputs=input_layer, outputs=outputs)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
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
            f.write(f"sharpe_lookback:{self.sharpe_lookback}\n")
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
                            elif key == "sharpe_lookback":
                                self.sharpe_lookback = int(value)
                print(f"Modelo cargado desde {name}.h5 con epsilon = {self.epsilon} y parámetros.")
            else:
                print("Archivo de parámetros no encontrado, manteniendo valores por defecto.")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            print("Manteniendo valores por defecto.")

    def plot_training_metrics(self, save_path='resultados_cv'):
        min_length = min(len(self.profit_history), len(self.epsilon_history), len(self.trades_history), len(self.loss_history), len(self.drawdown_history), len(self.sharpe_ratios), len(self.accuracy_history), len(self.avg_win_history), len(self.avg_loss_history))

        episodes = range(1, min_length + 1)

        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
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

def initialize_mt5():
    if not mt5.initialize():
        print("Error al inicializar MetaTrader 5")
        print(f"Error: {mt5.last_error()}")
        return False
    else:
        print("MetaTrader 5 inicializado correctamente")
        print(mt5.version())
        return True

def dataset_loader_mt5(symbol, desde, hasta, timeframe):
    desde_dt = datetime.strptime(desde, "%Y-%m-%d")
    hasta_dt = datetime.strptime(hasta, "%Y-%m-%d")
    timeframe_map = {
        "1m": mt5.TIMEFRAME_M1, 
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30, "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4, "1d": mt5.TIMEFRAME_D1,
    }
    if timeframe not in timeframe_map:
        print(f"Intervalo {timeframe} no válido. Opciones: {list(timeframe_map.keys())}")
        return None
    symbols = mt5.symbols_get()
    symbol_names = [s.name for s in symbols]
    if symbol not in symbol_names:
        print(f"Símbolo {symbol} no disponible en MT5. Disponibles: {symbol_names[:10]} ...")
        return None
    rates = mt5.copy_rates_range(symbol, timeframe_map[timeframe], desde_dt, hasta_dt)
    if rates is None or len(rates) == 0:
        print(f"No se pudieron obtener datos para {symbol} en el período.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    print(f"Datos cargados: {len(df)} registros para {symbol} desde {desde} hasta {hasta}")
    return df[['close', 'tick_volume']]

def dataset_loader_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, index_col='time', parse_dates=True)
        print(f"Datos cargados desde: {csv_path}")
        if 'tick_volume' in df.columns and 'close' in df.columns:
            return df[['close', 'tick_volume']]
        elif 'close' in df.columns:
            print("Advertencia: La columna 'tick_volume' no se encontró en el CSV. Se utilizarán datos sin volumen.")
            df['tick_volume'] = 0  # Puedes inicializar el volumen con 0 si no está presente
            return df[['close', 'tick_volume']]
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

def main():
    if not initialize_mt5():
        print("No se pudo inicializar MetaTrader 5. Finalizando.")
        sys.exit(1)

    resultados_dir = 'resultados_cv'
    os.makedirs(resultados_dir, exist_ok=True)

    symbol = "EURUSD"
    desde = "2023-12-01"
    hasta = "2024-01-01"
    intervalo = "5m"
    window_size = 20
    episodes = 20
    batch_size = 64
    test_size_ratio = 0.2  # 20% para prueba

    data = dataset_loader_mt5(symbol, desde, hasta, intervalo)
    if data is None or len(data) < 2 * window_size:
        print("No hay suficientes datos.")
        mt5.shutdown()
        sys.exit(1)

    train_size = int(len(data) * (1 - test_size_ratio))
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()

    state_size = (window_size - 1) * 2 + window_size + window_size
    trader = AI_Trader(state_size)
    cargar_modelo = False
    modelo_existente = "resultados_cv/ai_trader_dueling_dqn_fold_3_4h"

    if cargar_modelo:
        try:
            trader.load_model(modelo_existente)
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar el modelo {modelo_existente}: {str(e)}")

    n_folds = 3
    fold_size = len(train_data) // n_folds
    all_fold_metrics = []

    for fold in range(n_folds):
        print(f"\n{'='*30} Fold {fold + 1}/{n_folds} {'='*30}")
        start = fold * fold_size
        end = (fold + 1) * fold_size
        fold_data = train_data.iloc[start:end].copy()
        data_samples = len(fold_data) - 1
        states = [state_creator(fold_data, t, window_size) for t in range(data_samples)]

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
        trader.memory.clear()
        trader.episode_profits = []

        for episode in range(1, episodes + 1):
            print(f"Episodio: {episode}/{episodes}")
            state = states[0]
            total_profit = 0
            trader.inventory = []
            trades_count = 0
            wins = 0
            losses = 0
            winning_profits = []
            losing_profits = []
            peak_equity = 1.0
            current_equity = 1.0
            drawdown_history_episode = []
            episode_returns = []
            buy_points = []
            sell_points = []

            for t in range(data_samples):
                action = trader.trade(state)
                next_state = states[t + 1] if t + 1 < data_samples else state
                reward = 0
                current_price = fold_data['close'].iloc[t].item()
                timestamp = fold_data.index[t]

                buy_price = current_price + trader.spread / 2 if action == 1 else current_price
                sell_price = current_price - trader.spread / 2 if action == 2 and len(trader.inventory) > 0 else current_price
                profit = 0  # Inicializar profit aquí

                if action == 1:  # Comprar
                    trader.inventory.append(buy_price)
                    trades_count += 1
                    current_equity -= buy_price * (1 + trader.commission_per_trade)
                    if episode == episodes and fold == n_folds -1 : buy_points.append((timestamp, buy_price))

                elif action == 2 and len(trader.inventory) > 0:  # Vender
                    original_buy_price = trader.inventory.pop(0)
                    profit = sell_price - original_buy_price
                    reward = profit
                    total_profit += profit
                    trades_count += 1
                    current_equity += sell_price * (1 - trader.commission_per_trade)
                    episode_returns.append(profit)
                    if profit > 0:
                        wins += 1
                        winning_profits.append(profit)
                    else:
                        losses += 1
                        losing_profits.append(profit)
                    if episode == episodes and fold == n_folds -1 : sell_points.append((timestamp, sell_price))

                if current_equity > peak_equity:
                    peak_equity = current_equity
                drawdown = (peak_equity - current_equity) / peak_equity if peak_equity != 0 else 0
                drawdown_history_episode.append(drawdown)
                reward -= 0.1 * sigmoid(drawdown * 10) - 0.05
                reward += np.random.normal(0, trader.reward_noise_std)

                # Incorporar Sharpe Ratio en la recompensa (opcional)
                if len(trader.episode_profits) >= trader.sharpe_lookback:
                    recent_returns = np.array(trader.episode_profits[-trader.sharpe_lookback:])
                    sharpe = calculate_sharpe_ratio(recent_returns)
                    reward += 0.001 * sharpe # Ajusta la escala según sea necesario

                done = (t == data_samples - 1)
                trader.memory.append((state, action, reward, next_state, done))
                state = next_state
                trader.episode_profits.append(profit)

                if len(trader.memory) > batch_size:
                    trader.batch_train(batch_size)

            sharpe = calculate_sharpe_ratio(np.array(episode_returns))
            accuracy = wins / trades_count if trades_count > 0 else 0
            avg_win = np.mean(winning_profits) if winning_profits else 0
            avg_loss = np.mean(losing_profits) if losing_profits else 0
            max_drawdown = max(drawdown_history_episode) if drawdown_history_episode else 0

            print(f"Fin Episodio {episode}: Beneficio={price_format(total_profit)}, Trades={trades_count}, Sharpe={sharpe:.2f}, Drawdown={max_drawdown:.2%}, Accuracy={accuracy:.2%}")
            trader.profit_history.append(total_profit)
            trader.epsilon_history.append(trader.epsilon)
            trader.trades_history.append(trades_count)
            trader.loss_history.append(np.mean(trader.loss_history[-10:]) if trader.loss_history else 0)
            trader.drawdown_history.append(max_drawdown)
            trader.sharpe_ratios.append(sharpe)
            trader.accuracy_history.append(accuracy)
            trader.avg_win_history.append(avg_win)
            trader.avg_loss_history.append(avg_loss)
            trader.episode_profits = [] # Reiniciar para el siguiente episodio

        trader.plot_training_metrics(save_path=resultados_dir)
        trader.save_model(os.path.join(resultados_dir, f"ai_trader_dueling_dqn_fold_{fold + 1}_{intervalo}"))
        if buy_points or sell_points:
            plot_trading_session(fold_data, buy_points, sell_points, symbol, intervalo, save_path=resultados_dir)

        fold_metrics = {
            'fold': fold + 1,
            'final_profit': total_profit,
            'total_trades': trades_count,
            'final_sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'final_accuracy': accuracy,
            'avg_win': avg_win,
            'avg_loss': avg_loss
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
        test_states = [state_creator(test_data, t, window_size) for t in range(test_samples)]
        test_profit = 0
        test_inventory = []
        test_trades = 0
        test_returns = []
        test_peak_equity = 1.0
        test_current_equity = 1.0
        test_drawdown_history = []
        test_buy_points = []
        test_sell_points = []
        wins_test = 0
        losses_test = 0
        winning_profits_test = []
        losing_profits_test = []

        for t in range(test_samples):
            test_action = trader.trade(test_states[t])
            current_price = test_data['close'].iloc[t].item()
            timestamp = test_data.index[t]

            buy_price_test = current_price + trader.spread / 2 if test_action == 1 else current_price
            sell_price_test = current_price - trader.spread / 2 if test_action == 2 and len(test_inventory) > 0 else current_price
            profit_test = 0 # Inicializar profit_test aquí

            if test_action == 1:
                test_inventory.append(buy_price_test)
                test_trades += 1
                test_current_equity -= buy_price_test * (1 + trader.commission_per_trade)
                test_buy_points.append((timestamp, buy_price_test))
            elif test_action == 2 and len(test_inventory) > 0:
                original_buy_price_test = test_inventory.pop(0)
                profit_test = sell_price_test - original_buy_price_test
                test_profit += profit_test
                test_trades += 1
                test_current_equity += sell_price_test * (1 - trader.commission_per_trade)
                test_returns.append(profit_test)
                if profit_test > 0:
                    wins_test += 1
                    winning_profits_test.append(profit_test)
                else:
                    losses_test += 1
                    losing_profits_test.append(profit_test)
                test_sell_points.append((timestamp, sell_price_test))

            if test_current_equity > test_peak_equity:
                test_peak_equity = test_current_equity
            test_drawdown = (test_peak_equity - test_current_equity) / test_peak_equity if test_peak_equity != 0 else 0
            test_drawdown_history.append(test_drawdown)

        test_sharpe = calculate_sharpe_ratio(np.array(test_returns))
        test_accuracy = wins_test / test_trades if test_trades > 0 else 0
        test_max_drawdown = max(test_drawdown_history) if test_drawdown_history else 0
        avg_win_test = np.mean(winning_profits_test) if winning_profits_test else 0
        avg_loss_test = np.mean(losing_profits_test) if losing_profits_test else 0

        print(f"Resultados en Prueba: Beneficio={price_format(test_profit)}, Trades={test_trades}, Sharpe={test_sharpe:.2f}, Drawdown={test_max_drawdown:.2%}, Accuracy={test_accuracy:.2%}")
        plot_trading_session(test_data, test_buy_points, test_sell_points, symbol, intervalo, save_path=resultados_dir)

    mt5.shutdown()
    print("Conexión con MetaTrader 5 cerrada")

if __name__ == "__main__":
    main()