# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:44:38 2025

@author: fabia
"""

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from dotenv import load_dotenv
import os

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
