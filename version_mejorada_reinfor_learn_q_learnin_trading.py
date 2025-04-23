import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
from collections import deque
import sys

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

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def stocks_price_format(n):
    n = float(n)
    if n < 0:
        return "- $ {0:.6f}".format(abs(n))
    else:
        return "$ {0:.6f}".format(abs(n))

def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size + 1
    if starting_id >= 0:
        windowed_data = data.iloc[starting_id:timestep+1].values.flatten()
    else:
        windowed_data = -starting_id * [data.iloc[0].item()] + list(data.iloc[0:timestep+1].values.flatten())
    
    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))
    
    # Asegurar la forma correcta (1, window_size - 1)
    return np.array(state).reshape(1, -1)

def dataset_loader(stock_name, desde, hasta, intervalo):
    dataset = yf.download(stock_name, desde, hasta, interval=intervalo)
    return dataset['Close']

def main():
    # Parámetros
    stock_name = "AAPL"
    desde = "2020-01-01"
    hasta = "2024-01-24"
    intervalo = "1d"
    data = dataset_loader(stock_name, desde, hasta, intervalo)

    # El estado tendrá tamaño window_size - 1 porque calculamos diferencias
    window_size = 11  # Aumentamos en 1 para seguir teniendo 10 elementos en el estado
    state_size = window_size - 1
    episodes = 11
    batch_size = 32
    data_samples = len(data) - 1

    # Configuración para cargar modelo existente
    cargar_modelo = False  # Cambiar a False para empezar con un modelo nuevo
    modelo_existente = "ai_trader_10"  # Nombre del modelo a cargar (sin extensión)

    trader = AI_Trader(state_size)
    
    # Cargar modelo existente si se especifica
    if cargar_modelo:
        try:
            trader.load_model(modelo_existente)
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"  Error al cargar el modelo {modelo_existente}: {str(e)}")
            print("Deteniendo la ejecución...")
            sys.exit(1)

    # Precalcular todos los estados
    states = [state_creator(data, t, window_size) for t in range(data_samples)]

    for episode in range(1, episodes + 1):
        print("Episodio: {}/{}".format(episode, episodes))

        state = states[0]
        total_profit = 0
        trader.inventory = []

        for t in range(data_samples):
            action = trader.trade(state)
            next_state = states[t + 1] if t + 1 < data_samples else state  # Ensure we don't go out of bounds
            reward = 0
            current_price = data.iloc[t].item()

            if action == 1:  # Comprar
                trader.inventory.append(current_price)
                print("episodio: ",episode,"AI Trader compró: ", stocks_price_format(current_price))

            elif action == 2 and len(trader.inventory) > 0:  # Vender
                buy_price = trader.inventory.pop(0)
                reward = max(current_price - buy_price, 0)
                total_profit += current_price - buy_price
                print("AI Trader vendió: ", stocks_price_format(current_price),
                    " Beneficio: " + stocks_price_format(current_price - buy_price))

            done = (t == data_samples - 1)
            trader.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("########################")
                print("BENEFICIO TOTAL: {}".format(stocks_price_format(total_profit)))
                print("########################")

        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

        # Guardar el modelo cada 10 episodios para no sobrecargar el proceso
        if episode % 10 == 0:
            trader.save_model(f"ai_trader_{episode}")

if __name__ == "__main__":
    main()