# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 00:12:39 2025
@author: fabian
"""

import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

class AI_Trader():
    def __init__(self, state_size, action_space=3, model_name="AITrader"):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        self.model = self.model_builder()

    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(self.state_size,)))
        model.add(tf.keras.layers.Dense(units=32, activation='relu'))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        actions = self.model.predict(state, verbose=0)
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        batch = list(self.memory)[-batch_size:]
        for state, action, reward, next_state, done in batch:
            if not done:
                reward += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target = self.model.predict(state, verbose=0)
            target[0][action] = reward
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

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
    return np.array([state])

def dataset_loader(stock_name, desde, hasta, intervalo):
    dataset = yf.download(stock_name, desde, hasta, interval=intervalo)
    return dataset['Close']

# Parámetros
stock_name = "AAPL"
desde = "2020-01-01"
hasta = "2024-01-24"
intervalo = "1d"
data = dataset_loader(stock_name, desde, hasta, intervalo)
data


window_size = 10
episodes = 1000
batch_size = 32
data_samples = len(data) - 1

trader = AI_Trader(window_size)

for episode in range(1, episodes + 1):
    print("Episodio: {}/{}".format(episode, episodes))

    state = state_creator(data, 0, window_size + 1)
    state
    total_profit = 0
    trader.inventory = []

    for t in range(data_samples):
        action = trader.trade(state)
        next_state = state_creator(data, t + 1, window_size + 1)
        reward = 0
        current_price = data.iloc[t].item()

        if action == 1:  # Comprar
            trader.inventory.append(current_price)
            print("AI Trader compró: ", stocks_price_format(current_price))

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

    if episode % 10 == 0:
        trader.model.save("ai_trader_{}.h5".format(episode))
