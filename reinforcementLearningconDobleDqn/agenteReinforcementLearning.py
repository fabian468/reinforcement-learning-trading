# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:13:02 2025

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
        self.memory = deque(maxlen=1000)  # Reduce memory size for efficiency
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
                print(f"Modelo cargado desde {name}.h5 con epsilon = {self.epsilon}")
            else:
                print(f"Archivo de parámetros no encontrado, manteniendo valores por defecto")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            print(f"Manteniendo valores por defecto")

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
        plt.savefig('training_evolution_detailed.png')
        plt.show()
    
        # Guardar los datos en CSV
        evolution_data = pd.DataFrame({
            'Episode': range(1, min_length + 1),
            'Profit': adjusted_profit_history,
            'Epsilon': adjusted_epsilon_history,
            'Trades': adjusted_trades_history,
            'Loss': adjusted_loss_history
        })
        evolution_data.to_csv('training_evolution_detailed.csv', index=False)
        print("Datos de evolución guardados en 'training_evolution_detailed.csv'")

