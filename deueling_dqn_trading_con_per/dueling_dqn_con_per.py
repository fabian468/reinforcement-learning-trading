
"""
Created on Mon May 12 10:44:38 2025

@author: fabia
"""

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from SumTree_class import SumTree

@tf.keras.utils.register_keras_serializable()
def combine_value_and_advantage(inputs):
    value, advantage = inputs
    advantage_mean = tf.keras.backend.mean(advantage, axis=1, keepdims=True)
    return value + (advantage - advantage_mean)


class AI_Trader_per():
    def __init__(self,
                 state_size,
                 action_space=3,
                 model_name="AITrader",
                 random_market_event_probability=0.01,
                 spread=0.20,
                 commission_per_trade=0.07,
                 gamma=0.95,
                 epsilon=0.9,
                 epsilon_final=0.2,
                 epsilon_decay=0.9999,
                 use_double_dqn=True,
                 target_model_update=100,
                 learning_rate=0.001,
                 memory_size=10000, # Tamaño de la memoria
                 alpha=0.6,        # Hiperparámetro para la priorización
                 beta_start=0.4,   # Hiperparámetro para la corrección de importancia
                 beta_frames=100000, # Número de frames para alcanzar beta=1
                 epsilon_priority=1e-6 # Pequeña constante para evitar probabilidad cero
                 ):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = SumTree(memory_size) # Usamos SumTree ahora
        self.memory_size = memory_size
        self.inventory = []
        self.model_name = model_name
        self.reward_noise_std = 0.01
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

        # Hiperparámetros PER
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.epsilon_priority = epsilon_priority
        
    def model_builder(self):
        input_layer = tf.keras.layers.Input(shape=(self.state_size,))
        
        # Normalización de entrada - añadimos esta capa para normalizar los inputs
        x = tf.keras.layers.LayerNormalization()(input_layer)
        
        # Capa densa con Batch Normalization y regularización L2
        x = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization
        x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)  # Activación LeakyReLU con pendiente 0.1 para valores negativos
        x = tf.keras.layers.Dropout(0.2)(x)  # Agregamos Dropout con tasa del 20%
    
        # Capa adicional con regularización L2
        x = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization
        x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)  # Activación LeakyReLU con pendiente 0.1 para valores negativos
        x = tf.keras.layers.Dropout(0.2)(x)  # Agregamos Dropout con tasa del 20%
        
        x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization
        x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)  # Activación LeakyReLU con pendiente 0.1 para valores negativos
        x = tf.keras.layers.Dropout(0.2)(x)  # Agregamos Dropout con tasa del 20%
    
        x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization
        x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)  # Activación LeakyReLU con pendiente 0.1 para valores negativos
        x = tf.keras.layers.Dropout(0.2)(x)  # Agregamos Dropout con tasa del 20%
        
        x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization
        x = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)  # Activación LeakyReLU con pendiente 0.1 para valores negativos
        x = tf.keras.layers.Dropout(0.2)(x)  # Agregamos Dropout con tasa del 20%
    
        # Valor (value stream) con regularización L2
        value_stream = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        value_stream = tf.keras.layers.BatchNormalization()(value_stream)  # Batch Normalization
        value_stream = tf.keras.layers.LeakyReLU(negative_slope=0.1)(value_stream)  # Activación LeakyReLU con pendiente 0.1
        value_stream = tf.keras.layers.Dropout(0.2)(value_stream)  # Agregamos Dropout con tasa del 20%
        value = tf.keras.layers.Dense(1)(value_stream)
        
        # Ventaja (advantage stream) con regularización L2
        advantage_stream = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        advantage_stream = tf.keras.layers.BatchNormalization()(advantage_stream)  # Batch Normalization
        advantage_stream = tf.keras.layers.LeakyReLU(negative_slope=0.1)(advantage_stream)  # Activación LeakyReLU con pendiente 0.1
        advantage_stream = tf.keras.layers.Dropout(0.2)(advantage_stream)  # Agregamos Dropout con tasa del 20%
        advantage = tf.keras.layers.Dense(self.action_space)(advantage_stream)
        
        # Combinamos valor y ventaja
        outputs = tf.keras.layers.Lambda(combine_value_and_advantage)([value, advantage])
    
        # Definimos el modelo
        model = tf.keras.models.Model(inputs=input_layer, outputs=outputs)
    
        # Compilamos el modelo
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        
        return model

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon_priority) ** self.alpha

    def remember(self, state, action, reward, next_state, done):
        # state ya debería tener la forma (1, state_size)
        q_value = self.model.predict(state, verbose=0)[0][action]
        if done:
            target_q = reward
        else:
            if state.ndim == 1:
                state = np.expand_dims(state, axis=0)
            # Asegúrate de que next_state tenga la forma correcta (1, state_size)
            if next_state.ndim == 1:
                next_state = np.expand_dims(next_state, axis=0)
    
            if self.use_double_dqn:
                next_action = np.argmax(self.model.predict(next_state, verbose=0)[0])
                target_q = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][next_action]
            else:
                target_q = reward + self.gamma * np.max(self.model.predict(next_state, verbose=0)[0])
        error = np.abs(target_q - q_value)
        priority = self._get_priority(error)
        self.memory.add(priority, (state, action, reward, next_state, done))
        
    def batch_train(self, batch_size):
        tree_idx = np.empty((batch_size,), dtype=np.int32)
        batch = np.empty((batch_size,), dtype=object)
        priorities = np.empty((batch_size,), dtype=np.float32)
        segment = self.memory.total_priority / batch_size
        self.beta = min(1., self.beta + (1 - self.beta_start) / self.beta_frames)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.memory.get_leaf(s)
            tree_idx[i] = idx
            batch[i] = data
            priorities[i] = p

        states = np.array([item[0] for item in batch])
        next_states = np.array([item[3] for item in batch])
        rewards = np.array([item[2] for item in batch])
        
        self.total_rewards += np.sum(rewards)
        actions = np.array([item[1] for item in batch])
        dones = np.array([item[4] for item in batch])

        if states.ndim > 2:
            states = np.squeeze(states, axis=1)
        if next_states.ndim > 2:
            next_states = np.squeeze(next_states, axis=1)

        target_q = self.model.predict(states, verbose=0)
        next_q = self.model.predict(next_states, verbose=0)
        target_next_q = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            old_val = target_q[i, actions[i]]
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                if self.use_double_dqn:
                    best_action = np.argmax(next_q[i])
                    target_q[i, actions[i]] = rewards[i] + self.gamma * target_next_q[i, best_action]
                else:
                    target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
            # Calcular el error TD para actualizar la prioridad
            error = np.abs(target_q[i, actions[i]] - old_val)
            self.memory.update(tree_idx[i], self._get_priority(error))

        # Calcular los pesos de importancia
        sampling_probabilities = priorities / self.memory.total_priority
        weights = np.power(self.memory_size * sampling_probabilities, -self.beta)
        weights /= weights.max()

        history = self.model.fit(states, target_q, sample_weight=weights, epochs=1, verbose=0)
        self.loss_history.append(history.history['loss'][0])

        if self.use_double_dqn and self.step_counter % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())
            print(f"Modelo objetivo actualizado en el paso {self.step_counter}")

        self.step_counter += 1
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
    
        actions = self.model.predict(state, verbose=0)
        return np.argmax(actions[0])

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
            f.write(f"alpha:{self.alpha}\n")
            f.write(f"beta_start:{self.beta_start}\n")
            f.write(f"beta_frames:{self.beta_frames}\n")
            f.write(f"epsilon_priority:{self.epsilon_priority}\n")
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
                            elif key == "alpha":
                                self.alpha = float(value)
                            elif key == "beta_start":
                                self.beta_start = float(value)
                            elif key == "beta_frames":
                                self.beta_frames = int(value)
                            elif key == "epsilon_priority":
                                self.epsilon_priority = float(value)
                print(f"Modelo cargado desde {name}.h5 con epsilon = {self.epsilon} y parámetros.")
            else:
                print("Archivo de parámetros no encontrado, manteniendo valores por defecto.")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            print("Manteniendo valores por defecto.")

    def plot_training_metrics(self, save_path='resultados_cv'):
        min_length = min(len(self.profit_history), len(self.rewards_history), len(self.epsilon_history), len(self.trades_history), len(self.loss_history), len(self.drawdown_history), len(self.sharpe_ratios), len(self.accuracy_history), len(self.avg_win_history), len(self.avg_loss_history))

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

        axs[3, 1].axis('off') # Para la celda vacía en la última fila

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_path, 'training_metrics.png'))
        plt.show()
