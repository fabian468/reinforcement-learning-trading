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
                 epsilon_priority=1e-6, # Pequeña constante para evitar probabilidad cero
                 # Nuevos parámetros para el scheduler
                 scheduler_type='exponential_decay',  # 'exponential_decay', 'cosine_decay', 'polynomial_decay', 'reduce_on_plateau'
                 lr_decay_rate=0.96,        # Factor de decaimiento para exponential
                 lr_decay_steps=1000,       # Pasos entre decaimientos
                 lr_min=1e-6,              # Learning rate mínimo
                 patience=10,              # Para reduce_on_plateau
                 factor=0.5,               # Factor de reducción para reduce_on_plateau
                 cosine_restarts=False,    # Para cosine decay con restarts
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
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.use_double_dqn = use_double_dqn
        self.target_model_update = target_model_update
        self.step_counter = 0
        self.total_rewards = 0

        # Parámetros del scheduler
        self.scheduler_type = scheduler_type
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.lr_min = lr_min
        self.patience = patience
        self.factor = factor
        self.cosine_restarts = cosine_restarts
        
        # Variables para reduce_on_plateau
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Historial del learning rate
        self.lr_history = []

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
        
    def get_scheduled_lr(self, step):
        """Calcula el learning rate basado en el scheduler configurado"""
        if self.scheduler_type == 'exponential_decay':
            # Decaimiento exponencial: lr = initial_lr * decay_rate^(step/decay_steps)
            decay_factor = self.lr_decay_rate ** (step / self.lr_decay_steps)
            new_lr = self.initial_learning_rate * decay_factor
            return max(new_lr, self.lr_min)
            
        elif self.scheduler_type == 'polynomial_decay':
            # Decaimiento polinomial
            if step < self.lr_decay_steps:
                decay_factor = (1 - step / self.lr_decay_steps) ** 0.9
                new_lr = (self.initial_learning_rate - self.lr_min) * decay_factor + self.lr_min
            else:
                new_lr = self.lr_min
            return new_lr
            
        elif self.scheduler_type == 'cosine_decay':
            # Decaimiento coseno
            if self.cosine_restarts:
                # Cosine decay with restarts
                restart_period = self.lr_decay_steps
                t = step % restart_period
                cosine_factor = 0.5 * (1 + np.cos(np.pi * t / restart_period))
            else:
                # Cosine decay sin restarts
                cosine_factor = 0.5 * (1 + np.cos(np.pi * min(step, self.lr_decay_steps) / self.lr_decay_steps))
            
            new_lr = self.lr_min + (self.initial_learning_rate - self.lr_min) * cosine_factor
            return new_lr
            
        else:  # 'constant' o cualquier otro valor
            return self.learning_rate
    
    def update_learning_rate_on_plateau(self, current_loss):
        """Actualiza el learning rate basado en el loss (reduce on plateau)"""
        if self.scheduler_type == 'reduce_on_plateau':
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                new_lr = max(self.learning_rate * self.factor, self.lr_min)
                if new_lr < self.learning_rate:
                    self.learning_rate = new_lr
                    # Actualizar el learning rate del optimizador de forma segura
                    self._update_optimizer_lr(new_lr)
                    print(f"Learning rate reducido a {self.learning_rate:.6f} en el paso {self.step_counter}")
                self.patience_counter = 0
    
    def _update_optimizer_lr(self, new_lr):
        """Actualiza el learning rate del optimizador de forma segura"""
        try:
            # Método más robusto para actualizar el learning rate
            self.model.optimizer.learning_rate.assign(new_lr)
            if self.use_double_dqn and hasattr(self, 'target_model'):
                self.target_model.optimizer.learning_rate.assign(new_lr)
        except (AttributeError, TypeError):
            try:
                # Método alternativo
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, float(new_lr))
                if self.use_double_dqn and hasattr(self, 'target_model'):
                    tf.keras.backend.set_value(self.target_model.optimizer.learning_rate, float(new_lr))
            except (AttributeError, TypeError):
                # Último recurso: recompilar el modelo
                self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=float(new_lr)))
                if self.use_double_dqn and hasattr(self, 'target_model'):
                    self.target_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=float(new_lr)))
                
    def update_learning_rate(self):
        """Actualiza el learning rate según el scheduler configurado"""
        if self.scheduler_type != 'reduce_on_plateau':
            new_lr = self.get_scheduled_lr(self.step_counter)
            if abs(new_lr - self.learning_rate) > 1e-8:  # Solo actualizar si hay cambio significativo
                self.learning_rate = new_lr
                # Actualizar el learning rate del optimizador de forma segura
                self._update_optimizer_lr(new_lr)
        
        # Guardar en historial
        self.lr_history.append(self.learning_rate)
        
    def _create_lr_schedule(self):
        """Crea un schedule de TensorFlow nativo (alternativa más robusta)"""
        if self.scheduler_type == 'exponential_decay':
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.initial_learning_rate,
                decay_steps=self.lr_decay_steps,
                decay_rate=self.lr_decay_rate,
                staircase=False
            )
        elif self.scheduler_type == 'cosine_decay':
            if self.cosine_restarts:
                return tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=self.initial_learning_rate,
                    first_decay_steps=self.lr_decay_steps,
                    t_mul=2.0,
                    m_mul=1.0,
                    alpha=self.lr_min/self.initial_learning_rate
                )
            else:
                return tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=self.initial_learning_rate,
                    decay_steps=self.lr_decay_steps,
                    alpha=self.lr_min/self.initial_learning_rate
                )
        elif self.scheduler_type == 'polynomial_decay':
            return tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.initial_learning_rate,
                decay_steps=self.lr_decay_steps,
                end_learning_rate=self.lr_min,
                power=0.9
            )
        else:
            # Para 'reduce_on_plateau' o 'constant', usamos LR fijo
            return self.learning_rate

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
    
        # Crear el schedule de learning rate
        lr_schedule = self._create_lr_schedule()
        
        # Compilamos el modelo con el schedule
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
        
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

        # Actualizar learning rate solo para reduce_on_plateau y manual
        if self.scheduler_type in ['reduce_on_plateau', 'constant']:
            self.update_learning_rate()

        history = self.model.fit(states, target_q, sample_weight=weights, epochs=1, verbose=0)
        current_loss = history.history['loss'][0]
        self.loss_history.append(current_loss)
        
        # Obtener el learning rate actual del optimizador
        try:
            if hasattr(self.model.optimizer.learning_rate, 'numpy'):
                current_lr = float(self.model.optimizer.learning_rate.numpy())
            else:
                current_lr = float(self.model.optimizer.learning_rate)
            self.learning_rate = current_lr
        except:
            # Fallback: usar el LR calculado manualmente
            if self.scheduler_type not in ['reduce_on_plateau', 'constant']:
                self.learning_rate = self.get_scheduled_lr(self.step_counter)
        
        # Actualizar learning rate basado en el loss si usamos reduce_on_plateau
        self.update_learning_rate_on_plateau(current_loss)
        
        # Guardar en historial
        self.lr_history.append(self.learning_rate)

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
            f.write(f"scheduler_type:{self.scheduler_type}\n")
            f.write(f"lr_decay_rate:{self.lr_decay_rate}\n")
            f.write(f"lr_decay_steps:{self.lr_decay_steps}\n")
            f.write(f"lr_min:{self.lr_min}\n")
            f.write(f"initial_learning_rate:{self.initial_learning_rate}\n")
            f.write(f"current_learning_rate:{self.learning_rate}\n")
        print(f"Modelo guardado como {name}.h5 y parámetros en {name}_params.txt")
        

    def load_model(self, name):
        try:
            self.model = tf.keras.models.load_model(f"{name}.h5", custom_objects={'combine_value_and_advantage': combine_value_and_advantage}, compile=False)
            self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
            if self.use_double_dqn and tf.io.gfile.exists(f"{name}_target.h5"):
                self.target_model = tf.keras.models.load_model(f"{name}_target.h5", custom_objects={'combine_value_and_advantage': combine_value_and_advantage}, compile=False)
                self.target_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
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
                            elif key == "scheduler_type":
                                self.scheduler_type = value
                            elif key == "lr_decay_rate":
                                self.lr_decay_rate = float(value)
                            elif key == "lr_decay_steps":
                                self.lr_decay_steps = int(value)
                            elif key == "lr_min":
                                self.lr_min = float(value)
                            elif key == "initial_learning_rate":
                                self.initial_learning_rate = float(value)
                            elif key == "current_learning_rate":
                                self.learning_rate = float(value)
                print(f"Modelo cargado desde {name}.h5 con epsilon = {self.epsilon} y parámetros.")
            else:
                print("Archivo de parámetros no encontrado, manteniendo valores por defecto.")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            print("Manteniendo valores por defecto.")

    def plot_training_metrics(self, save_path='resultados_cv'):
        min_length = min(len(self.profit_history), len(self.rewards_history), 
                        len(self.epsilon_history), len(self.trades_history), 
                        len(self.loss_history), len(self.drawdown_history), 
                        len(self.sharpe_ratios), len(self.accuracy_history), 
                        len(self.avg_win_history), len(self.avg_loss_history),
                        len(self.lr_history))

        episodes = range(1, min_length + 1)

        fig, axs = plt.subplots(5, 2, figsize=(15, 20))  # 5 filas, 2 columnas
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
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        axs[1, 1].plot(episodes, self.drawdown_history[:min_length], label='Drawdown Máximo', color='orange')
        axs[1, 1].set_ylabel('Drawdown')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        axs[2, 0].plot(episodes, self.sharpe_ratios[:min_length], label='Ratio de Sharpe', color='green')
        axs[2, 0].set_ylabel('Ratio de Sharpe')
        axs[2, 0].grid(True)
        axs[2, 0].legend()

        axs[2, 1].plot(episodes, self.accuracy_history[:min_length], label='Frecuencia de Aciertos', color='brown')
        axs[2, 1].set_ylabel('Frecuencia de Aciertos')
        axs[2, 1].grid(True)
        axs[2, 1].legend()

        axs[3, 0].plot(episodes, self.rewards_history[:min_length], label='Recompensa por Episodio', color='blue')
        axs[3, 0].set_ylabel('Recompensa')
        axs[3, 0].grid(True)
        axs[3, 0].legend()

        # Nueva gráfica: Learning Rate
        axs[3, 1].plot(episodes, self.lr_history[:min_length], label='Learning Rate', color='magenta')
        axs[3, 1].set_ylabel('Learning Rate')
        axs[3, 1].set_xlabel('Episodio')
        axs[3, 1].grid(True)
        axs[3, 1].legend()
        axs[3, 1].set_yscale('log')  # Escala logarítmica para mejor visualización

        # Gráfica combinada: Loss vs Learning Rate
        axs[4, 0].plot(episodes, self.loss_history[:min_length], label='Loss', color='purple', alpha=0.7)
        axs[4, 0].set_ylabel('Loss', color='purple')
        axs[4, 0].tick_params(axis='y', labelcolor='purple')
        
        ax2 = axs[4, 0].twinx()
        ax2.plot(episodes, self.lr_history[:min_length], label='Learning Rate', color='magenta', alpha=0.7)
        ax2.set_ylabel('Learning Rate', color='magenta')
        ax2.tick_params(axis='y', labelcolor='magenta')
        ax2.set_yscale('log')
        
        axs[4, 0].set_xlabel('Episodio')
        axs[4, 0].grid(True)
        axs[4, 0].set_title('Loss vs Learning Rate')

        axs[4, 1].axis('off') # Para la celda vacía en la última fila

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_path, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def print_scheduler_info(self):
        """Imprime información sobre el scheduler configurado"""
        print(f"\n=== Configuración del Learning Rate Scheduler ===")
        print(f"Tipo de scheduler: {self.scheduler_type}")
        print(f"Learning rate inicial: {self.initial_learning_rate}")
        print(f"Learning rate actual: {self.learning_rate}")
        print(f"Learning rate mínimo: {self.lr_min}")
        
        if self.scheduler_type == 'exponential_decay':
            print(f"Factor de decaimiento: {self.lr_decay_rate}")
            print(f"Pasos entre decaimientos: {self.lr_decay_steps}")
        elif self.scheduler_type == 'reduce_on_plateau':
            print(f"Paciencia: {self.patience}")
            print(f"Factor de reducción: {self.factor}")
            print(f"Mejor loss: {self.best_loss}")
        elif self.scheduler_type == 'cosine_decay':
            print(f"Pasos totales: {self.lr_decay_steps}")
            print(f"Con restarts: {self.cosine_restarts}")
        print("=" * 50)