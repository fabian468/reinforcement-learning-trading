"""
Created on Mon May 12 10:44:38 2025

@author: fabia
"""

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import pickle
from collections import deque

from SumTree_class import SumTree
from model_pytorch import DuelingDQN


class AI_Trader_per():
    def __init__(self,
                 state_size,
                 action_space=5,
                 model_name="AITrader",
                 random_market_event_probability=0.01,
                 spread=0.20,
                 commission_per_trade=0.07,
                 gamma=0.95,
                 epsilon=1,
                 epsilon_final=0.15,
                 epsilon_decay=0.9999,
                 use_double_dqn=True,
                 target_model_update=100,
                 learning_rate=0.001,
                 memory_size=500000, # Tamaño de la memoria
                 alpha=0.5,        # Hiperparámetro para la priorización
                 beta_start=0.4,   # Hiperparámetro para la corrección de importancia
                 beta_frames=100000, # Número de frames para alcanzar beta=1
                 epsilon_priority=1e-5, # Pequeña constante para evitar probabilidad cero
                 # Nuevos parámetros para el scheduler
                 scheduler_type='cosine_decay',  # 'exponential_decay', 'cosine_decay', 'polynomial_decay', 'reduce_on_plateau'
                 lr_decay_rate=0.97,        # Factor de decaimiento para exponential
                 lr_decay_steps=1000,       # Pasos entre decaimientos
                 lr_min=0.000001,              # Learning rate mínimo
                 patience=10,              # Para reduce_on_plateau
                 factor=0.5,               # Factor de reducción para reduce_on_plateau
                 cosine_restarts=True,    # Para cosine decay con restarts
                 ):
        
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Mixed precision: usa float16 en GPU para forward/backward más rápidos
        self.use_amp = self.device.type == "cuda"
        self.scaler_amp = torch.amp.GradScaler("cuda") if self.use_amp else None
        
        
        self.state_size = state_size
        self.action_space = action_space
        self.memory = SumTree(memory_size) # Usamos SumTree ahora
        self.memory_size = memory_size
        self.inventory = []
        self.inventory_sell = []
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
        self.rewards_epsilon_buffer = deque(maxlen=30)

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
        
        # Historial del learning rate
        self.lr_history = []
        self.lr_history_episode = []  # Una entrada por episodio (para el plot)

        # Crear modelos
        self.model = DuelingDQN(state_size, action_space).to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.has_noise = hasattr(self.model, "reset_noise")  # Cacheado una vez

        if self.use_double_dqn:
            self.target_model = DuelingDQN(state_size, action_space).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())

        self.profit_history = []
        self.rewards_history = []
        self.rewards_history_episode = []
        self.epsilon_history = []
        self.trades_history = []
        self.loss_history = []
        self.drawdown_history = []
        self.sharpe_ratios = []
        self.accuracy_history = []
        self.avg_win_history = []
        self.avg_loss_history = []

        # Contadores de acciones random vs modelo
        self.random_action_count  = 0
        self.model_action_count   = 0
        self.last_action_was_random = False

        # Hiperparámetros PER
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.epsilon_priority = epsilon_priority
        
    def _create_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)
    
    def _create_scheduler(self):
        if self.scheduler_type == 'exponential_decay':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay_rate)
        elif self.scheduler_type == 'cosine_decay':
            if self.cosine_restarts:
                return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=self.lr_decay_steps, eta_min=self.lr_min
                )
            else:
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_decay_steps, eta_min=self.lr_min
                )
        elif self.scheduler_type == 'polynomial_decay':
            return optim.lr_scheduler.PolynomialLR(
                self.optimizer, total_iters=self.lr_decay_steps, power=0.9
            )
        elif self.scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.factor, 
                patience=self.patience, min_lr=self.lr_min
            )
        else:  # 'constant'
            return None

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon_priority) ** self.alpha

    def remember(self, state, action, reward, next_state, done):
        # Usar prioridad máxima actual para nuevas experiencias (estándar en PER).
        # Evita los 2 forward passes por paso; las prioridades se corrigen en batch_train.
        priority = self.memory.max_priority
        self.memory.add(priority, (state, action, reward, next_state, done))
        
    def adaptative_epsilon_from_history(self, reward, step=0.01, min_epsilon=0.01, max_epsilon=1.0, window=10):
        """
        Ajusta epsilon dinámicamente en función del promedio móvil de recompensas.
        """
     
        # Guardar la nueva recompensa
        self.rewards_epsilon_buffer.append(reward)
    
        # Verificar si hay suficiente historial
        if len(self.rewards_epsilon_buffer) < 2 * window:
            return self.epsilon
        
        
        # Obtener las dos ventanas de recompensa
        rewards = list(self.rewards_epsilon_buffer)
        prev_avg = np.mean(rewards[-2*window:-window])
        curr_avg = np.mean(rewards[-window:])

        # Ajustar epsilon según tendencia
        if curr_avg > prev_avg:
            self.epsilon = max(self.epsilon - step, min_epsilon)
        elif curr_avg < prev_avg:
            self.epsilon = min(self.epsilon + step, max_epsilon)
        

        
    def batch_train(self, batch_size):
        # Inicialización para la extracción del batch
        tree_idx = np.empty((batch_size,), dtype=np.int32)
        batch_data = [] # Usamos una lista temporal para recolectar las tuplas de experiencia
        priorities = np.empty((batch_size,), dtype=np.float32)
        segment = self.memory.total_priority / batch_size
        
        # Actualización del parámetro beta para PER
        self.beta = min(1., self.beta + (1 - self.beta_start) / self.beta_frames)
    
        # Extraer el batch de la memoria priorizada
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.memory.get_leaf(s)
            tree_idx[i] = idx
            batch_data.append(data)
            priorities[i] = p

        # Desempaquetar el batch de manera vectorizada
        states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*batch_data)
    
        # Convertir a tensores de PyTorch
        states = torch.FloatTensor(np.array(states_list)).to(self.device)
        actions = torch.LongTensor(np.array(actions_list)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards_list)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states_list)).to(self.device)
        dones = torch.BoolTensor(np.array(dones_list)).to(self.device)
    
        # Reducir dimensiones si es necesario
        if states.dim() > 2:
            states = states.squeeze(1)
        if next_states.dim() > 2:
            next_states = next_states.squeeze(1)
    
        # Pesos de importancia (IS weights) — calculados antes del forward pass
        sampling_probabilities = priorities / self.memory.total_priority
        weights = np.power(self.memory_size * sampling_probabilities, -self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)

        autocast_ctx = torch.amp.autocast("cuda") if self.use_amp else torch.amp.autocast("cpu", enabled=False)

        with autocast_ctx:
            # Predicciones de los modelos
            current_q_values = self.model(states)
            with torch.no_grad():
                next_q_values = self.model(next_states)
                target_next_q_values = self.target_model(next_states) if self.use_double_dqn else next_q_values

                if self.use_double_dqn:
                    best_actions = torch.argmax(next_q_values, dim=1)
                    target_next_q_values_selected = target_next_q_values.gather(1, best_actions.unsqueeze(1)).squeeze()
                else:
                    target_next_q_values_selected = torch.max(target_next_q_values, dim=1)[0]

                q_targets = rewards + self.gamma * target_next_q_values_selected * (~dones)

            current_q_values_for_actions = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Calcular el error TD
            errors = torch.abs(q_targets - current_q_values_for_actions).detach().cpu().numpy()

            # Pérdida ponderada
            loss = F.mse_loss(current_q_values_for_actions, q_targets, reduction='none')
            weighted_loss = (loss * weights).mean()

        # Actualizar prioridades en memoria
        new_priorities = self._get_priority(errors)
        for i in range(batch_size):
            self.memory.update(tree_idx[i], new_priorities[i])

        # Backward pass con scaler AMP si disponible
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler_amp.scale(weighted_loss).backward()
            self.scaler_amp.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler_amp.step(self.optimizer)
            self.scaler_amp.update()
        else:
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        current_loss = weighted_loss.item()
        self.loss_history.append(current_loss)
        
        # Actualizar learning rate usando schedulers nativos de PyTorch
        self._update_learning_rate(current_loss)
        
        # Actualizar learning_rate actual desde el optimizador
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(self.learning_rate)
            
        # Actualizar el modelo objetivo (target model) si se usa Double DQN
        if self.use_double_dqn and self.step_counter % self.target_model_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
        # Incrementar el contador de pasos
        self.step_counter += 1
        
        #avg_reward = rewards.mean().item()

    # Ajustar epsilon según la historia de recompensas
        #self.adaptative_epsilon_from_history(reward = avg_reward)
        
        #if self.epsilon > self.epsilon_final:
             #self.epsilon *= self.epsilon_decay
            
        return current_loss
    
    def _update_learning_rate(self, current_loss):
        """Actualiza el learning rate usando los schedulers nativos de PyTorch"""
        if self.scheduler is None or self.scheduler_type == 'reduce_on_plateau':
            return

        if self.scheduler_type == 'exponential_decay':
            if self.step_counter % self.lr_decay_steps == 0 and self.step_counter > 0:
                self.scheduler.step()
        else:
            # cosine y polynomial: actualizar en cada paso
            self.scheduler.step()

    def step_episode_scheduler(self, episode_avg_loss):
        """Llamar al final de cada episodio. Requerido para reduce_on_plateau;
        para los demás schedulers registra el LR del episodio en lr_history_episode."""
        if self.scheduler is not None and self.scheduler_type == 'reduce_on_plateau':
            self.scheduler.step(episode_avg_loss)
            self.learning_rate = self.optimizer.param_groups[0]['lr']
        self.lr_history_episode.append(self.learning_rate)
            
    def trade(self, state):
        if np.random.random() <= self.epsilon:
            self.random_action_count += 1
            self.last_action_was_random = True
            return random.randrange(self.action_space)

        self.model_action_count += 1
        self.last_action_was_random = False

        if self.has_noise:
            self.model.reset_noise()

        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            actions = self.model(state)
            return torch.argmax(actions[0]).cpu().numpy()

    def save_model(self, name):
        # Guardar modelo principal
        model_save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Guardar scheduler si existe
        if self.scheduler is not None:
            model_save_dict['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(model_save_dict, f"{name}.pth")
        
        # Guardar modelo target si existe
        if self.use_double_dqn:
            torch.save({
                'model_state_dict': self.target_model.state_dict(),
            }, f"{name}_target.pth")
            
        # Guardar parámetros
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
            f.write(f"patience:{self.patience}\n")
            f.write(f"factor:{self.factor}\n")
            f.write(f"cosine_restarts:{self.cosine_restarts}\n")
            
        # Guardar memoria
        with open(f"{name}_memory.pkl", "wb") as f:
            pickle.dump(self.memory, f)

        print(f"Modelo guardado como {name}.pth, parámetros en {name}_params.txt y memoria en {name}_memory.pkl")
    
    def load_model(self, name, cargar_memoria_buffer):
        try:
            if cargar_memoria_buffer:
                with open(f"{name}_memory.pkl", "rb") as f:
                    self.memory = pickle.load(f)
                    print("Memoria cargada")
                    
            # Cargar modelo principal
            checkpoint = torch.load(f"{name}.pth", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Cargar modelo target si existe
            if self.use_double_dqn and os.path.exists(f"{name}_target.pth"):
                target_checkpoint = torch.load(f"{name}_target.pth", map_location=self.device)
                self.target_model.load_state_dict(target_checkpoint['model_state_dict'])
                
            # Cargar parámetros
            if os.path.exists(f"{name}_params.txt"):
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
                            # scheduler_type, lr_decay_rate, lr_decay_steps, lr_min,
                            # patience, factor, cosine_restarts se toman de parametros.py
                            # (configuración del entrenamiento actual), no del checkpoint.
                            elif key == "initial_learning_rate":
                                self.initial_learning_rate = float(value)
                            elif key == "current_learning_rate":
                                self.learning_rate = float(value)
                                
                # Recrear scheduler después de cargar parámetros, luego restaurar su estado
                self.scheduler = self._create_scheduler()
                if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                print(f"Modelo cargado desde {name}.pth con epsilon = {self.epsilon} y parámetros.")
            else:
                print("Archivo de parámetros no encontrado, manteniendo valores por defecto.")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            print("Manteniendo valores por defecto.")

    def plot_training_metrics(self, fold, save_path='resultados_cv'):
        # Solo incluir las historias que se grafican realmente
        min_length = min(len(self.profit_history), len(self.rewards_history),
                        len(self.epsilon_history), len(self.loss_history),
                        len(self.drawdown_history), len(self.sharpe_ratios),
                        len(self.accuracy_history), len(self.lr_history_episode),
                        len(self.rewards_history_episode))

        if min_length == 0:
            print(f"[plot] Sin datos suficientes para graficar fold {fold}")
            return

        episodes = range(1, min_length + 1)

        fig, axs = plt.subplots(5, 2, figsize=(15, 20))
        fig.suptitle(f'Métricas de Entrenamiento — Fold {fold}', fontsize=16)

        axs[0, 0].plot(episodes, self.profit_history[:min_length], label='Beneficio Total', color='blue')
        axs[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)  # Línea horizontal en y=0
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

        axs[3, 0].plot(episodes, self.rewards_history[:min_length], label='Recompensa total', color='blue')
        axs[3, 0].set_ylabel('Recompensa')
        axs[3, 0].grid(True)
        axs[3, 0].legend()

        # Nueva gráfica: Learning Rate
        axs[3, 1].plot(episodes, self.lr_history_episode[:min_length], label='Learning Rate', color='magenta')
        axs[3, 1].set_ylabel('Learning Rate')
        axs[3, 1].set_xlabel('Episodio')
        axs[3, 1].grid(True)
        axs[3, 1].legend()
        axs[3, 1].set_yscale('log')

        # Gráfica combinada: Loss vs Learning Rate
        axs[4, 0].plot(episodes, self.loss_history[:min_length], label='Loss', color='purple', alpha=0.7)
        axs[4, 0].set_ylabel('Loss', color='purple')
        axs[4, 0].tick_params(axis='y', labelcolor='purple')

        ax2 = axs[4, 0].twinx()
        ax2.plot(episodes, self.lr_history_episode[:min_length], label='Learning Rate', color='magenta', alpha=0.7)
        ax2.set_ylabel('Learning Rate', color='magenta')
        ax2.tick_params(axis='y', labelcolor='magenta')
        ax2.set_yscale('log')
        
        axs[4, 0].set_xlabel('Episodio')
        axs[4, 0].grid(True)
        axs[4, 0].set_title('Loss vs Learning Rate')

        axs[4, 1].plot(episodes, self.rewards_history_episode[:min_length], label='Recompensa por episodios', color='green')
        axs[4, 1].set_ylabel('Recompensa')
        axs[4, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axs[4, 1].grid(True)
        axs[4, 1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_path, f'training_metrics_fold_{fold}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    def print_scheduler_info(self):
        """Imprime información sobre el scheduler configurado"""
        print("\n=== Configuración del Learning Rate Scheduler ===")
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
        elif self.scheduler_type == 'cosine_decay':
            print(f"Pasos totales: {self.lr_decay_steps}")
            print(f"Con restarts: {self.cosine_restarts}")
        elif self.scheduler_type == 'polynomial_decay':
            print(f"Pasos totales: {self.lr_decay_steps}")
            print(f"Potencia: 0.9")
        print("=" * 50)