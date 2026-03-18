# -*- coding: utf-8 -*-
"""
backtesting.py — Backtesting independiente del modelo Dueling DQN entrenado.
100% autónomo: no depende de ningún otro módulo del proyecto.

Uso:
    python backtesting.py

Configura el CSV a testear y la ruta del modelo en la sección
CONFIGURACIÓN al inicio del main().
"""

import os
import math
import random
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from sklearn.preprocessing import StandardScaler
from numba import jit


# ══════════════════════════════════════════════════════════════════════════════
# INDICADORES (de indicadores.py)
# ══════════════════════════════════════════════════════════════════════════════

def rsi(data, period=14):
    """RSI usando el método de Wilder vectorizado (ewm con alpha=1/period)."""
    if len(data) < period + 1:
        return pd.Series([50] * len(data), index=data.index)
    delta = data['close'].diff(1)
    up   = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    avg_up   = up.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    rsi_values = np.where(avg_down == 0, 100, rsi_values)
    rsi_values = np.where(avg_up   == 0,   0, rsi_values)
    return pd.Series(rsi_values, index=data.index).fillna(50)


def macd(data, fast_period=12, slow_period=26, signal_period=9):
    """MACD robusto para series cortas."""
    if len(data) < 2:
        z = pd.Series([0] * len(data), index=data.index)
        return z, z
    actual_fast   = min(fast_period,   len(data) // 2)
    actual_slow   = min(slow_period,   len(data) - 1)
    actual_signal = min(signal_period, len(data) // 3)
    if actual_fast >= actual_slow:
        actual_fast = max(1, actual_slow - 1)
    ema_fast = data['close'].ewm(span=actual_fast,   adjust=False).mean()
    ema_slow = data['close'].ewm(span=actual_slow,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    if actual_signal > 0:
        signal_line = macd_line.ewm(span=actual_signal, adjust=False).mean()
    else:
        signal_line = macd_line.copy()
    return macd_line.bfill().fillna(0), signal_line.bfill().fillna(0)


def add_ema200_distance(data, period=200):
    ema_200    = data['close'].ewm(span=period, adjust=False).mean()
    dist_ema200 = data['close'] - ema_200
    return dist_ema200.bfill().fillna(0), ema_200.bfill().fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# CREADORES DE ESTADO (de state_creator.py)
# ══════════════════════════════════════════════════════════════════════════════

def create_all_states_ohcl(data, window_size, scaler, hora_int):
    from numpy.lib.stride_tricks import sliding_window_view
    N = len(data)
    features        = data[['open', 'high', 'low', 'close', 'tick_volume']].values
    features_scaled = scaler.transform(features).astype(np.float32)
    padding = np.tile(features_scaled[0:1], (window_size - 1, 1))
    padded  = np.vstack([padding, features_scaled])
    windows = sliding_window_view(padded, (window_size, 5)).reshape(N, window_size * 5)
    h        = hora_int.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * h / 24).reshape(-1, 1)
    hour_cos = np.cos(2 * np.pi * h / 24).reshape(-1, 1)
    return np.hstack([windows, hour_sin, hour_cos])


def create_all_states_advanced(data, window_size, scaler, hora_int):
    from numpy.lib.stride_tricks import sliding_window_view
    N = len(data)

    # OHLCV
    features_ohlc = data[['open', 'high', 'low', 'close', 'tick_volume']].values
    ohlc_scaled   = scaler.transform(features_ohlc).astype(np.float32)
    padding_ohlc  = np.tile(ohlc_scaled[0:1], (window_size - 1, 1))
    padded_ohlc   = np.vstack([padding_ohlc, ohlc_scaled])
    windows_ohlc  = sliding_window_view(padded_ohlc, (window_size, 5)).reshape(N, window_size * 5)

    # RSI
    try:
        rsi_values = rsi(data, period=14).fillna(50).values
    except Exception:
        rsi_values = np.full(N, 50.0)
    rsi_normalized = (rsi_values / 100.0).astype(np.float32).reshape(-1, 1)

    # MACD
    try:
        macd_line, signal_line = macd(data, fast_period=12, slow_period=26, signal_period=9)
        macd_line   = macd_line.fillna(0).values
        signal_line = signal_line.fillna(0).values
        histogram   = macd_line - signal_line
    except Exception:
        macd_line = signal_line = histogram = np.zeros(N)
    macd_norm   = (macd_line   / 10.0).astype(np.float32).reshape(-1, 1)
    signal_norm = (signal_line / 10.0).astype(np.float32).reshape(-1, 1)
    hist_norm   = (histogram   / 10.0).astype(np.float32).reshape(-1, 1)

    # Día
    if hasattr(data.index, 'dayofweek'):
        day_of_week = data.index.dayofweek.values
    else:
        day_of_week = np.zeros(N)
    day_sin = np.sin(2 * np.pi * day_of_week / 7).astype(np.float32).reshape(-1, 1)
    day_cos = np.cos(2 * np.pi * day_of_week / 7).astype(np.float32).reshape(-1, 1)

    # Hora
    h        = hora_int.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * h / 24).reshape(-1, 1)
    hour_cos = np.cos(2 * np.pi * h / 24).reshape(-1, 1)

    return np.hstack([
        windows_ohlc, rsi_normalized,
        macd_norm, signal_norm, hist_norm,
        day_sin, day_cos, hour_sin, hour_cos,
    ]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# NOISY LINEAR (de NoisyLinear.py)
# ══════════════════════════════════════════════════════════════════════════════

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


# ══════════════════════════════════════════════════════════════════════════════
# DUELING DQN (de model_pytorch.py)
# ══════════════════════════════════════════════════════════════════════════════

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_space):
        super().__init__()
        self.layer_norm  = nn.LayerNorm(state_size)
        self.fc1         = nn.Linear(state_size, 512)
        self.ln1         = nn.LayerNorm(512)
        self.dropout1    = nn.Dropout(0.02)
        self.fc2         = nn.Linear(512, 256)
        self.ln2         = nn.LayerNorm(256)
        self.dropout2    = nn.Dropout(0.02)
        self.fc3         = nn.Linear(256, 128)
        self.ln3         = nn.LayerNorm(128)
        self.dropout3    = nn.Dropout(0.02)
        self.value_stream   = nn.Linear(128, 64)
        self.value_ln       = nn.LayerNorm(64)
        self.value_dropout  = nn.Dropout(0.02)
        self.value          = NoisyLinear(64, 1)
        self.advantage_stream  = nn.Linear(128, 128)
        self.advantage_ln      = nn.LayerNorm(128)
        self.advantage_dropout = nn.Dropout(0.02)
        self.advantage         = NoisyLinear(128, action_space)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, NoisyLinear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dropout1(F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.1))
        x = self.dropout2(F.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.1))
        x = self.dropout3(F.leaky_relu(self.ln3(self.fc3(x)), negative_slope=0.1))
        v = self.value_dropout(F.leaky_relu(self.value_ln(self.value_stream(x)), negative_slope=0.1))
        value = self.value(v)
        a = self.advantage_dropout(F.leaky_relu(self.advantage_ln(self.advantage_stream(x)), negative_slope=0.1))
        advantage = self.advantage(a)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self):
        self.value.reset_noise()
        self.advantage.reset_noise()


# ══════════════════════════════════════════════════════════════════════════════
# SUM TREE (de SumTree_class.py)
# ══════════════════════════════════════════════════════════════════════════════

class SumTree:
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1)
        self.data      = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.data_ptr  = 0

    def add(self, priority, data):
        idx = self.capacity - 1 + self.data_ptr
        self.data[self.data_ptr] = data
        self.update(idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent = 0
        while True:
            left  = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            if v <= self.tree[left]:
                parent = left
            else:
                v -= self.tree[left]
                parent = right
        data_idx = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

    @property
    def max_priority(self):
        if self.n_entries == 0:
            return 1.0
        m = self.tree[self.capacity - 1: self.capacity - 1 + self.n_entries].max()
        return float(m) if m > 0 else 1.0

    def __len__(self):
        return self.n_entries


# ══════════════════════════════════════════════════════════════════════════════
# AGENTE — solo lo necesario para inferencia (de dueling_dqn_con_per.py)
# ══════════════════════════════════════════════════════════════════════════════

class AI_Trader_per:
    def __init__(self, state_size, action_space=5,
                 epsilon_decay=0.995, commission_per_trade=0.0,
                 gamma=0.98, target_model_update=200,
                 memory_size=250000, alpha=0.6, beta_start=0.4,
                 beta_frames=100000, epsilon_priority=1e-3,
                 scheduler_type='cosine_decay', learning_rate=0.001,
                 lr_decay_rate=0.97, lr_decay_steps=1000, lr_min=1e-5,
                 **kwargs):

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.state_size       = state_size
        self.action_space     = action_space
        self.epsilon          = 1.0
        self.epsilon_decay    = epsilon_decay
        self.epsilon_final    = 0.0
        self.gamma            = gamma
        self.target_model_update = target_model_update
        self.step_counter     = 0
        self.memory           = SumTree(memory_size)
        self.memory_size      = memory_size
        self.alpha            = alpha
        self.beta_start       = beta_start
        self.beta             = beta_start
        self.beta_frames      = beta_frames
        self.epsilon_priority = epsilon_priority
        self.scheduler_type   = scheduler_type
        self.learning_rate    = learning_rate
        self.initial_learning_rate = learning_rate
        self.lr_decay_rate    = lr_decay_rate
        self.lr_decay_steps   = lr_decay_steps
        self.lr_min           = lr_min
        self.use_double_dqn   = True
        self.commission_per_trade = commission_per_trade
        self.spread           = 0.0
        self.reward_noise_std = 0.01
        self.random_market_event_probability = 0.01
        self.patience         = 10
        self.factor           = 0.5
        self.cosine_restarts  = True

        self.model        = DuelingDQN(state_size, action_space).to(self.device)
        self.target_model = DuelingDQN(state_size, action_space).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer    = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.001)
        self.scheduler    = self._create_scheduler()
        self.has_noise    = hasattr(self.model, "reset_noise")

    def _create_scheduler(self):
        if self.scheduler_type == 'cosine_decay':
            if self.cosine_restarts:
                return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=self.lr_decay_steps, eta_min=self.lr_min)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.lr_decay_steps, eta_min=self.lr_min)
        if self.scheduler_type == 'exponential_decay':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay_rate)
        if self.scheduler_type == 'polynomial_decay':
            return optim.lr_scheduler.PolynomialLR(
                self.optimizer, total_iters=self.lr_decay_steps, power=0.9)
        if self.scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.factor,
                patience=self.patience, min_lr=self.lr_min)
        return None

    def trade(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        if self.has_noise:
            self.model.reset_noise()
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)[0]).cpu().numpy()

    def load_model(self, name, cargar_memoria_buffer=False):
        if cargar_memoria_buffer and os.path.exists(f"{name}_memory.pkl"):
            with open(f"{name}_memory.pkl", "rb") as f:
                self.memory = pickle.load(f)
            print("Memoria cargada")

        checkpoint = torch.load(f"{name}.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.use_double_dqn and os.path.exists(f"{name}_target.pth"):
            t = torch.load(f"{name}_target.pth", map_location=self.device)
            self.target_model.load_state_dict(t['model_state_dict'])

        if os.path.exists(f"{name}_params.txt"):
            with open(f"{name}_params.txt", "r") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        if   key == "epsilon":        self.epsilon        = float(value)
                        elif key == "step_counter":   self.step_counter   = int(value)
                        elif key == "use_double_dqn": self.use_double_dqn = value.lower() == "true"
                        elif key == "scheduler_type": self.scheduler_type = value
                        elif key == "lr_decay_steps": self.lr_decay_steps = int(value)
                        elif key == "lr_min":         self.lr_min         = float(value)
                        elif key == "cosine_restarts":self.cosine_restarts= value.lower() == "true"
            self.scheduler = self._create_scheduler()
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Parámetros cargados. epsilon = {self.epsilon}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def price_format(n):
    n = float(n)
    return "- {0:.3f}".format(abs(n)) if n < 0 else "{0:.3f}".format(abs(n))


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate
    std = np.std(excess)
    return float(np.mean(excess) / std) if std != 0 else 0.0


@jit(nopython=True)
def calculate_profit_fast(buy_price, sell_price, pip_value, commission, lot_size):
    profit_pips    = sell_price - buy_price
    profit_dollars = (profit_pips * pip_value) - (commission * lot_size)
    return profit_pips, profit_dollars


@jit(nopython=True)
def calculate_short_profit_fast(sell_price, buy_price, pip_value, commission, lot_size):
    profit_pips    = sell_price - buy_price
    profit_dollars = (profit_pips * pip_value) - (commission * lot_size)
    return profit_pips, profit_dollars


def get_full_state(base_market_state, inventory, inventory_sell,
                   current_price, pip_value, initial_balance):
    has_long  = 1.0 if inventory      else 0.0
    has_short = 1.0 if inventory_sell else 0.0
    if inventory:
        raw_upnl = (current_price - inventory[0]) * pip_value
    elif inventory_sell:
        raw_upnl = (inventory_sell[0] - current_price) * pip_value
    else:
        raw_upnl = 0.0
    upnl_norm = float(np.tanh(raw_upnl / (initial_balance * 0.02)))
    return np.concatenate([base_market_state, [has_long, has_short, upnl_norm]]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def dataset_loader_csv(csv_path):
    try:
        df = pd.read_csv(f"data/{csv_path}", sep='\t')
        df.columns = [col.strip('<>').lower() for col in df.columns]
        ema_200_diferencia, _ = add_ema200_distance(df)

        if 'date' not in df.columns:
            print("Error: falta columna 'date'.")
            return None

        if len(ema_200_diferencia) > 0:
            df['ema_diference_close'] = ema_200_diferencia

        if 'time' in df.columns and df['time'].notnull().all():
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str))
        else:
            df['datetime'] = pd.to_datetime(df['date'])

        df.set_index('datetime', inplace=True)

        if 'tickvol' in df.columns:
            df.rename(columns={'tickvol': 'tick_volume'}, inplace=True)

        print(f"Datos cargados: {csv_path}  ({len(df)} filas)")

        required = {'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'time', 'ema_diference_close'}
        if required.issubset(df.columns):
            return df[['open', 'time', 'close', 'tick_volume', 'spread', 'low', 'ema_diference_close', 'high']]
        elif {'close', 'tick_volume'}.issubset(df.columns):
            return df[['close', 'tick_volume']]
        else:
            print("Error: el CSV no tiene las columnas mínimas necesarias.")
            return None

    except FileNotFoundError:
        print(f"Error: archivo no encontrado: data/{csv_path}")
        return None
    except Exception as e:
        print(f"Error al leer CSV: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO
# ══════════════════════════════════════════════════════════════════════════════

def plot_backtest_session(data, buy_points, sell_points, symbol, save_path, equity_curve):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Backtesting — {symbol}', fontsize=14, fontweight='bold')

    ax1.plot(data.index, data['close'], color='#555555', linewidth=0.8, label='Precio (close)')
    if buy_points:
        bx, by = zip(*buy_points)
        ax1.scatter(bx, by, marker='^', color='#00CC44', s=60, zorder=5, label='Compra / Cubre short')
    if sell_points:
        sx, sy = zip(*sell_points)
        ax1.scatter(sx, sy, marker='v', color='#FF3333', s=60, zorder=5, label='Venta / Abre short')
    ax1.set_ylabel('Precio')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)

    if equity_curve:
        eq_idx = data.index[:len(equity_curve)]
        ax2.plot(eq_idx, equity_curve, color='#1a7abf', linewidth=1.2, label='Equity')
        ax2.axhline(equity_curve[0], color='gray', linestyle='--', linewidth=0.7)
        ax2.set_ylabel('Equity ($)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, f'backtest_{symbol}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado en: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL DE BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(data, trader, scaler, config):
    window_size   = config['window_size']
    tipo_estado   = config['tipo_estado']
    pip_value     = config['pip_value']
    lot_size      = config['lot_size']
    commission    = config['commission']
    balance_first = config['balance_first']

    n_samples = len(data) - 1

    has_time = 'time' in data.columns and data['time'].notnull().all()
    hora_int = (
        np.array([int(h.split(":")[0]) for h in data['time'].values])
        if has_time else np.zeros(len(data), dtype=np.int32)
    )

    print(f"Generando estados ({tipo_estado})...")
    if tipo_estado == 'advanced':
        all_states = create_all_states_advanced(data, window_size, scaler, hora_int)
    else:
        all_states = create_all_states_ohcl(data, window_size, scaler, hora_int)
    print(f"  Estados generados: {len(all_states)}")

    close_prices = data['close'].values
    low_prices   = data['low'].values   if 'low'    in data.columns else close_prices
    high_prices  = data['high'].values  if 'high'   in data.columns else close_prices
    spreads      = data['spread'].values if 'spread' in data.columns else np.zeros(len(data))
    timestamps   = data.index.values

    inventory      = []
    inventory_sell = []
    current_equity = balance_first
    peak_equity    = balance_first
    best_low       = 9999999.0
    best_high      = 0.0

    total_profit_pips = 0.0
    total_profit_usd  = 0.0
    trades_count      = 0
    wins = losses     = 0
    win_pips_list     = []
    loss_pips_list    = []
    returns_pips      = []
    drawdown_history  = []
    equity_curve      = []
    buy_points        = []
    sell_points       = []

    trader.model.eval()
    trader.epsilon = 0.0

    print(f"\nEjecutando backtesting ({n_samples} pasos)...")

    for t in range(n_samples):
        current_price = float(close_prices[t])
        current_low   = float(low_prices[t])
        current_high  = float(high_prices[t])
        spread        = float(spreads[t])

        buy_exec  = current_price + spread * lot_size * 0.5
        sell_exec = current_price - spread * lot_size * 0.5

        if inventory and current_low < best_low:
            best_low = current_low
        elif not inventory:
            best_low = 9999999.0

        if inventory_sell and current_high > best_high:
            best_high = current_high
        elif not inventory_sell:
            best_high = 0.0

        state  = get_full_state(all_states[t], inventory, inventory_sell,
                                current_price, pip_value, balance_first)
        action = trader.trade(state)

        profit_pips  = 0.0
        profit_usd   = 0.0
        trade_closed = False

        if action == 1 and not inventory:
            inventory.append(buy_exec)
            buy_points.append((timestamps[t], buy_exec))

        elif action == 2 and inventory:
            entry = inventory.pop(0)
            profit_pips, profit_usd = calculate_profit_fast(entry, sell_exec, pip_value, commission, lot_size)
            sell_points.append((timestamps[t], sell_exec))
            best_low = 9999999.0
            trade_closed = True

        elif action == 3 and not inventory_sell:
            inventory_sell.append(sell_exec)
            sell_points.append((timestamps[t], sell_exec))

        elif action == 4 and inventory_sell:
            entry = inventory_sell.pop(0)
            profit_pips, profit_usd = calculate_short_profit_fast(entry, buy_exec, pip_value, commission, lot_size)
            buy_points.append((timestamps[t], buy_exec))
            best_high = 0.0
            trade_closed = True

        elif has_time and hora_int[t] == 23:
            if inventory:
                entry = inventory.pop(0)
                profit_pips, profit_usd = calculate_profit_fast(entry, sell_exec, pip_value, commission, lot_size)
                sell_points.append((timestamps[t], sell_exec))
                best_low = 9999999.0
                trade_closed = True
            elif inventory_sell:
                entry = inventory_sell.pop(0)
                profit_pips, profit_usd = calculate_short_profit_fast(entry, buy_exec, pip_value, commission, lot_size)
                buy_points.append((timestamps[t], buy_exec))
                best_high = 0.0
                trade_closed = True

        if trade_closed:
            total_profit_pips += profit_pips
            total_profit_usd  += profit_usd
            current_equity    += profit_usd
            trades_count      += 1
            returns_pips.append(profit_pips)
            if current_equity > peak_equity:
                peak_equity = current_equity
            if profit_pips > 0:
                wins += 1
                win_pips_list.append(profit_pips)
            else:
                losses += 1
                loss_pips_list.append(profit_pips)

        dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
        drawdown_history.append(dd)
        equity_curve.append(current_equity)

    sharpe       = calculate_sharpe_ratio(np.array(returns_pips)) if returns_pips else 0.0
    accuracy     = wins / trades_count if trades_count > 0 else 0.0
    avg_win      = float(np.mean(win_pips_list))  if win_pips_list  else 0.0
    avg_loss     = float(np.mean(loss_pips_list)) if loss_pips_list else 0.0
    max_drawdown = max(drawdown_history) if drawdown_history else 0.0

    return {
        'total_profit_pips': total_profit_pips,
        'total_profit_usd':  total_profit_usd,
        'trades':            trades_count,
        'wins':              wins,
        'losses':            losses,
        'sharpe':            sharpe,
        'accuracy':          accuracy,
        'avg_win_pips':      avg_win,
        'avg_loss_pips':     avg_loss,
        'max_drawdown':      max_drawdown,
        'final_equity':      current_equity,
        'equity_curve':      equity_curve,
        'buy_points':        buy_points,
        'sell_points':       sell_points,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ══════════════════════════════════════════════════════
    #  CONFIGURACIÓN — modifica aquí según tu necesidad
    # ══════════════════════════════════════════════════════
    NOMBRE_CSV     = "XAUUSD_M15_2025_03_01_2025_03_31.csv"
    MODELO_PATH    = "resultados_cv/XAUUSD_M15_2025_03_01_2025_03_31.csv"
    CARGAR_MEMORIA = False

    SYMBOL         = "GOLD"
    BALANCE        = 1000
    LOT_SIZE       = 0.01
    COMMISSION     = 0.0
    PIP_VALUE      = 10 * LOT_SIZE
    WINDOW_SIZE    = 18
    TIPO_ESTADO    = 'advanced'
    RESULTS_DIR    = 'resultados_cv'
    ACTION_SPACE   = 5

    EPSILON_DECAY       = 0.995
    GAMMA               = 0.98
    TARGET_MODEL_UPDATE = 200
    MEMORY_SIZE         = 250000
    ALPHA               = 0.6
    BETA_START          = 0.4
    BETA_FRAMES         = 100000
    EPSILON_PRIORITY    = 1e-3
    SCHEDULER_TYPE      = 'cosine_decay'
    LEARNING_RATE       = 0.001
    LR_DECAY_RATE       = 0.97
    LR_DECAY_STEPS      = 1000
    LR_MIN              = 1e-5
    # ══════════════════════════════════════════════════════

    if TIPO_ESTADO == 'ohlc':
        state_size = WINDOW_SIZE * 5 + 2 + 3
    elif TIPO_ESTADO == 'advanced':
        state_size = WINDOW_SIZE * 5 + 8 + 3
    else:
        raise ValueError(f"TIPO_ESTADO desconocido: {TIPO_ESTADO}")

    print("=" * 60)
    print(f"  BACKTESTING — {SYMBOL}")
    print(f"  CSV     : {NOMBRE_CSV}")
    print(f"  Modelo  : {MODELO_PATH}")
    print(f"  Estado  : {TIPO_ESTADO}  |  Window: {WINDOW_SIZE}")
    print(f"  Balance : ${BALANCE}  |  Lot: {LOT_SIZE}  |  Comisión: ${COMMISSION}")
    print("=" * 60)

    # ── 1. Cargar datos ──
    data = dataset_loader_csv(NOMBRE_CSV)
    if data is None:
        print("No se pudo cargar el CSV. Abortando.")
        return

    if len(data) <= WINDOW_SIZE:
        print(f"Datos insuficientes ({len(data)} filas) para window_size={WINDOW_SIZE}.")
        return

    # ── 2. Verificar modelo ──
    if not os.path.exists(f"{MODELO_PATH}.pth"):
        print(f"No se encontró el modelo: {MODELO_PATH}.pth — entrena primero.")
        return

    # ── 3. Scaler ──
    scaler = StandardScaler()
    scaler.fit(data[['open', 'high', 'low', 'close', 'tick_volume']].values)

    # ── 4. Cargar modelo ──
    trader = AI_Trader_per(
        state_size=state_size,
        action_space=ACTION_SPACE,
        epsilon_decay=EPSILON_DECAY,
        commission_per_trade=COMMISSION,
        gamma=GAMMA,
        target_model_update=TARGET_MODEL_UPDATE,
        memory_size=MEMORY_SIZE,
        alpha=ALPHA,
        beta_start=BETA_START,
        beta_frames=BETA_FRAMES,
        epsilon_priority=EPSILON_PRIORITY,
        scheduler_type=SCHEDULER_TYPE,
        learning_rate=LEARNING_RATE,
        lr_decay_rate=LR_DECAY_RATE,
        lr_decay_steps=LR_DECAY_STEPS,
        lr_min=LR_MIN,
    )

    try:
        trader.load_model(MODELO_PATH, cargar_memoria_buffer=CARGAR_MEMORIA)
        print(f"Modelo cargado desde: {MODELO_PATH}.pth")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # ── 5. Ejecutar backtesting ──
    config = {
        'window_size':   WINDOW_SIZE,
        'tipo_estado':   TIPO_ESTADO,
        'pip_value':     PIP_VALUE,
        'lot_size':      LOT_SIZE,
        'commission':    COMMISSION,
        'balance_first': BALANCE,
    }
    results = run_backtest(data, trader, scaler, config)

    # ── 6. Imprimir resultados ──
    print("\n" + "=" * 60)
    print("  RESULTADOS DEL BACKTESTING")
    print("=" * 60)
    print(f"  Beneficio (pips) : {results['total_profit_pips']:.2f}")
    print(f"  Beneficio (USD)  : {price_format(results['total_profit_usd'])}")
    print(f"  Equity final     : ${results['final_equity']:.2f}")
    print(f"  Trades totales   : {results['trades']}")
    print(f"  Wins / Losses    : {results['wins']} / {results['losses']}")
    print(f"  Accuracy         : {results['accuracy']:.2%}")
    print(f"  Sharpe ratio     : {results['sharpe']:.4f}")
    print(f"  Max Drawdown     : {results['max_drawdown']:.2%}")
    print(f"  Avg win (pips)   : {results['avg_win_pips']:.4f}")
    print(f"  Avg loss (pips)  : {results['avg_loss_pips']:.4f}")
    ratio = (abs(results['avg_win_pips']) / abs(results['avg_loss_pips'])
             if results['avg_loss_pips'] != 0 else float('inf'))
    print(f"  Win/Loss ratio   : {ratio:.2f}")
    print("=" * 60)

    # ── 7. Graficar ──
    plot_backtest_session(
        data=data,
        buy_points=results['buy_points'],
        sell_points=results['sell_points'],
        symbol=SYMBOL,
        save_path=RESULTS_DIR,
        equity_curve=results['equity_curve'],
    )

    print("\nBacktesting completado.")


if __name__ == "__main__":
    main()
