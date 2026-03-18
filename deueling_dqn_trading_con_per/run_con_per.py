# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 00:25:00 2025
@author: fabia
"""

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from notificador import enviar_alerta

from sklearn.preprocessing import StandardScaler

from numba import jit
import warnings
warnings.filterwarnings('ignore')

import cProfile
import pstats

load_dotenv()

import torch

if torch.cuda.is_available():
    print("GPU disponible:", torch.cuda.get_device_name(0))
else:
    print("GPU no disponible, ejecutando en CPU")

# Importar parámetros centralizados
from parametros import (
    ConfigEntorno, ConfigTrading, ConfigAgente,
    ConfigMemoria, ConfigScheduler, ConfigReward,
    ConfigEntrenamiento, ConfigModelo, ConfigBackend,
    get_state_size, get_config_trader, imprimir_config
)

from dueling_dqn_con_per import AI_Trader_per
from state_creator import (
    state_creator_ohcl_vectorized,
    create_all_states_ohcl, create_all_states_advanced
)
from AdvancedRewardSystem import AdvancedRewardSystem , calculate_advanced_reward
from request_datos_backend import upload_training_data
from plot_stadist import plot_trading_session

from indicadores import  add_ema200_distance
from tensorboard_logger import TBLogger
from live_plot import LivePlot

def price_format(n):
    n = float(n)
    return "- {0:.3f}".format(abs(n)) if n < 0 else "{0:.3f}".format(abs(n))


def get_full_state(base_market_state, inventory, inventory_sell, current_price, pip_value, initial_balance):
    """
    Concatena características de posición al estado de mercado.
    El agente necesita saber si tiene posición abierta y cuánto gana/pierde.

    Args:
        base_market_state : np.ndarray 1D (state del mercado)
        inventory         : lista de precios de entrada long
        inventory_sell    : lista de precios de entrada short
        current_price     : precio actual
        pip_value         : valor por pip en $ (pip_value_eur_usd)
        initial_balance   : balance inicial

    Returns:
        np.ndarray 1D (market_state + [has_long, has_short, upnl_norm])
    """
    has_long  = 1.0 if inventory else 0.0
    has_short = 1.0 if inventory_sell else 0.0

    if inventory:
        raw_upnl = (current_price - inventory[0]) * pip_value
    elif inventory_sell:
        raw_upnl = (inventory_sell[0] - current_price) * pip_value
    else:
        raw_upnl = 0.0

    upnl_norm = float(np.tanh(raw_upnl / (initial_balance * 0.02)))  # escala: 2% del balance

    return np.concatenate([base_market_state, [has_long, has_short, upnl_norm]]).astype(np.float32)



def dataset_loader_csv(csv_path):
    try:
        df = pd.read_csv(f"data/{csv_path}", sep='\t')  # Usa tabulador como separador
        df.columns = [col.strip('<>').lower() for col in df.columns]  # Limpia < > y normaliza nombres
        ema_200_diferencia, _ = add_ema200_distance(df)
        # Verifica si existen las columnas necesarias
        
        if 'date' not in df.columns:
            print("Error: La columna 'date' es obligatoria.")
            return None
        if len(ema_200_diferencia) > 0:
            df['ema_diference_close'] = ema_200_diferencia
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
            return df[[ 'open','time', 'close', 'tick_volume' , 'spread' , 'low' , 'ema_diference_close' , 'high']]
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

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0.0


@jit(nopython=True)
def calculate_profit_fast(buy_price, sell_price, pip_value, commission, lot_size):
    """Cálculo rápido de profit usando Numba"""
    profit_pips = sell_price - buy_price
    profit_dollars = (profit_pips * pip_value) - (commission * lot_size)
    return profit_pips, profit_dollars

@jit(nopython=True) 
def calculate_short_profit_fast(sell_price, buy_price, pip_value, commission, lot_size):
    """Cálculo rápido de profit para ventas en corto"""
    profit_pips = sell_price - buy_price
    profit_dollars = (profit_pips * pip_value) - (commission * lot_size)
    return profit_pips, profit_dollars

#como puedo mejorar la convergencia el 

#comienzo del codigo
def main():

    # Imprimir configuración actual (opcional)
    # imprimir_config()

    # === Usar parámetros centralizados ===
    nombre_csv = ConfigEntorno.NOMBRE_CSV

    cargar_modelo = ConfigModelo.CARGAR_MODELO
    modelo_existente = ConfigModelo.MODELO_EXISTENTE
    cargar_memoria_buffer = ConfigModelo.CARGAR_MEMORIA_BUFFER

    guardar_estadisticas_en_backend = ConfigBackend.GUARDAR_ESTADISTICAS_EN_BACKEND
    mostrar_prints = ConfigBackend.MOSTRAR_PRINTS

    # TensorBoard
    logger = TBLogger() if ConfigBackend.TENSORBOARD else None

    # Gráfico en vivo
    live = LivePlot(
        window_prices=ConfigBackend.LIVE_PLOT_WINDOW,
        update_every=ConfigBackend.LIVE_PLOT_UPDATE
    ) if ConfigBackend.LIVE_PLOT else None

    symbol = ConfigEntorno.SYMBOL
    intervalo = ConfigEntorno.INTERVALO

    nombre_modelo_guardado = ConfigModelo.NOMBRE_MODELO

    # Parámetros del mercado
    es_indice = ConfigEntorno.ES_INDICE
    es_forex = ConfigEntorno.ES_FOREX
    es_metal = ConfigEntorno.ES_METAL
    tick_value = ConfigEntorno.TICK_VALUE
    pip_multiplier = ConfigEntorno.PIP_MULTIPLIER

    # Entrenamiento
    cada_cuantos_episodes_guardar_el_modelo = ConfigEntrenamiento.GUARDAR_MODELO_CADA
    episodes = ConfigEntrenamiento.EPISODES
    n_folds = ConfigEntrenamiento.N_FOLDS
    batch_size = ConfigEntrenamiento.BATCH_SIZE
    train_frequency = ConfigEntrenamiento.TRAIN_FREQUENCY
    train_iterations = ConfigEntrenamiento.TRAIN_ITERATIONS

    # Agente
    epsilon_decay = ConfigAgente.EPSILON_DECAY
    gamma = ConfigAgente.GAMMA
    cada_cuanto_actualizar = ConfigAgente.TARGET_MODEL_UPDATE
    learning_rate = ConfigAgente.LEARNING_RATE

    # Entorno
    window_size = ConfigEntorno.WINDOW_SIZE
    ventana_para_los_estados_de_datos = ConfigEntorno.VENTANA_DATOS

    # Trading
    balance_first = ConfigTrading.BALANCE_INICIAL
    lot_size = ConfigTrading.LOT_SIZE
    commission_per_trade = ConfigTrading.COMMISSION_PER_TRADE
    test_size_ratio = ConfigEntorno.TEST_SIZE_RATIO

    # XAUUSD: 1 lote estándar = 100 oz, pip = $0.01/oz
    # Para 0.01 lots (1 oz): profit = price_diff * 100 * lot_size = price_diff * 1.0
    pip_value_eur_usd = 100 * lot_size

    # Creación de carpeta para guardar los resultados
    resultados_dir = ConfigModelo.DIRECTORIO_RESULTADOS
    os.makedirs(resultados_dir, exist_ok=True)

    reward_system = AdvancedRewardSystem(initial_balance=balance_first, weights=ConfigReward.get_pesos())
    reward_system.weights = ConfigReward.get_pesos()  # Sincronizar pesos desde parametros.py

    data = dataset_loader_csv(nombre_csv)
        
    # Calcular los datos para el entramiento y para el test el X e Y
    train_size = int(len(data) * (1 - test_size_ratio))
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()

    # Scaler global ajustado sobre todos los datos de entrenamiento (para normalizar el test)
    global_scaler = StandardScaler()
    global_scaler.fit(train_data[['open', 'high', 'low', 'close', 'tick_volume']].values)
    
    if 'time' in data.columns and data['time'].notnull().all():
        hora = data['time']
    
    if 'ema_diference_close' in data.columns and data['ema_diference_close'].notnull().all():
        alcista = data['ema_diference_close']
    
    
    # Usar función centralizada para calcular state_size
    state_size = get_state_size()

    # state_size_rsi_macd ya no se usa, mantener por compatibilidad
    # state_size_rsi_macd = (window_size - 1) * 4 + window_size + window_size + 4

    # Obtener parámetros del trader desde parametros.py
    trader_params = get_config_trader()

    # Carga el modelo y ve si cargar uno o crear uno nuevo
    trader = AI_Trader_per(
        state_size,
        **trader_params
    )

    if cargar_modelo:
        try:
            trader.load_model(modelo_existente , cargar_memoria_buffer)
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar el modelo {modelo_existente}: {str(e)}")

    
    fold_size = len(train_data) // n_folds
    all_fold_metrics = []
    

    # Comienza el entrenamiento del primero fold
    for fold in range(n_folds):
        print(f"\n{'='*30} Fold {fold + 1}/{n_folds} {'='*30}")
        start = fold  * fold_size
        end = (fold + 1) * fold_size
        
        
        
        absolute_start = max(start, ventana_para_los_estados_de_datos)  # Garantiza que mínimo empiece desde fila 10
        fold_data = train_data.iloc[absolute_start:end].copy()
        
        scaler = StandardScaler()
        scaler.fit(fold_data[['open', 'high', 'low', 'close', 'tick_volume']].values)

        data_samples = len(fold_data) - 1 
        
        print(f"Pre-calculando datos del fold {fold + 1 }...")
        close_prices = fold_data['close'].values
        low_prices = fold_data['low'].values  
        high_prices = fold_data['high'].values
        spreads = fold_data['spread'].values
        timestamps = fold_data.index.values
        
        # Pre-calcular precios de compra y venta
        spread_half = spreads * lot_size * 0.5
        buy_prices_pre = close_prices + spread_half
        sell_prices_pre = close_prices - spread_half
        
        # Pre-calcular alcista si existe
        if 'alcista' in locals():
            alcista_values = alcista.values
        else:
            alcista_values = np.zeros(len(fold_data))
        
        # Pre-calcular horas si existe
        if 'hora' in locals() and 'time' in data.columns:
            hora_values = hora.values
            has_time = True
            # Pre-parsear las horas como enteros para evitar string.split en el hot loop
            hora_int_pre = np.array([int(h.split(":")[0]) for h in fold_data['time'].values])
        else:
            has_time = False
            hora_int_pre = None


        print("Generando estados...")
        # Seleccionar tipo de estado según configuración
        if ConfigEntorno.TIPO_ESTADO == 'advanced':
            all_states = create_all_states_advanced(fold_data, window_size, scaler, hora_int_pre)
            print(f"  [Advanced] Estados generados: {len(all_states)}")
        else:
            all_states = create_all_states_ohcl(fold_data, window_size, scaler, hora_int_pre)
            print(f"  [OHLC] Estados generados: {len(all_states)}")
        
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
        trader.step_counter = 0

        # No se borra la memoria ni se resetea epsilon entre folds
        # para que el agente acumule experiencia y explote lo aprendido
        reward_system = AdvancedRewardSystem(initial_balance=balance_first, weights=ConfigReward.get_pesos())
        reward_system.weights = ConfigReward.get_pesos()  # Sincronizar pesos desde parametros.py

        # Comienza los episodios
        for episode in range(1, episodes + 1):
            print(f"Episodio: {episode}/{episodes}")
            if live:
                live.reset_episode(fold + 1, episode)
            
            #reward_system.weights = reward_system.get_adaptive_weights(episode)
            # Crea las estadísticas del episodio
            # Estado inicial: inventario vacío → position features = [0, 0, 0]
            state = get_full_state(all_states[0], trader.inventory, trader.inventory_sell,
                                   close_prices[0], pip_value_eur_usd, balance_first)
            total_profit_pips = 0
            trades_count = 0
            wins = 0
            losses = 0
            worse_equity = 9999999
            profit_dollars = 0
            profit_dollars_total = 0
            winning_profits_pips = []
            losing_profits_pips = []
            peak_equity = balance_first
            current_equity = balance_first
            worst_mae_pips = 0.0
            worst_mae_usd  = 0.0
            drawdown_history_episode = []
            episode_returns_pips = []
            buy_points = []
            sell_points = []
            current_loss = 0
            force_closes = 0
            best_low =9999999
            best_high = 0
            
            trader.rewards_history.clear()
            trader.rewards_history_episode.clear()
            
            reward_system.sumaRecompensaProfit = 0
            reward_system.sumaRecompensaSharpe = 0
            reward_system.sumaRecompensaDrawndown = 0
            reward_system.sumaRecompensaConsistency =0
            reward_system.sumaRecompensaRiskAdjusted = 0
            reward_system.sumaRecompensaMomentum = 0
            reward_system.sumaRecompensaTradeQuality = 0
            
            #trader.total_rewards = 0
            
            reward_episode = 0
            if len(trader.inventory) > 0:
                trader.inventory.clear()
            
            if len(trader.inventory_sell) > 0:
                trader.inventory_sell.clear()
                

            # Bucle que recorre cada estado en los datos que descargue de mt5
            for t in range(data_samples):
            
                # La ia toma una decision
                action = trader.trade(state)

                # Comenzando la recompensa
                reward = 0
                # Precio actual
                current_price = close_prices[t]
                current_low = low_prices[t]
                current_high = high_prices[t]
                spread = spreads[t] * lot_size
                timestamp = timestamps[t]

                # Coloca el precio de compra actual con
                buy_price = buy_prices_pre[t] if action == 1 else current_price
                sell_price = sell_prices_pre[t] if action == 2 and len(trader.inventory) > 0 else current_price
                    
                if len(trader.inventory) > 0 and best_low > current_low:
                    best_low = current_low
                elif len(trader.inventory) <= 0:
                    best_low = 9999999
                    
                if len(trader.inventory_sell) > 0 and best_high < current_high:
                    best_high = current_high
                elif len(trader.inventory_sell) <= 0:
                    best_high = 0
           
                # Si la accion de la ia es igua a 1 compra
                if action == 1 and not trader.inventory:  # Comprar
                    trader.inventory.append(buy_price)
                    #reward += 0.01 if t < len(alcista_values) and alcista_values[t] > 0 else -0.01
                    if episode == episodes and fold == n_folds - 1:
                        buy_points.append((timestamp, buy_price))
                        
                #elif action == 0 and len(trader.inventory) <= 0 and t < len(alcista_values) and alcista_values[t] > 0:
                                #reward += -0.01
                        
                # Si la accion de la ia es igual a 2 y hay un trade abierto vende esa accion
                elif action == 2 and len(trader.inventory) > 0:  # Vender
                    original_buy_price = trader.inventory.pop(0)
                    
                    # USAR FUNCIÓN OPTIMIZADA PARA CÁLCULOS
                    profit_pips, profit_dollars = calculate_profit_fast(
                        original_buy_price, sell_price, pip_value_eur_usd, trader.commission_per_trade, lot_size
                    )
                    
                    pip_drawdrow_real, profit_drawdrow_real = calculate_profit_fast(
                        original_buy_price, best_low, pip_value_eur_usd, trader.commission_per_trade, lot_size
                    )
                    
                    # Actualizar variables
                    total_profit_pips += profit_pips
                    profit_dollars_total += profit_dollars
                    trades_count += 1
                    current_equity += profit_dollars
                    if pip_drawdrow_real < worst_mae_pips:
                        worst_mae_pips = pip_drawdrow_real
                        worst_mae_usd  = profit_drawdrow_real

                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    elif current_equity < worse_equity:
                        worse_equity = current_equity

                    episode_returns_pips.append(profit_pips)
                    
                    
                    reward, _ = calculate_advanced_reward(
                        reward_system, profit_dollars, current_equity, peak_equity,
                        episode_returns_pips, is_trade_closed=True, add_noise=False,
                        episode_wins=wins + (1 if profit_pips > 0 else 0),
                        episode_losses=losses + (1 if profit_pips <= 0 else 0)
                    )

                    if profit_pips > 0:
                        wins += 1
                        winning_profits_pips.append(profit_pips)
                    else:
                        losses += 1
                        losing_profits_pips.append(profit_pips)

                    if episode == episodes and fold == n_folds - 1:
                        sell_points.append((timestamp, sell_price))

                    # Reset best values
                    best_low = 9999999.0
                    best_high = 0.0

                elif action == 3 and len(trader.inventory_sell) <= 0:  # Venta en corto
                    trader.inventory_sell.append(sell_price)
                    #reward += 0.01 if t < len(alcista_values) and alcista_values[t] < 0 else -0.01
                    if episode == episodes and fold == n_folds - 1:
                        sell_points.append((timestamp, sell_price))
                        
                elif action == 4 and len(trader.inventory_sell) > 0:  # Cerrar venta en corto
                    original_sell_price = trader.inventory_sell.pop(0)

                    # USAR FUNCIÓN OPTIMIZADA
                    profit_pips, profit_dollars = calculate_short_profit_fast(
                        original_sell_price, buy_price, pip_value_eur_usd, trader.commission_per_trade, lot_size
                    )

                    pip_drawdrow_real, profit_drawdrow_real = calculate_short_profit_fast(
                        original_sell_price, best_high, pip_value_eur_usd, trader.commission_per_trade, lot_size
                    )

                    # Actualizar variables (mismo código que arriba)
                    total_profit_pips += profit_pips
                    profit_dollars_total += profit_dollars
                    trades_count += 1
                    current_equity += profit_dollars
                    if pip_drawdrow_real < worst_mae_pips:
                        worst_mae_pips = pip_drawdrow_real
                        worst_mae_usd  = profit_drawdrow_real

                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    elif current_equity < worse_equity:
                        worse_equity = current_equity

                    episode_returns_pips.append(profit_pips)


                    reward, _ = calculate_advanced_reward(
                        reward_system, profit_dollars, current_equity, peak_equity,
                        episode_returns_pips, is_trade_closed=True, add_noise=False,
                        episode_wins=wins + (1 if profit_pips > 0 else 0),
                        episode_losses=losses + (1 if profit_pips <= 0 else 0)
                    )

                    if profit_pips > 0:
                        wins += 1
                        winning_profits_pips.append(profit_pips)
                    else:
                        losses += 1
                        losing_profits_pips.append(profit_pips)

                    if episode == episodes and fold == n_folds - 1:
                        buy_points.append((timestamp, buy_price))

                    best_low = 9999999.0
                    best_high = 0.0

                elif (len(trader.inventory) > 0 or len(trader.inventory_sell) > 0) and has_time and \
                     t < len(hora_int_pre) and hora_int_pre[t] == 23:
                    
                    if len(trader.inventory) > 0:
                        original_buy_price = trader.inventory.pop(0)
                        profit_pips, profit_dollars = calculate_profit_fast(
                            original_buy_price, sell_price, pip_value_eur_usd, trader.commission_per_trade, lot_size
                        )
                        pip_drawdrow_real, profit_drawdrow_real = calculate_profit_fast(
                            original_buy_price, best_low, pip_value_eur_usd, trader.commission_per_trade, lot_size
                        )
                        
                    elif len(trader.inventory_sell) > 0:
                        original_sell_price = trader.inventory_sell.pop(0)
                        profit_pips, profit_dollars = calculate_short_profit_fast(
                            original_sell_price, buy_price, pip_value_eur_usd, trader.commission_per_trade, lot_size
                        )
                        pip_drawdrow_real, profit_drawdrow_real = calculate_short_profit_fast(
                            original_sell_price, best_high, pip_value_eur_usd, trader.commission_per_trade, lot_size
                        )
                    
                    # Misma lógica de actualización que arriba
                    total_profit_pips += profit_pips
                    profit_dollars_total += profit_dollars
                    trades_count += 1
                    current_equity += profit_dollars
                    if pip_drawdrow_real < worst_mae_pips:
                        worst_mae_pips = pip_drawdrow_real
                        worst_mae_usd  = profit_drawdrow_real

                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    elif current_equity < worse_equity:
                        worse_equity = current_equity

                    episode_returns_pips.append(profit_pips)
                    
                    
                    reward, reward_components = calculate_advanced_reward(
                        reward_system, profit_dollars, current_equity, peak_equity,
                        episode_returns_pips, is_trade_closed=True, add_noise=False,
                        episode_wins=wins + (1 if profit_pips > 0 else 0),
                        episode_losses=losses + (1 if profit_pips <= 0 else 0)
                    )
                    # Penalización por cierre forzado: el agente no decidió cerrar
                    reward -= 0.02
                    force_closes += 1

                    if profit_pips > 0:
                        wins += 1
                        winning_profits_pips.append(profit_pips)
                    else:
                        losses += 1
                        losing_profits_pips.append(profit_pips)

                    if episode == episodes and fold == n_folds - 1:
                        sell_points.append((timestamp, sell_price))
                
                                    
                drawdown = (peak_equity - current_equity) / peak_equity if peak_equity != 0 else 0
                drawdown_history_episode.append(drawdown)

                
                # Step reward: pequeña señal de P&L no realizado para reducir esparsidad.
                # Enseña al agente a mantener ganadores y cortar perdedores.
                if reward == 0:
                    if len(trader.inventory) > 0:
                        upnl = (current_price - trader.inventory[0]) * pip_value_eur_usd
                        reward += float(np.tanh(upnl / (balance_first * 0.02))) * 0.01
                    elif len(trader.inventory_sell) > 0:
                        upnl = (trader.inventory_sell[0] - current_price) * pip_value_eur_usd
                        reward += float(np.tanh(upnl / (balance_first * 0.02))) * 0.01

                # next_state incluye posición ACTUALIZADA tras la acción de este paso
                next_t = min(t + 1, len(all_states) - 1)
                next_price = close_prices[min(t + 1, data_samples - 1)]
                next_state = get_full_state(all_states[next_t], trader.inventory, trader.inventory_sell,
                                            next_price, pip_value_eur_usd, balance_first)

                # Ver si termino el episodio
                done = (t == data_samples - 1)
                trader.total_rewards += reward
                reward_episode += reward
                trader.remember(state, action, reward, next_state, done)
                state = next_state
                profit_dollars = 0

                if live:
                    live.update(
                        t, current_price, current_equity, action,
                        trader.inventory, trader.inventory_sell,
                        epsilon=trader.epsilon,
                        reward_episode=reward_episode,
                        trades=trades_count,
                        wins=wins,
                        losses=losses,
                        random_count=trader.random_action_count,
                        model_count=trader.model_action_count,
                        loss=trader.loss_history[-1] if trader.loss_history else 0.0,
                        is_random=trader.last_action_was_random,
                    )

                if len(trader.memory) > batch_size and t % train_frequency == 0:
                    for _ in range(train_iterations):
                        current_loss = trader.batch_train(batch_size)

                    if logger:
                        logger.log_train_step(trader.step_counter, current_loss, trader.learning_rate)

                    if trader.has_noise:
                        trader.model.reset_noise()
            
            # MOSTRAR PRINTS MENOS FRECUENTEMENTE
                if mostrar_prints and t % 3000 == 0:  # Aumenté la frecuencia
                    print(f"Tiempo {t}/{data_samples}, Episodio: {episode}, Recompensa: {reward_episode:.2f}, Equity: {current_equity:.2f}")

        # Calcular métricas finales del episodio
            sharpe = calculate_sharpe_ratio(np.array(episode_returns_pips)) if episode_returns_pips else 0
            accuracy = wins / trades_count if trades_count > 0 else 0
            avg_win = np.mean(winning_profits_pips) if winning_profits_pips else 0
            avg_loss = np.mean(losing_profits_pips) if losing_profits_pips else 0
            max_drawdown = max(drawdown_history_episode) if drawdown_history_episode else 0
            
            reward_system.reset_episode()
            
            # MOSTRAR RESULTADOS MENOS FRECUENTEMENTE
            # Solo cada 10 episodios o el último
            expectancy = (accuracy * avg_win) + ((1 - accuracy) * avg_loss) if trades_count > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            print(f"""
{'='*60}
  EPISODIO {episode}/{episodes}  |  FOLD {fold+1}/{n_folds}
{'='*60}
  [P&L]
    Beneficio pips  : {total_profit_pips:+.2f}
    Beneficio USD   : {price_format(profit_dollars_total)}
    Equity actual   : {current_equity:.2f}

  [TRADES]
    Total           : {trades_count}   |  Wins: {wins}   Losses: {losses}
    Accuracy        : {accuracy:.2%}
    Avg Win (pips)  : {avg_win:+.2f}
    Avg Loss (pips) : {avg_loss:+.2f}
    Profit Factor   : {profit_factor:.2f}
    Expectancy      : {expectancy:+.2f} pips/trade
    Force Closes    : {force_closes}  (cierres forzados hora 23)

  [RIESGO]
    Drawdown        : {max_drawdown:.2%}
    Peor MAE (pips) : {worst_mae_pips:+.2f}  ({worst_mae_usd:+.2f} USD)
    Sharpe          : {sharpe:.3f}

  [AGENTE]
    Epsilon         : {trader.epsilon:.4f}
    LR actual       : {trader.learning_rate:.6f}
    Loss (TD error) : {current_loss:.4f}

  [REWARD BREAKDOWN]
    Profit          : {reward_system.sumaRecompensaProfit:+.2f}
    Sharpe          : {reward_system.sumaRecompensaSharpe:+.2f}
    Drawdown        : {reward_system.sumaRecompensaDrawndown:+.2f}
    Consistencia    : {reward_system.sumaRecompensaConsistency:+.2f}
    Riesgo ajustado : {reward_system.sumaRecompensaRiskAdjusted:+.2f}
    Momentum        : {reward_system.sumaRecompensaMomentum:+.2f}
    Trade Quality   : {reward_system.sumaRecompensaTradeQuality:+.2f}
    ─────────────────────────────
    Episodio        : {reward_episode:+.2f}
    Acumulado       : {trader.total_rewards:+.2f}
{'='*60}
                """)
            
            # Decay de epsilon una vez por episodio
            if trader.epsilon > trader.epsilon_final:
                trader.epsilon *= trader.epsilon_decay

            # Actualizar scheduler por episodio (reduce_on_plateau lo requiere; los demás registran LR)
            trader.step_episode_scheduler(current_loss)

            # Guardar estadísticas
            trader.profit_history.append(profit_dollars_total)
            trader.epsilon_history.append(trader.epsilon)
            trader.trades_history.append(trades_count)
            trader.rewards_history.append(trader.total_rewards)
            trader.rewards_history_episode.append(reward_episode)
            trader.drawdown_history.append(max_drawdown)
            trader.sharpe_ratios.append(sharpe)
            trader.accuracy_history.append(accuracy)
            trader.avg_win_history.append(avg_win)
            trader.avg_loss_history.append(avg_loss)

            if logger:
                logger.log_episode(
                    fold=fold + 1, episode=episode,
                    profit_usd=profit_dollars_total, profit_pips=total_profit_pips,
                    equity=current_equity, sharpe=sharpe, drawdown=max_drawdown,
                    accuracy=accuracy, trades=trades_count,
                    epsilon=trader.epsilon, reward_episode=reward_episode
                )
            
            # GUARDAR MODELO MENOS FRECUENTEMENTE
            if episode % cada_cuantos_episodes_guardar_el_modelo == 0:  # Menos frecuente
                trader.plot_training_metrics(fold + 1 , save_path=resultados_dir)
                trader.save_model(os.path.join(resultados_dir, nombre_modelo_guardado))
                
                if guardar_estadisticas_en_backend:
                    status, result = upload_training_data(
                        url=ConfigBackend.URL_BACKEND,
                        model_file_path="no",
                        graph_image_paths=[],
                        episode=episode,
                        reward=trader.total_rewards,
                        loss=avg_loss,
                        profit_usd=current_equity,
                        epsilon=trader.epsilon,
                        drawdown=max_drawdown,
                        hit_frequency=accuracy
                    )
                
        # Plotting final
        if buy_points or sell_points:
            plot_trading_session(fold_data, buy_points, sell_points, symbol, intervalo, fold, save_path=resultados_dir)

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

        if logger:
            logger.log_fold_summary(
                fold=fold + 1,
                profit_usd=profit_dollars_total, profit_pips=total_profit_pips,
                sharpe=sharpe, drawdown=max_drawdown, accuracy=accuracy,
                avg_win=avg_win, avg_loss=avg_loss, trades=trades_count
            )

    print(f"\n{'='*30} Resultados de Validación Cruzada {'='*30}")
    metrics_df = pd.DataFrame(all_fold_metrics)
    print(metrics_df)
    print("\nPromedio de Métricas:")
    print(metrics_df.mean(numeric_only=True))

    # Evaluación final en el conjunto de prueba (opcional)
    print(f"\n{'='*30} Evaluación en Conjunto de Prueba {'='*30}")
    if len(test_data) > window_size:
        test_samples = len(test_data) - 1

        # Pre-calculate test data components
        test_close_prices = test_data['close'].values
        test_low_prices = test_data['low'].values
        test_high_prices = test_data['high'].values
        test_spreads = test_data['spread'].values
        test_timestamps = test_data.index.values

        # Check for 'time' and 'ema_diference_close' in test_data specifically
        has_time_test = 'time' in test_data.columns and test_data['time'].notnull().all()
        hora_test_values = test_data['time'].values if has_time_test else None
        hora_int_test_pre = np.array([int(h.split(":")[0]) for h in hora_test_values]) if has_time_test else None

        has_alcista_test = 'ema_diference_close' in test_data.columns and test_data['ema_diference_close'].notnull().all()
        alcista_test_values = test_data['ema_diference_close'].values if has_alcista_test else None

        print("Generando estados para el conjunto de prueba...")
        _hora_int_test = hora_int_test_pre if hora_int_test_pre is not None else np.zeros(len(test_data), dtype=np.int32)
        if ConfigEntorno.TIPO_ESTADO == 'advanced':
            test_states = create_all_states_advanced(test_data, window_size, global_scaler, _hora_int_test)
        else:
            test_states = create_all_states_ohcl(test_data, window_size, global_scaler, _hora_int_test)
        print(f"Estados de prueba generados: {len(test_states)} - Tipo: {ConfigEntorno.TIPO_ESTADO}")

        # Modo evaluación: sin dropout, sin exploración aleatoria
        trader.model.eval()
        epsilon_backup = trader.epsilon
        trader.epsilon = 0.0

        test_total_profit_pips = 0
        test_inventory = [] # Agent's inventory for test
        test_inventory_sell = [] # Agent's inventory for short positions for test
        test_trades = 0
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
        best_low_test = 9999999.0
        best_high_test = 0.0


        for t in range(test_samples):
            current_price_test = test_close_prices[t]
            # Estado con features de posición (igual que en entrenamiento)
            test_state_full = get_full_state(test_states[t], test_inventory, test_inventory_sell,
                                             current_price_test, pip_value_eur_usd, balance_first)
            test_action = trader.trade(test_state_full)
            current_low_test = test_low_prices[t]
            current_high_test = test_high_prices[t]
            spread_test = test_spreads[t]
            timestamp_test = test_timestamps[t]

            current_buy_exec_price_test = current_price_test + (spread_test * lot_size * 0.5)
            current_sell_exec_price_test = current_price_test - (spread_test * lot_size * 0.5)

            if len(test_inventory) > 0 and current_low_test < best_low_test:
                best_low_test = current_low_test
            elif len(test_inventory) <= 0:
                best_low_test = 9999999.0

            if len(test_inventory_sell) > 0 and current_high_test > best_high_test:
                best_high_test = current_high_test
            elif len(test_inventory_sell) <= 0:
                best_high_test = 0.0


            if test_action == 1 and not test_inventory: # Buy
                test_inventory.append(current_buy_exec_price_test)
                test_buy_points.append((timestamp_test, current_buy_exec_price_test))


            elif test_action == 2 and len(test_inventory) > 0: # Sell (close long)
                original_buy_price_test = test_inventory.pop(0)

                profit_test_pips, profit_test_dollars = calculate_profit_fast(
                    original_buy_price_test, current_sell_exec_price_test, pip_value_eur_usd, trader.commission_per_trade, lot_size
                )
                # For test, we don't need `pip_drawdrow_real` for reward calculation, but we can track for metrics
                _, _ = calculate_profit_fast(
                    original_buy_price_test, best_low_test, pip_value_eur_usd, trader.commission_per_trade, lot_size
                )


                test_total_profit_pips += profit_test_pips
                test_trades += 1
                test_profit_dollars_total += profit_test_dollars
                test_current_equity += profit_test_dollars

                if test_current_equity > test_peak_equity:
                    test_peak_equity = test_current_equity

                test_returns_pips.append(profit_test_pips)
                if profit_test_pips > 0:
                    wins_test += 1
                    winning_profits_test_pips.append(profit_test_pips)
                else:
                    losses_test += 1
                    losing_profits_test_pips.append(profit_test_pips)

                test_sell_points.append((timestamp_test, current_sell_exec_price_test))

                best_low_test = 9999999.0
                best_high_test = 0.0

            elif test_action == 3 and len(test_inventory_sell) <= 0: # Short Sell (open short)
                test_inventory_sell.append(current_sell_exec_price_test)
                test_sell_points.append((timestamp_test, current_sell_exec_price_test))

            elif test_action == 4 and len(test_inventory_sell) > 0: # Buy to Close Short
                original_sell_price_test = test_inventory_sell.pop(0)

                profit_test_pips, profit_test_dollars = calculate_short_profit_fast(
                    original_sell_price_test, current_buy_exec_price_test, pip_value_eur_usd, trader.commission_per_trade, lot_size
                )
                # For test, we don't need `pip_drawdrow_real` for reward calculation
                _, _ = calculate_short_profit_fast(
                    original_sell_price_test, best_high_test, pip_value_eur_usd, trader.commission_per_trade, lot_size
                )

                test_total_profit_pips += profit_test_pips
                test_trades += 1
                test_profit_dollars_total += profit_test_dollars
                test_current_equity += profit_test_dollars

                if test_current_equity > test_peak_equity:
                    test_peak_equity = test_current_equity

                test_returns_pips.append(profit_test_pips)
                if profit_test_pips > 0:
                    wins_test += 1
                    winning_profits_test_pips.append(profit_test_pips)
                else:
                    losses_test += 1
                    losing_profits_test_pips.append(profit_test_pips)

                test_buy_points.append((timestamp_test, current_buy_exec_price_test))

                best_low_test = 9999999.0
                best_high_test = 0.0

            # Time-based closure for test data, similar to training
            elif (len(test_inventory) > 0 or len(test_inventory_sell) > 0) and has_time_test and \
                 t < len(hora_int_test_pre) and hora_int_test_pre[t] == 23:

                if len(test_inventory) > 0: # Close long
                    original_buy_price_test = test_inventory.pop(0)
                    profit_test_pips, profit_test_dollars = calculate_profit_fast(
                        original_buy_price_test, current_sell_exec_price_test, pip_value_eur_usd, trader.commission_per_trade, lot_size
                    )
                    test_sell_points.append((timestamp_test, current_sell_exec_price_test))

                elif len(test_inventory_sell) > 0: # Close short
                    original_sell_price_test = test_inventory_sell.pop(0)
                    profit_test_pips, profit_test_dollars = calculate_short_profit_fast(
                        original_sell_price_test, current_buy_exec_price_test, pip_value_eur_usd, trader.commission_per_trade, lot_size
                    )
                    test_buy_points.append((timestamp_test, current_buy_exec_price_test))

                test_total_profit_pips += profit_test_pips
                test_trades += 1
                test_profit_dollars_total += profit_test_dollars
                test_current_equity += profit_test_dollars

                if test_current_equity > test_peak_equity:
                    test_peak_equity = test_current_equity

                test_returns_pips.append(profit_test_pips)
                if profit_test_pips > 0:
                    wins_test += 1
                    winning_profits_test_pips.append(profit_test_pips)
                else:
                    losses_test += 1
                    losing_profits_test_pips.append(profit_test_pips)

                best_low_test = 9999999.0
                best_high_test = 0.0

            # Calculate and append drawdown for testing
            test_current_drawdown = (test_peak_equity - test_current_equity) / test_peak_equity if test_peak_equity != 0 else 0
            test_drawdown_history.append(test_current_drawdown)

        # Restaurar modo entrenamiento y epsilon tras evaluación
        trader.model.train()
        trader.epsilon = epsilon_backup

        # Final calculations for the test set
        test_sharpe = calculate_sharpe_ratio(np.array(test_returns_pips)) if test_returns_pips else 0
        test_accuracy = wins_test / test_trades if test_trades > 0 else 0
        test_avg_win = np.mean(winning_profits_test_pips) if winning_profits_test_pips else 0
        test_avg_loss = np.mean(losing_profits_test_pips) if losing_profits_test_pips else 0
        test_max_drawdown = max(test_drawdown_history) if test_drawdown_history else 0

        print(f"""
    Resultados del Test Final:
    Beneficio (pips)={test_total_profit_pips:.2f}
    Beneficio (usd)={price_format(test_profit_dollars_total)}
    Trades={test_trades}
    Wins={wins_test}
    Sharpe={test_sharpe:.2f}
    Drawdown={test_max_drawdown:.2%}
    Accuracy={test_accuracy:.2%}
    Equity Final={test_current_equity:.2f}
        """)

        if logger:
            logger.log_test_results(
                profit_usd=test_profit_dollars_total, profit_pips=test_total_profit_pips,
                sharpe=test_sharpe, drawdown=test_max_drawdown, accuracy=test_accuracy,
                equity=test_current_equity, trades=test_trades
            )

        # Plotting the test trading session
        plot_trading_session(test_data, test_buy_points, test_sell_points, symbol, intervalo,1, save_path=resultados_dir,
                            )
    else:
        print("El conjunto de prueba es demasiado pequeño para realizar la evaluación.")

    if logger:
        logger.close()

    if live:
        live.close()

def run_main():
    main()

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        run_main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(20) # Muestra las 20 funciones más lentas
    
    
    
    
    
    
    
    
    
    
    
    