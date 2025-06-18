# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 00:25:00 2025
@author: fabia
"""

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import dropbox

from notificador import enviar_alerta


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

from dueling_dqn_con_per import AI_Trader_per 
from state_creator import  state_creator_vectorized 
from AdvancedRewardSystem import AdvancedRewardSystem , calculate_advanced_reward
from request_datos_backend import upload_training_data
from plot_stadist import plot_trading_session

from indicadores import  add_ema200_distance

DROPBOX_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN_DROPBOX")

dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def price_format(n):
    n = float(n)
    return "- {0:.3f}".format(abs(n)) if n < 0 else "{0:.3f}".format(abs(n))



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
            return df[['time', 'close', 'tick_volume' , 'spread' , 'low' , 'ema_diference_close' , 'high']]
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
    
    nombre_csv = "XAUUSD_H1_2015_01_01_2024_05_31.csv"
    
    cargar_modelo = False
    modelo_existente = "resultados_cv/model_XAUUSD_H1_2015_01_01_2024_05_31.csv"
    
    cargar_memoria_buffer = True
    
    guardar_estadisticas_en_backend = True   
    guardar_en_dropbox = False
    mostrar_prints = False
    
    symbol = "GOLD"
    intervalo = "daily"
    
    nombre_modelo_guardado = "model_" + nombre_csv
    
    es_indice = False
    es_forex = False
    es_metal = True
    tick_value = 5  
    pip_multiplier = 10000  
    
    cada_cuantos_episodes_guardar_el_modelo = 5
  
    episodes =2000
    n_folds = 3
    batch_size = 256
    epsilon_decay = 0.99995
    gamma = 0.98
    cada_cuanto_actualizar = 200
    learning_rate = 0.001
    window_size = 18
    ventana_para_los_estados_de_datos = 4

    balance_first = 100 # dinero inicial
    lot_size = 0.01    
    commission_per_trade = 4.5
    test_size_ratio = 0.2  # 20% para prueba
    
    pip_value_eur_usd = 10 * lot_size
    
    # Creación de carpeta para guardar los resultados
    resultados_dir = 'resultados_cv'
    os.makedirs(resultados_dir, exist_ok=True)
    
    reward_system = AdvancedRewardSystem(initial_balance=balance_first)

    data = dataset_loader_csv(nombre_csv)
        
    # Calcular los datos para el entramiento y para el test el X e Y
    train_size = int(len(data) * (1 - test_size_ratio))
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()
    
    if 'time' in data.columns and data['time'].notnull().all():
        hora = data['time']
    
    if 'ema_diference_close' in data.columns and data['ema_diference_close'].notnull().all():
        alcista = data['ema_diference_close']
    
    
    state_size = (window_size - 1) * 4 + window_size + window_size + 4

    # Carga el modelo y ve si cargar uno o crear uno nuevo
    trader = AI_Trader_per(state_size,
                       epsilon_decay= epsilon_decay,
                       commission_per_trade= commission_per_trade,
                       gamma = gamma,
                       target_model_update = cada_cuanto_actualizar,
                       memory_size=200000, # Asegúrate de tener esto
                       alpha=0.6,
                       beta_start=0.4,
                       beta_frames=100000,
                       epsilon_priority=1e-3,
                       scheduler_type='cosine_decay',
                       learning_rate=learning_rate,
                       lr_decay_rate=0.97,      # LR se multiplica por 0.96 cada lr_decay_steps
                       lr_decay_steps=1000,     # Cada 1000 pasos de entrenamiento
                       lr_min=1e-5
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
        else:
            has_time = False


        print("Generando estados...")
        states = [state_creator_vectorized(fold_data, t, window_size) for t in range(data_samples)]
        print(f"Estados generados: {len(states)}")
        
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

        trader.memory.clear()
        
        reward_system = AdvancedRewardSystem(initial_balance=balance_first)

        # Comienza los episodios
        for episode in range(1, episodes + 1):
            print(f"Episodio: {episode}/{episodes}")
            
            #reward_system.weights = reward_system.get_adaptive_weights(episode)
            # Crea las estadísticas del episodio
            state = states[0]
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
            current_drawdown_real =balance_first
            peak_equity_drawdrown_real = balance_first
            drawdown_history_episode = []
            drawdown_real_history_episode = []
            episode_returns_pips = []
            buy_points = []
            sell_points = []
            current_loss = 0
            best_low =9999999
            best_high = 0
            
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
                # Siguiente estado de la ia
                next_state = states[t + 1] if t + 1 < data_samples else state
            
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
                    
                if len(trader.inventory) > 0 and best_high < current_high:
                    best_high = current_high
                elif len(trader.inventory) <= 0:
                    best_high = 0
           
                # Si la accion de la ia es igua a 1 compra
                if action == 1 and not trader.inventory:  # Comprar
                    trader.inventory.append(buy_price)
                    reward += 0.01 if t < len(alcista_values) and alcista_values[t] > 0 else -0.01
                    if episode == episodes and fold == n_folds - 1:
                        buy_points.append((timestamp, buy_price))
                        
                elif action == 0 and len(trader.inventory) <= 0 and t < len(alcista_values) and alcista_values[t] > 0:
                                reward += -0.01
                        
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
                    current_drawdown_real += profit_drawdrow_real
                    current_equity += profit_dollars
                    
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    elif current_equity < worse_equity:
                        worse_equity = current_equity
                    
                    if current_drawdown_real > peak_equity_drawdrown_real:
                        peak_equity_drawdrown_real = current_drawdown_real
    
                    episode_returns_pips.append(profit_pips)
                    
                    reward, _ = calculate_advanced_reward(
                        reward_system, profit_dollars, current_equity, peak_equity,
                        episode_returns_pips, is_trade_closed=True
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
                    reward += 0.01 if t < len(alcista_values) and alcista_values[t] < 0 else -0.01
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
                    current_drawdown_real += profit_drawdrow_real
                    current_equity += profit_dollars
                    
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    elif current_equity < worse_equity:
                        worse_equity = current_equity
                    
                    if current_drawdown_real > peak_equity_drawdrown_real:
                        peak_equity_drawdrown_real = current_drawdown_real
    
                    episode_returns_pips.append(profit_pips)
                    
                    reward, _ = calculate_advanced_reward(
                        reward_system, profit_dollars, current_equity, peak_equity,
                        episode_returns_pips, is_trade_closed=True
                    )
                    
                    if profit_pips > 0:
                        wins += 1
                        winning_profits_pips.append(profit_pips)
                    else:
                        losses += 1
                        losing_profits_pips.append(profit_pips)
                        
                    if episode == episodes and fold == n_folds - 1:
                        sell_points.append((timestamp, sell_price))
                    
                    best_low = 9999999.0
                    best_high = 0.0
                    
                    
                elif (len(trader.inventory) > 0 or len(trader.inventory_sell) > 0) and has_time and \
                     t < len(hora_values) and int(hora_values[t].split(":")[0]) == 23:
                    
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
                    current_drawdown_real += profit_drawdrow_real
                    current_equity += profit_dollars
                    
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    elif current_equity < worse_equity:
                        worse_equity = current_equity
                    
                    if current_drawdown_real > peak_equity_drawdrown_real:
                        peak_equity_drawdrown_real = current_drawdown_real
    
                    episode_returns_pips.append(profit_pips)
                    
                    reward, reward_components = calculate_advanced_reward(
                        reward_system, profit_dollars, current_equity, peak_equity,
                        episode_returns_pips, is_trade_closed=True
                    )
                    
                    if profit_pips > 0:
                        wins += 1
                        winning_profits_pips.append(profit_pips)
                    else:
                        losses += 1
                        losing_profits_pips.append(profit_pips)
                        
                    if episode == episodes and fold == n_folds - 1:
                        sell_points.append((timestamp, sell_price))
                
                                    
                drawdown_real = (peak_equity_drawdrown_real - current_drawdown_real) / peak_equity_drawdrown_real if peak_equity_drawdrown_real != 0 else 0
                drawdown = (peak_equity - current_equity) / peak_equity if peak_equity != 0 else 0
                
              
                drawdown_history_episode.append(drawdown)
                drawdown_real_history_episode.append(drawdown_real)

                
                # Ver si termino el episodio
                done = (t == data_samples - 1)
                trader.total_rewards += reward
                reward_episode += reward
                trader.remember(state, action, reward, next_state, done)
                state = next_state
                profit_dollars = 0
                
                
                
                if len(trader.memory) > batch_size and t % 10 == 0:  # Solo cada 10 pasos
                    current_loss = trader.batch_train(batch_size)
            
            # MOSTRAR PRINTS MENOS FRECUENTEMENTE
                if mostrar_prints and t % 3000 == 0:  # Aumenté la frecuencia
                    print(f"Tiempo {t}/{data_samples}, Episodio: {episode}, Recompensa: {reward_episode:.2f}, Equity: {current_equity:.2f}")

        # Calcular métricas finales del episodio
            sharpe = calculate_sharpe_ratio(np.array(episode_returns_pips)) if episode_returns_pips else 0
            accuracy = wins / trades_count if trades_count > 0 else 0
            avg_win = np.mean(winning_profits_pips) if winning_profits_pips else 0
            avg_loss = np.mean(losing_profits_pips) if losing_profits_pips else 0
            max_drawdown = max(drawdown_history_episode) if drawdown_history_episode else 0
            max_drawdown_real = max(drawdown_real_history_episode) if drawdown_real_history_episode else 0
            
            reward_system.reset_episode()
            
            # MOSTRAR RESULTADOS MENOS FRECUENTEMENTE
            # Solo cada 10 episodios o el último
            print(f"""
    Episodio {episode}: 
    Beneficio (pips)={total_profit_pips:.2f}
    Beneficio (usd)={price_format(profit_dollars_total)}
    Trades={trades_count}
    Wins={wins}
    Sharpe={sharpe:.2f}
    Drawdown={max_drawdown:.2%}
    Accuracy={accuracy:.2%}
    Equity={current_equity:.2f}
                """)
            
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
            
            # GUARDAR MODELO MENOS FRECUENTEMENTE
            if episode % cada_cuantos_episodes_guardar_el_modelo == 0:  # Menos frecuente
                trader.plot_training_metrics(save_path=resultados_dir)
                trader.save_model(os.path.join(resultados_dir, nombre_modelo_guardado))
                
                if guardar_estadisticas_en_backend:
                    status, result = upload_training_data(
                        url = 'https://back-para-entrenamiento.onrender.com/api/upload',
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
                
                if guardar_en_dropbox:
                    local_file_path =[
                        f'resultados_cv/{nombre_modelo_guardado}.h5',
                        f'resultados_cv/{nombre_modelo_guardado}_params.txt',
                        f'resultados_cv/{nombre_modelo_guardado}_target.h5',
                        "resultados_cv/training_metrics.png"
                    ]
                    
                    for file in local_file_path:
                        dropbox_destination_path = f"/{os.path.basename(file)}"
                        with open(file, 'rb') as f:
                            dbx.files_upload(f.read(), dropbox_destination_path, mode=dropbox.files.WriteMode.overwrite)

    # Plotting final
    if buy_points or sell_points:
        plot_trading_session(fold_data, buy_points, sell_points, symbol, intervalo, save_path=resultados_dir)

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

    print("\n{'='*30} Resultados de Validación Cruzada {'='*30}")
    metrics_df = pd.DataFrame(all_fold_metrics)
    print(metrics_df)
    print("\nPromedio de Métricas:")
    print(metrics_df.mean(numeric_only=True))

    # Evaluación final en el conjunto de prueba (opcional)
    print("\n{'='*30} Evaluación en Conjunto de Prueba {'='*30}")
    if len(test_data) > window_size:
        test_samples = len(test_data) - 1

        print("Generando estados para el conjunto de prueba...")
        test_states = np.array([state_creator_vectorized(test_data, t, window_size) for t in range(test_samples)])
        print(f"Estados de prueba generados: {len(test_states)}")

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

        # Pre-calculate test data components
        test_close_prices = test_data['close'].values
        test_low_prices = test_data['low'].values
        test_high_prices = test_data['high'].values
        test_spreads = test_data['spread'].values
        test_timestamps = test_data.index.values

        # Check for 'time' and 'ema_diference_close' in test_data specifically
        has_time_test = 'time' in test_data.columns and test_data['time'].notnull().all()
        hora_test_values = test_data['time'].values if has_time_test else None

        has_alcista_test = 'ema_diference_close' in test_data.columns and test_data['ema_diference_close'].notnull().all()
        alcista_test_values = test_data['ema_diference_close'].values if has_alcista_test else None


        for t in range(test_samples):
            # The agent is in evaluation mode, so epsilon is very low/0 (greedy action)
            test_action = trader.trade(test_states[t]) # Assuming AI_Trader has an is_eval mode for greedy actions
            current_price_test = test_close_prices[t]
            current_low_test = test_low_prices[t]
            current_high_test = test_high_prices[t]
            spread_test = test_spreads[t]
            timestamp_test = test_timestamps[t]

            current_buy_exec_price_test = current_price_test + (spread_test * 0.5)
            current_sell_exec_price_test = current_price_test - (spread_test * 0.5)

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
                 t < len(hora_test_values) and int(hora_test_values[t].split(":")[0]) == 23:

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

        # Plotting the test trading session
        plot_trading_session(test_data, test_buy_points, test_sell_points, symbol, intervalo, save_path=resultados_dir,
                            )
    else:
        print("El conjunto de prueba es demasiado pequeño para realizar la evaluación.")

def run_main():
    main()

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        run_main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(20) # Muestra las 20 funciones más lentas
    
    
    
    
    
    
    
    
    
    
    
    