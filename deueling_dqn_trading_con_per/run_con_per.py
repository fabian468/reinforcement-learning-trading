# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 00:25:00 2025
@author: fabia
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import dropbox

from notificador import enviar_alerta

import cProfile
import pstats

load_dotenv() 

from dueling_dqn_con_per import AI_Trader_per 
from state_creator import  state_creator_vectorized 
from AdvancedRewardSystem import AdvancedRewardSystem , calculate_advanced_reward
from request_datos_backend import upload_training_data

DROPBOX_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN_DROPBOX")

dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def price_format(n):
    n = float(n)
    return "- {0:.3f}".format(abs(n)) if n < 0 else "{0:.3f}".format(abs(n))



def dataset_loader_csv(csv_path):
    try:
        df = pd.read_csv(f"data/{csv_path}", sep='\t')  # Usa tabulador como separador
        df.columns = [col.strip('<>').lower() for col in df.columns]  # Limpia < > y normaliza nombres

        # Verifica si existen las columnas necesarias
        if 'date' not in df.columns:
            print("Error: La columna 'date' es obligatoria.")
            return None

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
            return df[['time', 'close', 'tick_volume' , 'spread' , 'low']]
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


def plot_trading_session(data, buy_points, sell_points, symbol, timeframe, save_path='resultados_cv'):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Línea de precio
    ax.plot(data.index, data['close'], label='Precio', color='blue')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    # Flechas de compra
    for point in buy_points:
        ax.scatter(point[0], point[1], color='green', s=100, marker='^', label='Compra')

    # Flechas de venta
    for point in sell_points:
        ax.scatter(point[0], point[1], color='red', s=100, marker='v', label='Venta')

    # Título y formato
    ax.set_title(f'Sesión de Trading - {symbol} ({timeframe})')
    fig.autofmt_xdate()

    # Leyenda sin duplicados
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper left')

    # Mostrar y guardar
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'trading_session_{symbol}_{timeframe}.png'))
    plt.show()



def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0.0


#como puedo mejorar la convergencia el 

#comienzo del codigo
def main():
    
    nombre_csv = "XAUUSD_M15_2025_03_01_2025_03_31.csv"
       
    cargar_modelo = False
    modelo_existente = "resultados_cv/Oro_15_min_2025_03_01_2025_03_30_prueba"
    
    guardar_estadisticas_en_backend = True   
    guardar_en_dropbox = False
    
    symbol = "GOLD"
    intervalo = "daily"
    
    
    nombre_modelo_guardado = "Oro_15_min_2025_03_01_2025_03_30_prueba"
    
    
    es_indice = False
    es_forex = False
    es_metal = True
    tick_value = 5  
    pip_multiplier = 10000  # Para el Nasdaq (2 decimales)
    
  
    episodes = 1000
    n_folds = 3
    batch_size = 256
    epsilon_decay = 0.9995
    gamma = 0.95
    cada_cuanto_actualizar = 30
    learning_rate = 0.001
    window_size = 15

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
    
    state_size = (window_size - 1) * 2 + window_size + window_size + 2

    # Carga el modelo y ve si cargar uno o crear uno nuevo
    trader = AI_Trader_per(state_size,
                       epsilon_decay= epsilon_decay,
                       commission_per_trade= commission_per_trade,
                       gamma = gamma,
                       target_model_update = cada_cuanto_actualizar,
                       memory_size=10000, # Asegúrate de tener esto
                       alpha=0.6,
                       beta_start=0.4,
                       beta_frames=100000,
                       epsilon_priority=1e-6,
                       scheduler_type='exponential_decay',
                       learning_rate=learning_rate,
                       lr_decay_rate=0.97,      # LR se multiplica por 0.96 cada lr_decay_steps
                       lr_decay_steps=100,     # Cada 1000 pasos de entrenamiento
                       lr_min=1e-6
                       )

    if cargar_modelo:
        try:
            trader.load_model(modelo_existente)
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar el modelo {modelo_existente}: {str(e)}")

    
    fold_size = len(train_data) // n_folds
    all_fold_metrics = []
    

    # Comienza el entrenamiento del primero fold
    for fold in range(n_folds):
        print(f"\n{'='*30} Fold {fold + 1}/{n_folds} {'='*30}")
        start = fold * fold_size
        end = (fold + 1) * fold_size
        
        absolute_start = max(start, ventana_para_los_estados_de_datos)  # Garantiza que mínimo empiece desde fila 10
        fold_data = train_data.iloc[absolute_start:end].copy()

        data_samples = len(fold_data) - 1 

        states = [state_creator_vectorized(fold_data, t , window_size) for t in range(data_samples)] # Usa la función vectorizada

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
        trader.epsilon = 1.0
        trader.step_counter = 0

        trader.memory.clear()
        
        reward_system = AdvancedRewardSystem(initial_balance=balance_first)

        # Comienza los episodios
        for episode in range(1, episodes + 1):
            print(f"Episodio: {episode}/{episodes}")
            
            reward_system.weights = reward_system.get_adaptive_weights(episode, episodes)
            # Crea las estadísticas del episodio
            state = states[0]
            total_profit_pips = 0
            trader.inventory = []
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
            
            best_low =9999999
            
            trader.total_rewards = 0
            
            if len(trader.inventory) > 0:
                trader.inventory.clear()

            # Bucle que recorre cada estado en los datos que descargue de mt5
            for t in range(data_samples):
     
                # La ia toma una decision
                action = trader.trade(state)
                # Siguiente estado de la ia
                next_state = states[t + 1] if t + 1 < data_samples else state
            
                # Comenzando la recompensa
                reward = 0
                # Precio actual
                current_price = fold_data['close'].iloc[t].item()
                current_low = fold_data['low'].iloc[t].item()
                spread = fold_data['spread'].iloc[t].item() * lot_size
                # Indice del precio en el estado actual
                timestamp = fold_data.index[t]

                # Coloca el precio de compra actual con
                buy_price = current_price +  spread / 2 if action == 1 else current_price
                sell_price = current_price - spread / 2 if action == 2 and len(trader.inventory) > 0 else current_price
                
                
                if len(trader.inventory) > 0 and best_low > current_low:
                    best_low = current_low
                elif len(trader.inventory) <= 0:
                    best_low = 9999999
           
                # Si la accion de la ia es igua a 1 compra
                if action == 1 and not trader.inventory :  # Comprar
                    # Agrega el precio de compra al inventario
                    trader.inventory.append(buy_price)
                    print("compro")
                    reward, reward_components = calculate_advanced_reward(
                    reward_system, 0, current_equity, peak_equity,
                    episode_returns_pips, is_trade_closed=False
                    )
           
                    if episode == episodes and fold == n_folds -1 : buy_points.append((timestamp, buy_price))

                # Si la accion de la ia es igual a 2 y hay un trade abierto vende esa accion
                elif action == 2 and len(trader.inventory) > 0:  # Vender
                    # Toma el precio que compro el activo
                    original_buy_price = trader.inventory.pop(0)
                    # Calcula el profit que obtuvo de la venta de activo (en pips)
                    if(es_indice):
                        profit_pips = (sell_price - original_buy_price) 
                        ticks = profit_pips / 0.25   
                        profit_dollars = ticks * tick_value * (lot_size / 1.0)
                    elif(es_forex):
                        profit_pips = (sell_price - original_buy_price) * pip_multiplier
                        # Calcula la ganancia/pérdida en dólares
                        profit_dollars = profit_pips * pip_value_eur_usd
                    elif(es_metal):
                        profit_pips = (sell_price - original_buy_price) 
                        pip_drawdrow_real = (  best_low - original_buy_price)
                        # Calcula la ganancia/pérdida en dólares
                        profit_drawdrow_real = (pip_drawdrow_real * pip_value_eur_usd) - ( trader.commission_per_trade * lot_size) 
                        profit_dollars = (profit_pips * pip_value_eur_usd) - ( trader.commission_per_trade * lot_size) 
                        
                    # Coloco el profit en la variable (en pips)
                    total_profit_pips += profit_pips
                    # Usando la variable pip_value que ya definiste
                    profit_dollars_total += profit_dollars
                    
                    # Suma al contador de trades
                    trades_count += 1
       
                    current_drawdown_real +=profit_drawdrow_real
                    current_equity += profit_dollars  
                    
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    elif current_equity < worse_equity:
                        worse_equity = current_equity
                    
                    if current_drawdown_real > peak_equity_drawdrown_real:
                        peak_equity_drawdrown_real = current_drawdown_real

                    # Agrega el retorno al retorno del episodio cada profit (en pips)
                    episode_returns_pips.append(profit_pips)
                    
                    reward, reward_components = calculate_advanced_reward(
                            reward_system, profit_dollars, current_equity, peak_equity,
                            episode_returns_pips, is_trade_closed=True
                        )
                    # Verifica si el profit salio ganador agrefa uno a wins y agrega el profit a winning_profits (en pips)
                    if profit_pips > 0:
                        wins += 1
                        winning_profits_pips.append(profit_pips)
                    # De lo contrario si sale perdedor se agrega uno a lose y agrega el profit a losing_profits (en pips)
                    else:
                        losses += 1
                        losing_profits_pips.append(profit_pips)
                    if episode == episodes and fold == n_folds -1 : sell_points.append((timestamp, sell_price))
                        
            
                    print(f"tiempo {t} de {data_samples} , episodio: {episode} ,recompensa:{reward:.2f} , por eleccion de ia  Venta a {sell_price:.5f}, Compra a {original_buy_price:.5f}, Profit (pips): {profit_pips:.2f}, Profit (USD): {profit_dollars:.2f}, total de dinero actual {current_equity:.2f}")
                    print("")
                    print(f"suma de recompensa {trader.total_rewards}")
                    print("")
                    
                    
                elif 'time' in data.columns and data['time'].notnull().all() and int(hora.iloc[t].split(":")[0]) == 23 and len(trader.inventory) > 0:
                    # Toma el precio que compro el activo
                    original_buy_price = trader.inventory.pop(0)
                    # Calcula el profit que obtuvo de la venta de activo (en pips)
                    if(es_indice):
                        profit_pips = (sell_price - original_buy_price) 
                        ticks = profit_pips / 0.25   
                        profit_dollars = ticks * tick_value * (lot_size / 1.0)
                    elif(es_forex):
                        profit_pips = (sell_price - original_buy_price) * pip_multiplier
                        # Calcula la ganancia/pérdida en dólares
                        profit_dollars = profit_pips * pip_value_eur_usd
                    elif(es_metal):
                        profit_pips = (sell_price - original_buy_price) 
                        # Calcula la ganancia/pérdida en dólares
                        pip_drawdrow_real = (  best_low - original_buy_price)
                        profit_drawdrow_real = (pip_drawdrow_real * pip_value_eur_usd) - ( trader.commission_per_trade * lot_size)
                        profit_dollars =( profit_pips * pip_value_eur_usd) - ( trader.commission_per_trade * lot_size) 
                    
                    
                    # vender forzadamente por cierre de jornada, por ejemplo      
                    # Coloco el profit en la variable (en pips)
                    total_profit_pips += profit_pips
                    # Usando la variable pip_value que ya definiste
                    profit_dollars_total +=  profit_dollars
                    

                    # Suma al contador de trades
                    trades_count += 1
                    
                    current_drawdown_real +=profit_drawdrow_real
                    current_equity += profit_dollars # Asumiendo EUR es la base
                    
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    elif current_equity < worse_equity:
                        worse_equity = current_equity
                    
                    if current_drawdown_real > peak_equity_drawdrown_real:
                        peak_equity_drawdrown_real = current_drawdown_real

                    # Agrega el retorno al retorno del episodio cada profit (en pips)
                    episode_returns_pips.append(profit_pips)
                    
                    reward, reward_components = calculate_advanced_reward(
                            reward_system, profit_dollars, current_equity, peak_equity,
                            episode_returns_pips, is_trade_closed=True)
                    
                    # Verifica si el profit salio ganador agrefa uno a wins y agrega el profit a winning_profits (en pips)
                    if profit_pips > 0:
                        wins += 1
                        winning_profits_pips.append(profit_pips)
                    # De lo contrario si sale perdedor se agrega uno a lose y agrega el profit a losing_profits (en pips)
                    else:
                        losses += 1
                        losing_profits_pips.append(profit_pips)
                    if episode == episodes and fold == n_folds -1 : sell_points.append((timestamp, sell_price))
                
                    # Puedes imprimir o guardar la ganancia en dólares si lo deseas
                   #print("")
                   
                    print(f"tiempo {t} de {data_samples} , episodio: {episode} ,recompensa:{reward:.2f}, Venta a {sell_price:.5f}, Compra a {original_buy_price:.5f}, Profit (pips): {profit_pips:.2f}, Profit (USD): {profit_dollars:.2f}, total de dinero actual {current_equity:.2f}")
    
                
                #else:
                # Para acciones que no cierran trades
                    #reward, _ = calculate_advanced_reward(
                        #reward_system, 0, current_equity, peak_equity,
                        #episode_returns_pips, is_trade_closed=False
                        #)
                 
                    
                print(f"total de dinero actual {current_equity:.2f}")
                
                drawdown_real = (peak_equity_drawdrown_real - current_drawdown_real) / peak_equity_drawdrown_real if peak_equity_drawdrown_real != 0 else 0
                # Calculo de drawdown como ($1070 - $1050) / $1070 ≈ 0.0187 o 1.87%.
                drawdown = (peak_equity - current_equity) / peak_equity if peak_equity != 0 else 0
                # Se agrega al array para ver cual es el max drawdown 
                
              
                drawdown_history_episode.append(drawdown)
                drawdown_real_history_episode.append(drawdown_real)

                
                # Ver si termino el episodio
                done = (t == data_samples - 1)
                print(f"recompensas {reward}")
                trader.total_rewards += reward
                trader.remember(state, action, reward, next_state, done)
                state = next_state
                profit_dollars = 0
                if len(trader.memory) > batch_size:
                    #entrenamiento !!!!! 
                    trader.batch_train(batch_size)

            sharpe = calculate_sharpe_ratio(np.array(episode_returns_pips))
            accuracy = wins / trades_count if trades_count > 0 else 0
            avg_win = np.mean(winning_profits_pips) if winning_profits_pips else 0
            avg_loss = np.mean(losing_profits_pips) if losing_profits_pips else 0
            max_drawdown = max(drawdown_history_episode) if drawdown_history_episode else 0
            max_drawdown_real = max(drawdown_real_history_episode) if drawdown_real_history_episode else 0
            #if(episode % 5 == 0):
               # enviar_alerta(f"Jefe vamos en el episodio {episode} y el total de recompensa es de {trader.total_rewards:.2f}. con un total en dinero de: {price_format(profit_dollars_total)} ")
            reward_system.reset_episode()
              
   
            print("")
            print("===========================================================================")
            print(f"""Fin Episodio {episode}: 
                  Beneficio (pips)={total_profit_pips}
                  Beneficio (usd)={price_format(profit_dollars_total)}
                  Trades={trades_count}
                  Peor balance={worse_equity}
                  Mejor Balance = {peak_equity}
                  wins={wins}
                  Sharpe={sharpe:.2f}
                  Maximo drawdown = {max_drawdown_real:.2%}
                  Drawdown={max_drawdown:.2%}
                  Accuracy={accuracy:.2%}
                  """)
            print("")
            print(f"total de recompensas: {trader.total_rewards:.2f} ")
            print("===========================================================================")
            print("")
            trader.profit_history.append(profit_dollars_total)
            trader.epsilon_history.append(trader.epsilon)
            trader.trades_history.append(trades_count)
            trader.rewards_history.append(trader.total_rewards)
            #trader.loss_history.append(np.mean(trader.loss_history[-10:]) if trader.loss_history else 0)
            trader.drawdown_history.append(max_drawdown)
            trader.sharpe_ratios.append(sharpe)
            trader.accuracy_history.append(accuracy)
            trader.avg_win_history.append(avg_win)
            trader.avg_loss_history.append(avg_loss)
            
            if episode % 5 == 0  :
                trader.plot_training_metrics(save_path=resultados_dir )
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
                    
                    print('Status:', status)
                    print('Resultado:', result)
                
                if guardar_en_dropbox :
                    local_file_path =[ f'resultados_cv/{nombre_modelo_guardado}.h5',
                                      f'resultados_cv/{nombre_modelo_guardado}_params.txt' ,
                                      f'resultados_cv/{nombre_modelo_guardado}_target.h5' ,
                                      "resultados_cv/training_metrics.png"
                                      ]
                    
                    for file in local_file_path:
                        dropbox_destination_path = f"/{os.path.basename(file)}"
                        with open(file, 'rb') as f:
                            dbx.files_upload(f.read(), dropbox_destination_path, mode=dropbox.files.WriteMode.overwrite)
                    
                    print("Archivos subidos exitosamente a Dropbox.")
            
            

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
   
        test_states = np.array([state_creator_vectorized(test_data, t, window_size) for t in range(test_samples)]) # Usa la función vectorizada

        test_profit_pips = 0
        test_inventory = []
        test_trades = 0
        test_profit_dollars = 0
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
        
        if 'time' in test_data.columns and test_data['time'].notnull().all():
            hora_test = test_data['time']
        

        for t in range(test_samples):
            test_action = trader.trade(test_states[t])
            current_price = test_data['close'].iloc[t].item()
            spread_test = test_data['spread'].iloc[t].item() * lot_size
            timestamp = test_data.index[t]
            
            
            buy_price_test = current_price + spread_test / 2 if test_action == 1 else current_price
            sell_price_test = current_price - spread_test / 2 if test_action == 2 and len(test_inventory) > 0 else current_price
            
            if test_action == 1 and not test_inventory:
                test_inventory.append(buy_price_test)
            
                #if(not es_indice):
                   # test_current_equity -= buy_price_test * lot_size * 100000 * (1 + trader.commission_per_trade)
                test_buy_points.append((timestamp, buy_price_test))
                
                
            elif test_action == 2 and len(test_inventory) > 0:
                original_buy_price_test = test_inventory.pop(0)
                
                if (es_indice):
                    profit_test_pips = (sell_price_test - original_buy_price_test) 
                    ticks = profit_test_pips  / 0.25   
                    test_profit_dollars = ticks * tick_value * (lot_size / 1.0)
                elif(es_forex): 
                    profit_test_pips = (sell_price_test - original_buy_price_test) * pip_multiplier
                    test_profit_dollars = profit_test_pips * pip_value_eur_usd
                elif(es_metal):
                    profit_test_pips = (sell_price_test - original_buy_price_test) 
                    test_profit_dollars = (profit_test_pips *pip_value_eur_usd ) -(trader.commission_per_trade * lot_size)

                test_profit_pips += profit_test_pips
                test_trades += 1
                test_profit_dollars_total +=test_profit_dollars
                if (es_indice):
                    # Calcula cuanto dinero tiene (considerando el lote)
                    test_current_equity += test_profit_dollars 
                else:
                    test_current_equity += test_profit_dollars 
                
                test_returns_pips.append(profit_test_pips)
                if profit_test_pips > 0:
                    wins_test += 1
                    winning_profits_test_pips.append(profit_test_pips)
                else:
                    losses_test += 1
                    losing_profits_test_pips.append(profit_test_pips)
                
                test_sell_points.append((timestamp, sell_price_test))
            
            elif 'time' in test_data.columns and test_data['time'].notnull().all() and int(hora_test.iloc[t].split(":")[0]) == 23 and len(trader.inventory) > 0:
                original_buy_price_test = test_inventory.pop(0)
                
                if (es_indice):
                    profit_test_pips = (sell_price_test - original_buy_price_test) 
                    ticks = profit_pips / 0.25   
                    test_profit_dollars = ticks * tick_value * (lot_size / 1.0)
                elif(es_forex): 
                    profit_test_pips = (sell_price_test - original_buy_price_test) * pip_multiplier
                    test_profit_dollars = profit_test_pips * pip_value_eur_usd
                elif(es_metal):
                    profit_test_pips = (sell_price_test - original_buy_price_test) 
                    test_profit_dollars = (profit_test_pips *pip_value_eur_usd ) - (trader.commission_per_trade * lot_size)
                                    
                test_profit_pips += profit_test_pips
                test_trades += 1
                test_profit_dollars_total +=test_profit_dollars
                if (es_indice):
                    # Calcula cuanto dinero tiene (considerando el lote)
                    test_current_equity += test_profit_dollars 
                else:
                    test_current_equity += test_profit_dollars
                
                test_returns_pips.append(profit_test_pips)
                if profit_test_pips > 0:
                    wins_test += 1
                    winning_profits_test_pips.append(profit_test_pips)
                else:
                    losses_test += 1
                    losing_profits_test_pips.append(profit_test_pips)
                
                test_sell_points.append((timestamp, sell_price_test))


            if test_current_equity > test_peak_equity:
                test_peak_equity = test_current_equity
            test_drawdown = (test_peak_equity - test_current_equity) / test_peak_equity if test_peak_equity != 0 else 0

            test_drawdown_history.append(test_drawdown)

        test_sharpe = calculate_sharpe_ratio(np.array(test_returns_pips))
        test_accuracy = wins_test / test_trades if test_trades > 0 else 0
        test_max_drawdown = max(test_drawdown_history) if test_drawdown_history else 0
        avg_win_test = np.mean(winning_profits_test_pips) if winning_profits_test_pips else 0
        avg_loss_test = np.mean(losing_profits_test_pips) if losing_profits_test_pips else 0

        print(f"Resultados en Prueba: Beneficio (pips)={price_format(test_profit_pips)},Beneficio (dollar) = {test_profit_dollars_total} Trades={test_trades}, Sharpe={test_sharpe:.2f}, Drawdown={test_max_drawdown:.2%}, Accuracy={test_accuracy:.2%}")
        plot_trading_session(test_data, test_buy_points, test_sell_points, symbol, intervalo, save_path=resultados_dir)

def run_main():
    main()

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        run_main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(20) # Muestra las 20 funciones más lentas
    
    
    
    
    
    
    
    
    
    
    
    