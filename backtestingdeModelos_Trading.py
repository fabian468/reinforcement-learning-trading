# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 01:11:13 2025

@author: fabia
"""

# -*- coding: utf-8 -*-
"""
Script de Backtesting para AI Trader
Evalúa el desempeño de un modelo entrenado previamente

@author: fabia
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sys
import MetaTrader5 as mt5
from datetime import datetime
from collections import deque

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def price_format(n):
    n = float(n)
    if n < 0:
        return "- {0:.6f}".format(abs(n))
    else:
        return "{0:.6f}".format(abs(n))

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

def initialize_mt5():
    """Inicializa la conexión con MetaTrader 5"""
    if not mt5.initialize():
        print("Error al inicializar MetaTrader 5")
        print(f"Error: {mt5.last_error()}")
        return False
    else:
        print("MetaTrader 5 inicializado correctamente")
        print(mt5.version())
        return True

def dataset_loader_mt5(symbol, desde, hasta, timeframe):
    """Carga datos históricos desde MetaTrader 5"""
    
    # Convertir parámetros de fecha a formato datetime
    desde_dt = datetime.strptime(desde, "%Y-%m-%d")
    hasta_dt = datetime.strptime(hasta, "%Y-%m-%d")
    
    # Mapeo de intervalos a constantes MT5
    timeframe_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    
    # Verificar que el intervalo es válido
    if timeframe not in timeframe_map:
        print(f"Intervalo {timeframe} no válido. Opciones disponibles: {list(timeframe_map.keys())}")
        return None
    
    # Verificar que el símbolo existe en MT5
    symbols = mt5.symbols_get()
    symbol_names = [s.name for s in symbols]
    if symbol not in symbol_names:
        print(f"El símbolo {symbol} no está disponible en MetaTrader 5")
        print("Símbolos disponibles:", symbol_names[:10], "...")
        return None
    
    # Obtener datos históricos
    rates = mt5.copy_rates_range(symbol, timeframe_map[timeframe], desde_dt, hasta_dt)
    
    if rates is None or len(rates) == 0:
        print(f"No se pudieron obtener datos para {symbol} en el período especificado")
        return None
    
    # Crear DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    print(f"Datos cargados: {len(df)} registros para {symbol} desde {desde} hasta {hasta}")
    
    return df['close']

class AI_Trader_Backtester():
    def __init__(self, state_size, action_space=3):
        self.state_size = state_size
        self.action_space = action_space
        self.model = None
        
    def load_model(self, name):
        """Carga un modelo guardado"""
        self.model = tf.keras.models.load_model(f"{name}.h5", compile=False)
        # Recompilar el modelo para evitar problemas de serialización
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        print(f"Modelo cargado desde {name}.h5")
            
    def predict_action(self, state):
        """Predice la acción basada en el estado actual"""
        if len(state.shape) == 3 and state.shape[1] == 1:
            state = state.reshape(state.shape[0], state.shape[2])
            
        actions = self.model.predict(state, verbose=0)
        return np.argmax(actions[0])

def run_backtest(model_name, symbol, desde, hasta, timeframe, window_size, initial_balance=10000, commission=0.0001):
    """Ejecuta un backtest con el modelo y los parámetros dados"""
    
    # Inicializar MT5
    if not initialize_mt5():
        print("No se pudo inicializar MetaTrader 5. Finalizando backtesting.")
        return
    
    # Cargar datos
    data = dataset_loader_mt5(symbol, desde, hasta, timeframe)
    
    if data is None or len(data) < window_size:
        print("No hay suficientes datos para realizar el backtesting")
        mt5.shutdown()
        return
    
    # Crear el trader para backtesting
    state_size = window_size - 1
    backtester = AI_Trader_Backtester(state_size)
    
    # Cargar modelo
    try:
        backtester.load_model(model_name)
    except Exception as e:
        print(f"Error al cargar el modelo {model_name}: {str(e)}")
        mt5.shutdown()
        return
    
    # Inicializar variables para el backtesting
    balance = initial_balance
    inventory = []  # Almacena los precios de compra
    trade_history = []  # Almacenar todas las operaciones
    buy_points = []  # Para graficar
    sell_points = []  # Para graficar
    balance_history = [initial_balance]  # Seguimiento del balance
    
    # Métricas de rendimiento
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    breakeven_trades = 0
    max_drawdown = 0
    peak_balance = initial_balance
    
    # Calcular todos los estados de antemano
    print("Calculando estados para backtesting...")
    data_samples = len(data) - 1
    states = [state_creator(data, t, window_size) for t in range(data_samples)]
    print(f"Estados calculados: {len(states)}")
    
    # Realizar el backtesting
    print(f"Iniciando backtesting con {model_name} en {symbol} ({timeframe})")
    print(f"Período: {desde} a {hasta}")
    print(f"Balance inicial: ${initial_balance}")
    
    for t in range(data_samples):
        state = states[t]
        current_price = data.iloc[t].item()
        timestamp = data.index[t]
        
        # Predecir acción
        action = backtester.predict_action(state)
        
        # Ejecutar la acción
        if action == 1:  # Comprar
            # Verificar si tenemos fondos suficientes
            if balance > current_price:
                inventory.append(current_price)
                balance -= current_price * (1 + commission)  # Restar comisión
                total_trades += 1
                buy_points.append((timestamp, current_price))
                trade_history.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': current_price,
                    'balance': balance
                })
                print(f"COMPRA: {price_format(current_price)} @ {timestamp}")
        
        elif action == 2 and len(inventory) > 0:  # Vender
            buy_price = inventory.pop(0)
            profit = current_price - buy_price
            balance += current_price * (1 - commission)  # Restar comisión
            sell_points.append((timestamp, current_price))
            trade_history.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': current_price,
                'buy_price': buy_price,
                'profit': profit,
                'balance': balance
            })
            print(f"VENTA: {price_format(current_price)} @ {timestamp}, Profit: {price_format(profit)}")
            
            # Actualizar métricas
            if profit > 0:
                winning_trades += 1
            elif profit < 0:
                losing_trades += 1
            else:
                breakeven_trades += 1
        
        # Actualizar balance máximo y drawdown
        if balance > peak_balance:
            peak_balance = balance
        else:
            drawdown = (peak_balance - balance) / peak_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Guardar historial de balance
        balance_history.append(balance)
    
    # Calcular métricas finales
    profit_factor = 0
    win_rate = 0
    
    if total_trades > 0:
        win_rate = winning_trades / total_trades * 100
        
    if losing_trades > 0:
        profit_factor = winning_trades / losing_trades
    
    # Cerrar posiciones abiertas al finalizar el periodo
    final_balance = balance
    for position in inventory:
        final_price = data.iloc[-1].item()
        profit = final_price - position
        final_balance += final_price * (1 - commission)
        print(f"CIERRE DE POSICIÓN ABIERTA: {price_format(final_price)}, Profit: {price_format(profit)}")
    
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    # Mostrar resultados
    print("\n" + "="*50)
    print(f"RESULTADOS DEL BACKTESTING - {model_name}")
    print("="*50)
    print(f"Balance inicial: ${initial_balance}")
    print(f"Balance final: ${final_balance:.2f}")
    print(f"Retorno total: {total_return:.2f}%")
    print(f"Operaciones totales: {total_trades}")
    print(f"Operaciones ganadoras: {winning_trades} ({win_rate:.2f}%)")
    print(f"Operaciones perdedoras: {losing_trades}")
    print(f"Operaciones en punto de equilibrio: {breakeven_trades}")
    print(f"Factor de beneficio: {profit_factor:.2f}")
    print(f"Máximo drawdown: {max_drawdown*100:.2f}%")
    print("="*50)
    
    # Guardar resultados en CSV
    results_df = pd.DataFrame(trade_history)
    if not results_df.empty:
        results_df.to_csv(f"backtest_{model_name}_{symbol}_{timeframe}.csv", index=False)
        print(f"Resultados guardados en backtest_{model_name}_{symbol}_{timeframe}.csv")
    
    # Gráficos
    plot_backtest_results(data, buy_points, sell_points, balance_history, model_name, symbol, timeframe)
    
    # Cerrar MT5
    mt5.shutdown()
    print("Conexión con MetaTrader 5 cerrada")
    
    return {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_return': total_return,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown*100
    }

def plot_backtest_results(data, buy_points, sell_points, balance_history, model_name, symbol, timeframe):
    """Genera gráficos con los resultados del backtesting"""
    # Configuración para los gráficos
    plt.style.use('ggplot')
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Gráfico de precios y operaciones
    ax1.plot(data.index, data.values, label=f'{symbol} - {timeframe}', color='blue', alpha=0.7)
    
    # Puntos de compra en verde
    if buy_points:
        buy_dates = [point[0] for point in buy_points]
        buy_prices = [point[1] for point in buy_points]
        ax1.scatter(buy_dates, buy_prices, color='green', s=100, marker='^', label='Compra')
    
    # Puntos de venta en rojo
    if sell_points:
        sell_dates = [point[0] for point in sell_points]
        sell_prices = [point[1] for point in sell_points]
        ax1.scatter(sell_dates, sell_prices, color='red', s=100, marker='v', label='Venta')
    
    ax1.set_title(f'Backtest {model_name} - {symbol} ({timeframe})', fontsize=16)
    ax1.set_ylabel('Precio', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Gráfico de balance
    balance_dates = data.index[:len(balance_history)]
    ax2.plot(balance_dates, balance_history, color='green', label='Balance')
    ax2.set_title('Evolución del Balance', fontsize=14)
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Balance ($)', fontsize=12)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'backtest_{model_name}_{symbol}_{timeframe}.png', dpi=300)
    plt.show()
    print(f"Gráfico guardado como backtest_{model_name}_{symbol}_{timeframe}.png")

def main():
    # Parámetros del backtesting
    model_name = "ai_trader_dueling_dqn_40_4h_target"  # Nombre del modelo guardado (sin extensión)
    symbol = "EURUSD"               # Símbolo a testear
    desde = "2024-01-01"           # Fecha de inicio para el backtesting
    hasta = "2025-01-01"           # Fecha de fin para el backtesting
    timeframe = "4h"               # Timeframe a utilizar
    window_size = 11               # Debe coincidir con el tamaño de ventana usado en el entrenamiento
    initial_balance = 10000        # Balance inicial para el backtesting
    commission = 0.0001            # Comisión por operación (0.01%)
    
    # Ejecutar backtesting
    results = run_backtest(
        model_name=model_name,
        symbol=symbol,
        desde=desde,
        hasta=hasta,
        timeframe=timeframe,
        window_size=window_size,
        initial_balance=initial_balance,
        commission=commission
    )
    
    # Puedes añadir aquí más análisis con los resultados si lo deseas

if __name__ == "__main__":
    main()