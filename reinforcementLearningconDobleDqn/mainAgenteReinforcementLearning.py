# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:32:14 2025

@author: fabia
"""

from metatradeAgenteReinforcementLearning import initialize_mt5
from dataLoaderAgenteReinforcementLearning import dataset_loader_mt5
from agenteReinforcementLearning import AI_Trader
from stateCreatorAgenteReinforcementLearning import state_creator
from utilsAgenteReinforcementLearning import price_format
from plotAgenteReinforcementLearning import plot_trading_session


import MetaTrader5 as mt5
import sys
import pandas as pd

def main():
    # Inicializar MT5
    if not initialize_mt5():
        print("No se pudo inicializar MetaTrader 5. Finalizando programa.")
        sys.exit(1)

    # Parámetros
    symbol = "EURUSD"  # Símbolo como aparece en MT5
    desde = "2023-12-01"
    hasta = "2024-01-01"
    intervalo = "4h"  # Usar "1m", "5m", "15m", "30m", "1h", "4h" o "1d"

    # Cargar datos
    data = dataset_loader_mt5(symbol, desde, hasta, intervalo)

    if data is None or len(data) < 50:  # Verificar que tenemos suficientes datos
        print("No hay suficientes datos para entrenar el modelo")
        mt5.shutdown()
        sys.exit(1)

    # El estado tendrá tamaño (window_size - 1) * número de características (precio y volumen)
    window_size = 11
    state_size = (window_size - 1) * 2
    episodes = 2
    batch_size = 32
    data_samples = len(data) - 1

    # Configuración para cargar modelo existente
    cargar_modelo = False  # Cambiar a True para cargar un modelo existente
    modelo_existente = "ai_trader_dueling_dqn_40_4h"  # Nombre del modelo a cargar (sin extensión)

    trader = AI_Trader(state_size)

    # Cargar modelo existente si se especifica
    if cargar_modelo:
        try:
            trader.load_model(modelo_existente)
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar el modelo {modelo_existente}: {str(e)}")
            print("Deteniendo la ejecución...")
            mt5.shutdown()
            sys.exit(1)

    # Precalcular todos los estados
    print("Calculando estados...")
    states = [state_creator(data, t, window_size) for t in range(data_samples)]
    print(f"Estados calculados: {len(states)}")

    # Para graficar la sesión de trading
    buy_points = []  # Lista de tuplas (fecha, precio)
    sell_points = []  # Lista de tuplas (fecha, precio)

    for episode in range(1, episodes + 1):
        print("Episodio: {}/{}".format(episode, episodes))

        state = states[0]
        total_profit = 0
        trader.inventory = []
        trades_count = 0  # Contador de operaciones en este episodio

        # Limpiar puntos de compra/venta para cada episodio si queremos visualizar solo el último
        if episode == episodes:  # Solo guardar puntos para el último episodio
            buy_points = []
            sell_points = []

        for t in range(data_samples):
            action = trader.trade(state)
            next_state = states[t + 1] if t + 1 < data_samples else state
            reward = 0
            current_price = data['close'].iloc[t].item()
            timestamp = data.index[t]  # Obtener la fecha/hora del índice

            if action == 1:  # Comprar
                trader.inventory.append(current_price)
                trades_count += 1
                print(f"Episodio: {episode}, AI Trader compró: {price_format(current_price)}")

                # Guardar punto de compra para graficar (solo último episodio)
                if episode == episodes:
                    buy_points.append((timestamp, current_price))

            elif action == 2 and len(trader.inventory) > 0:  # Vender
                buy_price = trader.inventory.pop(0)
                reward = max(current_price - buy_price, 0)
                total_profit += current_price - buy_price
                trades_count += 1
                print(f"AI Trader vendió: {price_format(current_price)}, Beneficio: {price_format(current_price - buy_price)}")

                # Guardar punto de venta para graficar (solo último episodio)
                if episode == episodes:
                    sell_points.append((timestamp, current_price))

            done = (t == data_samples - 1)
            trader.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("########################")
                print(f"BENEFICIO TOTAL: {price_format(total_profit)}")
                print(f"OPERACIONES REALIZADAS: {trades_count}")
                print("########################")

                # Guardar datos para graficar evolución
                trader.profit_history.append(total_profit)
                trader.epsilon_history.append(trader.epsilon)
                trader.trades_history.append(trades_count)

            if len(trader.memory) > batch_size:
                trader.batch_train(batch_size)

        # Guardar el modelo cada 20 episodios para no sobrecargar el proceso
        if episode % 20 == 0:
            trader.save_model(f"ai_trader_dueling_dqn_{episode}_{intervalo}")

    # Guardar el modelo final
    #trader.save_model(f"ai_trader_dueling_dqn_final_{intervalo}")

    # Graficar evolución del entrenamiento
    trader.plot_training_evolution()

    # Graficar la sesión de trading del último episodio
    if buy_points or sell_points:
        plot_trading_session(data, buy_points, sell_points, symbol, intervalo)

    # Cerrar la conexión con MT5 al finalizar
    mt5.shutdown()
    print("Conexión con MetaTrader 5 cerrada")

if __name__ == "__main__":
    main()