# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:44:01 2025

@author: fabia
"""

import matplotlib.pyplot as plt
import os

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
