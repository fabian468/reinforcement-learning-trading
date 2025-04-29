# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:29:56 2025

@author: fabia
"""
import matplotlib.pyplot as plt

def plot_trading_session(data, buy_points, sell_points, symbol, timeframe):
    """Grafica la sesión de trading con puntos de compra y venta"""
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['close'].values, label=f'{symbol} - {timeframe} (Precio)')

    # Graficar el volumen en un segundo eje
    ax2 = plt.gca().twinx()
    ax2.bar(data.index, data['tick_volume'].values, color='gray', alpha=0.3, label='Volumen')
    ax2.set_ylabel('Volumen', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Puntos de compra en verde
    for point in buy_points:
        plt.scatter(point[0], point[1], color='green', s=100, marker='^')

    # Puntos de venta en rojo
    for point in sell_points:
        plt.scatter(point[0], point[1], color='red', s=100, marker='v')

    plt.title(f'Sesión de Trading - {symbol} ({timeframe})')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'trading_session_{symbol}_{timeframe}.png')
    plt.show()
