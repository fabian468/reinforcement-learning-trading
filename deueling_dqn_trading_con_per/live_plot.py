# -*- coding: utf-8 -*-
"""
Visualización en tiempo real de compras/ventas durante el entrenamiento.
Para desactivar: ConfigBackend.LIVE_PLOT = False
Para eliminar: borrar este archivo y quitar las llamadas en run_con_per.py
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class LivePlot:
    def __init__(self, window_prices=150, update_every=100):
        """
        Args:
            window_prices : cuántos pasos de precio mostrar en el gráfico
            update_every  : actualizar el gráfico cada N pasos
        """
        self.window_prices = window_prices
        self.update_every  = update_every

        # Buffers del episodio actual
        self.prices    = []
        self.equity    = []
        self.buy_ts    = []   # índices donde compró
        self.buy_px    = []   # precios de compra
        self.sell_ts   = []   # índices donde vendió
        self.sell_px   = []   # precios de venta

        # Configurar figura
        plt.ion()
        self.fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
        self.ax_price  = self.fig.add_subplot(gs[0])
        self.ax_equity = self.fig.add_subplot(gs[1])
        self.fig.suptitle('Entrenamiento en vivo', fontsize=13, fontweight='bold')
        plt.show(block=False)

    def reset_episode(self, fold, episode):
        """Llamar al inicio de cada episodio."""
        self.prices  = []
        self.equity  = []
        self.buy_ts  = []
        self.buy_px  = []
        self.sell_ts = []
        self.sell_px = []
        self._fold   = fold
        self._ep     = episode

    def update(self, t, price, current_equity, action, inventory, inventory_sell):
        """
        Llamar en cada paso t del loop.
        action: 0=hold, 1=buy, 2=sell(close long), 3=short, 4=cover(close short)
        """
        self.prices.append(price)
        self.equity.append(current_equity)

        if action == 1 and not inventory:          # acaba de comprar
            self.buy_ts.append(t)
            self.buy_px.append(price)
        elif action == 2 and len(inventory) == 0:  # acaba de cerrar long (ya hizo pop)
            self.sell_ts.append(t)
            self.sell_px.append(price)
        elif action == 3 and not inventory_sell:   # acaba de abrir short
            self.sell_ts.append(t)
            self.sell_px.append(price)
        elif action == 4 and len(inventory_sell) == 0:  # acaba de cubrir short
            self.buy_ts.append(t)
            self.buy_px.append(price)

        if t % self.update_every == 0 and t > 0:
            self._draw(t)

    def _draw(self, t):
        # Ventana de precios a mostrar
        start = max(0, len(self.prices) - self.window_prices)
        px_window = self.prices[start:]
        x_range   = range(start, start + len(px_window))

        # Filtrar markers dentro de la ventana visible
        buy_x  = [i for i in self.buy_ts  if i >= start]
        buy_y  = [self.buy_px[self.buy_ts.index(i)]  for i in buy_x]
        sell_x = [i for i in self.sell_ts if i >= start]
        sell_y = [self.sell_px[self.sell_ts.index(i)] for i in sell_x]

        # --- Panel precio ---
        self.ax_price.clear()
        self.ax_price.plot(x_range, px_window, color='#4a90d9', linewidth=1, label='Precio')
        if buy_x:
            self.ax_price.scatter(buy_x, buy_y, marker='^', color='#2ecc71',
                                  s=80, zorder=5, label='Compra')
        if sell_x:
            self.ax_price.scatter(sell_x, sell_y, marker='v', color='#e74c3c',
                                  s=80, zorder=5, label='Venta')
        self.ax_price.set_title(
            f'Fold {self._fold} | Episodio {self._ep} | Paso {t}',
            fontsize=10
        )
        self.ax_price.set_ylabel('Precio')
        self.ax_price.legend(loc='upper left', fontsize=8)
        self.ax_price.grid(True, alpha=0.3)

        # --- Panel equity ---
        self.ax_equity.clear()
        eq = self.equity
        color = '#2ecc71' if eq[-1] >= eq[0] else '#e74c3c'
        self.ax_equity.plot(eq, color=color, linewidth=1)
        self.ax_equity.axhline(y=eq[0], color='gray', linestyle='--', linewidth=0.8)
        self.ax_equity.set_ylabel('Equity ($)')
        self.ax_equity.set_xlabel('Paso')
        self.ax_equity.grid(True, alpha=0.3)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)
