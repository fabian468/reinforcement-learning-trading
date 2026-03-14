# -*- coding: utf-8 -*-
"""
Visualización en tiempo real de compras/ventas durante el entrenamiento.
Corre en un proceso separado para no bloquear el entrenamiento.
Para desactivar: ConfigBackend.LIVE_PLOT = False
Para eliminar: borrar este archivo y quitar las llamadas en run_con_per.py
"""

import multiprocessing as mp
import numpy as np
from collections import deque


# ==============================================================================
# PROCESO SEPARADO: corre el gráfico de forma independiente
# ==============================================================================

def _plot_process(queue, window_prices, update_every):
    """
    Función que corre en el proceso hijo.
    Lee mensajes de la queue y actualiza el gráfico.
    """
    # Importar matplotlib aquí para no afectar el backend del proceso principal
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    prices    = deque(maxlen=window_prices)
    equity    = []
    buy_xs    = []
    buy_ys    = []
    sell_xs   = []
    sell_ys   = []
    t_global  = 0
    fold      = 1
    episode   = 1

    plt.ion()
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax_price  = fig.add_subplot(gs[0])
    ax_equity = fig.add_subplot(gs[1])
    fig.suptitle('Entrenamiento en vivo', fontsize=13, fontweight='bold')
    plt.show(block=False)

    def draw():
        px = list(prices)
        n  = len(px)
        if n == 0:
            return

        start = t_global - n + 1

        # Filtrar markers en la ventana visible
        bx = [x for x in buy_xs  if x >= start]
        by = [buy_ys[buy_xs.index(x)]   for x in bx]
        sx = [x for x in sell_xs if x >= start]
        sy = [sell_ys[sell_xs.index(x)] for x in sx]

        ax_price.clear()
        ax_price.plot(range(start, start + n), px, color='#4a90d9', linewidth=1, label='Precio')
        if bx:
            ax_price.scatter(bx, by, marker='^', color='#2ecc71', s=80, zorder=5, label='Compra')
        if sx:
            ax_price.scatter(sx, sy, marker='v', color='#e74c3c', s=80, zorder=5, label='Venta')
        ax_price.set_title(f'Fold {fold} | Episodio {episode} | Paso {t_global}', fontsize=10)
        ax_price.set_ylabel('Precio')
        ax_price.legend(loc='upper left', fontsize=8)
        ax_price.grid(True, alpha=0.3)

        ax_equity.clear()
        eq = list(equity)
        if eq:
            color = '#2ecc71' if eq[-1] >= eq[0] else '#e74c3c'
            ax_equity.plot(eq, color=color, linewidth=1)
            ax_equity.axhline(y=eq[0], color='gray', linestyle='--', linewidth=0.8)
        ax_equity.set_ylabel('Equity ($)')
        ax_equity.set_xlabel('Paso')
        ax_equity.grid(True, alpha=0.3)

        plt.pause(0.001)

    while True:
        # Vaciar la queue completamente antes de dibujar
        new_data = False
        while not queue.empty():
            try:
                msg = queue.get_nowait()
            except Exception:
                break

            kind = msg[0]

            if kind == 'reset':
                _, fold, episode = msg
                prices.clear()
                equity.clear()
                buy_xs.clear()
                buy_ys.clear()
                sell_xs.clear()
                sell_ys.clear()
                t_global = 0
                new_data = True

            elif kind == 'step':
                _, t, price, eq, action, inv, inv_sell = msg
                t_global = t
                prices.append(price)
                equity.append(eq)

                # El inventario ya fue actualizado antes de llamar a update(),
                # por eso chequeamos el estado POST-acción:
                # action 1 (compra): inventory pasó de 0 → 1
                # action 2 (venta): inventory pasó de 1 → 0
                # action 3 (short): inventory_sell pasó de 0 → 1
                # action 4 (cover): inventory_sell pasó de 1 → 0
                if action == 1 and len(inv) == 1:
                    buy_xs.append(t); buy_ys.append(price)
                elif action == 2 and len(inv) == 0:
                    sell_xs.append(t); sell_ys.append(price)
                elif action == 3 and len(inv_sell) == 1:
                    sell_xs.append(t); sell_ys.append(price)
                elif action == 4 and len(inv_sell) == 0:
                    buy_xs.append(t); buy_ys.append(price)

                if t % update_every == 0 and t > 0:
                    new_data = True

            elif kind == 'stop':
                plt.ioff()
                plt.close(fig)
                return

        if new_data:
            draw()
        else:
            plt.pause(0.05)   # espera breve para no quemar CPU cuando no hay datos


# ==============================================================================
# CLASE PÚBLICA: usada desde run_con_per.py
# ==============================================================================

class LivePlot:
    def __init__(self, window_prices=150, update_every=100):
        self._queue = mp.Queue()
        self._proc  = mp.Process(
            target=_plot_process,
            args=(self._queue, window_prices, update_every),
            daemon=True
        )
        self._proc.start()

    def reset_episode(self, fold, episode):
        self._queue.put(('reset', fold, episode))

    def update(self, t, price, current_equity, action, inventory, inventory_sell):
        # Copiar listas para evitar compartir estado mutable entre procesos
        self._queue.put((
            'step', t, price, current_equity, action,
            list(inventory), list(inventory_sell)
        ))

    def close(self):
        self._queue.put(('stop',))
        self._proc.join(timeout=3)
        if self._proc.is_alive():
            self._proc.terminate()
