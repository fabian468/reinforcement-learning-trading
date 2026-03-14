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
    from matplotlib.widgets import Button

    # --- Estado interno ---
    prices       = deque(maxlen=window_prices)
    equity       = []
    buy_markers  = []   # lista de (t, price)
    sell_markers = []   # lista de (t, price)
    trade_lines  = []   # lista de (t_open, px_open, t_close, px_close, is_profit)
    open_long    = None # (t, price) de la posición long abierta
    open_short   = None # (t, price) de la posición short abierta
    t_global     = 0
    fold         = 1
    episode      = 1
    paused       = [False]

    # --- Figura ---
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35,
                            top=0.93, bottom=0.12)
    ax_price  = fig.add_subplot(gs[0])
    ax_equity = fig.add_subplot(gs[1])
    fig.suptitle('Entrenamiento en vivo', fontsize=13, fontweight='bold')

    # --- Botón de pausa ---
    ax_btn = fig.add_axes([0.44, 0.02, 0.13, 0.045])
    btn    = Button(ax_btn, 'Pausa', color='#f0f0f0', hovercolor='#d0d0d0')

    def toggle_pause(event):
        paused[0] = not paused[0]
        btn.label.set_text('Reanudar' if paused[0] else 'Pausa')
        fig.canvas.draw_idle()

    btn.on_clicked(toggle_pause)
    plt.show(block=False)

    # --- Función de dibujo ---
    def draw():
        px = list(prices)
        n  = len(px)
        if n == 0:
            return

        start   = t_global - n + 1
        x_range = range(start, start + n)

        # Markers en la ventana visible
        bx = [t for t, p in buy_markers  if t >= start]
        by = [p for t, p in buy_markers  if t >= start]
        sx = [t for t, p in sell_markers if t >= start]
        sy = [p for t, p in sell_markers if t >= start]

        # Líneas de trades en la ventana visible (al menos un extremo visible)
        visible = [(t0, p0, t1, p1, ok)
                   for t0, p0, t1, p1, ok in trade_lines
                   if t1 >= start or t0 >= start]

        ax_price.clear()
        ax_price.plot(x_range, px, color='#4a90d9', linewidth=1, label='Precio')

        # Líneas de trades cerrados: entrada → salida
        for t0, p0, t1, p1, ok in visible:
            color = '#2ecc71' if ok else '#e74c3c'
            ax_price.plot([t0, t1], [p0, p1],
                          color=color, linewidth=1.5,
                          linestyle='--', alpha=0.75, zorder=3)

        # Línea "en vivo" para posición abierta (entrada → precio actual)
        current_price = px[-1] if px else None
        if current_price is not None:
            if open_long and open_long[0] >= start:
                unrealized = current_price - open_long[1]
                live_color = '#2ecc71' if unrealized >= 0 else '#e74c3c'
                ax_price.plot([open_long[0], t_global], [open_long[1], current_price],
                              color=live_color, linewidth=1.5,
                              linestyle=':', alpha=0.9, zorder=3)
            if open_short and open_short[0] >= start:
                unrealized = open_short[1] - current_price
                live_color = '#2ecc71' if unrealized >= 0 else '#e74c3c'
                ax_price.plot([open_short[0], t_global], [open_short[1], current_price],
                              color=live_color, linewidth=1.5,
                              linestyle=':', alpha=0.9, zorder=3)

        # Markers de compra / venta
        if bx:
            ax_price.scatter(bx, by, marker='^', color='#2ecc71',
                             s=100, zorder=5, label='Compra / Cierre short')
        if sx:
            ax_price.scatter(sx, sy, marker='v', color='#e74c3c',
                             s=100, zorder=5, label='Venta / Apertura short')

        estado = '  ⏸ PAUSADO' if paused[0] else ''
        ax_price.set_title(
            f'Fold {fold} | Episodio {episode} | Paso {t_global}{estado}',
            fontsize=10
        )
        ax_price.set_ylabel('Precio')
        ax_price.legend(loc='upper left', fontsize=8)
        ax_price.grid(True, alpha=0.3)

        # Panel equity
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

    # --- Loop principal del proceso ---
    while True:
        new_data = False

        # Siempre vaciamos la queue para no bloquear el training
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
                buy_markers.clear()
                sell_markers.clear()
                trade_lines.clear()
                open_long  = None
                open_short = None
                t_global   = 0
                new_data   = True

            elif kind == 'step':
                _, t, price, eq, action, inv, inv_sell = msg
                t_global = t
                prices.append(price)
                equity.append(eq)

                # El inventario ya está actualizado cuando llega el mensaje,
                # por eso chequeamos el estado POST-acción:
                if action == 1 and len(inv) == 1:          # compra ejecutada
                    open_long = (t, price)
                    buy_markers.append((t, price))

                elif action == 2 and len(inv) == 0:        # cierre long ejecutado
                    sell_markers.append((t, price))
                    if open_long:
                        is_profit = price > open_long[1]
                        trade_lines.append((open_long[0], open_long[1],
                                            t, price, is_profit))
                        open_long = None

                elif action == 3 and len(inv_sell) == 1:   # short ejecutado
                    open_short = (t, price)
                    sell_markers.append((t, price))

                elif action == 4 and len(inv_sell) == 0:   # cierre short ejecutado
                    buy_markers.append((t, price))
                    if open_short:
                        is_profit = price < open_short[1]  # short gana si baja
                        trade_lines.append((open_short[0], open_short[1],
                                            t, price, is_profit))
                        open_short = None

                if t % update_every == 0 and t > 0:
                    new_data = True

            elif kind == 'stop':
                plt.ioff()
                plt.close(fig)
                return

        # Dibujar solo si hay datos nuevos Y no está pausado
        if not paused[0] and new_data:
            draw()
        else:
            # Mantener el event loop activo (para el botón y el hover)
            plt.pause(0.05)


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
