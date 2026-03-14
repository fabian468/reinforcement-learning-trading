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
# PROCESO SEPARADO
# ==============================================================================

def _plot_process(queue, window_prices, update_every):

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.widgets import Button

    # --- Estado interno ---
    prices        = deque(maxlen=window_prices)
    equity        = []
    buy_markers   = []
    sell_markers  = []
    trade_lines   = []
    open_long     = None
    open_short    = None
    t_global      = 0
    fold          = 1
    episode       = 1
    paused        = [False]

    # Stats
    epsilon       = 1.0
    reward_ep     = 0.0
    trades        = 0
    wins          = 0
    losses        = 0
    random_count  = 0
    model_count   = 0
    loss_val      = 0.0

    # --- Figura ---
    plt.ion()
    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[4, 1.8, 1.2],
                            hspace=0.45, top=0.93, bottom=0.10)
    ax_price  = fig.add_subplot(gs[0])
    ax_equity = fig.add_subplot(gs[1])
    ax_stats  = fig.add_subplot(gs[2])
    fig.suptitle('Entrenamiento en vivo', fontsize=13, fontweight='bold')

    # Botón pausa
    ax_btn = fig.add_axes([0.44, 0.01, 0.13, 0.04])
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

        # Markers visibles
        bx = [t for t, p in buy_markers  if t >= start]
        by = [p for t, p in buy_markers  if t >= start]
        sx = [t for t, p in sell_markers if t >= start]
        sy = [p for t, p in sell_markers if t >= start]

        # Líneas de trades cerrados
        visible = [(t0, p0, t1, p1, ok) for t0, p0, t1, p1, ok in trade_lines
                   if t1 >= start or t0 >= start]

        # --- Panel precio ---
        ax_price.clear()
        ax_price.plot(x_range, px, color='#4a90d9', linewidth=1, label='Precio')

        for t0, p0, t1, p1, ok in visible:
            color = '#2ecc71' if ok else '#e74c3c'
            ax_price.plot([t0, t1], [p0, p1], color=color,
                          linewidth=1.5, linestyle='--', alpha=0.75, zorder=3)

        # Línea en vivo para posición abierta
        curr = px[-1] if px else None
        if curr is not None:
            if open_long and open_long[0] >= start:
                ok = curr >= open_long[1]
                ax_price.plot([open_long[0], t_global], [open_long[1], curr],
                              color='#2ecc71' if ok else '#e74c3c',
                              linewidth=1.5, linestyle=':', alpha=0.9, zorder=3)
            if open_short and open_short[0] >= start:
                ok = curr <= open_short[1]
                ax_price.plot([open_short[0], t_global], [open_short[1], curr],
                              color='#2ecc71' if ok else '#e74c3c',
                              linewidth=1.5, linestyle=':', alpha=0.9, zorder=3)

        if bx:
            ax_price.scatter(bx, by, marker='^', color='#2ecc71',
                             s=100, zorder=5, label='Compra / Cierre short')
        if sx:
            ax_price.scatter(sx, sy, marker='v', color='#e74c3c',
                             s=100, zorder=5, label='Venta / Apertura short')

        estado = '  ⏸ PAUSADO' if paused[0] else ''
        ax_price.set_title(
            f'Fold {fold} | Episodio {episode} | Paso {t_global}{estado}', fontsize=10)
        ax_price.set_ylabel('Precio')
        ax_price.legend(loc='upper left', fontsize=8)
        ax_price.grid(True, alpha=0.3)

        # --- Panel equity ---
        ax_equity.clear()
        eq = list(equity)
        if eq:
            color = '#2ecc71' if eq[-1] >= eq[0] else '#e74c3c'
            ax_equity.plot(eq, color=color, linewidth=1)
            ax_equity.axhline(y=eq[0], color='gray', linestyle='--', linewidth=0.8)
            ax_equity.set_ylabel('Equity ($)')
            ax_equity.set_xlabel('Paso')
        ax_equity.grid(True, alpha=0.3)

        # --- Panel stats ---
        ax_stats.clear()
        ax_stats.axis('off')

        total_actions = random_count + model_count
        pct_random = (random_count / total_actions * 100) if total_actions > 0 else 0
        pct_model  = (model_count  / total_actions * 100) if total_actions > 0 else 0
        win_rate   = (wins / trades * 100) if trades > 0 else 0
        eq_val     = eq[-1] if eq else 0.0
        pnl_color  = '#2ecc71' if eq_val >= (eq[0] if eq else eq_val) else '#e74c3c'

        stats = [
            ('Epsilon',   f'{epsilon:.3f}',          '#e67e22'),
            ('Reward ep', f'{reward_ep:.2f}',         '#2ecc71' if reward_ep >= 0 else '#e74c3c'),
            ('Loss',      f'{loss_val:.5f}',          '#9b59b6'),
            ('Trades',    f'{trades}',                '#4a90d9'),
            ('Wins',      f'{wins}  ({win_rate:.0f}%)', '#2ecc71'),
            ('Losses',    f'{losses}',                '#e74c3c'),
            ('Random',    f'{random_count}  ({pct_random:.0f}%)', '#e67e22'),
            ('Modelo',    f'{model_count}  ({pct_model:.0f}%)',   '#3498db'),
            ('Equity',    f'${eq_val:.2f}',           pnl_color),
        ]

        n_cols = len(stats)
        for i, (label, value, color) in enumerate(stats):
            x = (i + 0.5) / n_cols
            ax_stats.text(x, 0.72, label, ha='center', va='center',
                          fontsize=8, color='#888888', transform=ax_stats.transAxes)
            ax_stats.text(x, 0.28, value, ha='center', va='center',
                          fontsize=10, fontweight='bold', color=color,
                          transform=ax_stats.transAxes)
            # Separadores verticales
            if i < n_cols - 1:
                ax_stats.axvline(x=(i + 1) / n_cols, color='#dddddd',
                                 linewidth=0.8, transform=ax_stats.transAxes)

        ax_stats.set_facecolor('#f8f8f8')
        fig.patch.set_facecolor('#ffffff')

        plt.pause(0.001)

    # --- Loop principal ---
    while True:
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
                buy_markers.clear()
                sell_markers.clear()
                trade_lines.clear()
                open_long  = None
                open_short = None
                t_global   = 0
                new_data   = True

            elif kind == 'step':
                (_, t, price, eq, action, inv, inv_sell,
                 eps, rew, trd, w, l, rc, mc, lv) = msg

                t_global     = t
                epsilon      = eps
                reward_ep    = rew
                trades       = trd
                wins         = w
                losses       = l
                random_count = rc
                model_count  = mc
                loss_val     = lv

                prices.append(price)
                equity.append(eq)

                if action == 1 and len(inv) == 1:
                    open_long = (t, price)
                    buy_markers.append((t, price))
                elif action == 2 and len(inv) == 0:
                    sell_markers.append((t, price))
                    if open_long:
                        trade_lines.append((open_long[0], open_long[1],
                                            t, price, price > open_long[1]))
                        open_long = None
                elif action == 3 and len(inv_sell) == 1:
                    open_short = (t, price)
                    sell_markers.append((t, price))
                elif action == 4 and len(inv_sell) == 0:
                    buy_markers.append((t, price))
                    if open_short:
                        trade_lines.append((open_short[0], open_short[1],
                                            t, price, price < open_short[1]))
                        open_short = None

                if t % update_every == 0 and t > 0:
                    new_data = True

            elif kind == 'stop':
                plt.ioff()
                plt.close(fig)
                return

        if not paused[0] and new_data:
            draw()
        else:
            plt.pause(0.05)


# ==============================================================================
# CLASE PÚBLICA
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

    def update(self, t, price, current_equity, action, inventory, inventory_sell,
               epsilon=1.0, reward_episode=0.0, trades=0, wins=0, losses=0,
               random_count=0, model_count=0, loss=0.0):
        self._queue.put((
            'step', t, price, current_equity, action,
            list(inventory), list(inventory_sell),
            epsilon, reward_episode, trades, wins, losses,
            random_count, model_count, loss
        ))

    def close(self):
        self._queue.put(('stop',))
        self._proc.join(timeout=3)
        if self._proc.is_alive():
            self._proc.terminate()
