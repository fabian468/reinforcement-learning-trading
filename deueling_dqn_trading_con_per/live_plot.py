# -*- coding: utf-8 -*-
"""
Visualización en tiempo real de compras/ventas durante el entrenamiento.
Corre en el hilo principal para máxima compatibilidad con Spyder/Windows.
Para desactivar: ConfigBackend.LIVE_PLOT = False
Para eliminar: borrar este archivo y quitar las llamadas en run_con_per.py
"""

import numpy as np
from collections import deque


class LivePlot:
    def __init__(self, window_prices=150, update_every=100):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.widgets import Button

        self.plt     = plt
        self.update_every = update_every

        # --- Buffers ---
        self.prices       = deque(maxlen=window_prices)
        self.equity       = []
        self.buy_markers  = []
        self.sell_markers = []
        self.trade_lines  = []
        self.open_long    = None
        self.open_short   = None
        self.t_global     = 0
        self.fold         = 1
        self.episode      = 1
        self.paused       = False

        # Stats
        self.epsilon      = 1.0
        self.reward_ep    = 0.0
        self.trades       = 0
        self.wins         = 0
        self.losses       = 0
        self.random_count = 0
        self.model_count  = 0
        self.loss_val     = 0.0

        # --- Figura ---
        plt.ion()
        self.fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(3, 1, height_ratios=[4, 1.8, 1.2],
                               hspace=0.45, top=0.93, bottom=0.10)
        self.ax_price  = self.fig.add_subplot(gs[0])
        self.ax_equity = self.fig.add_subplot(gs[1])
        self.ax_stats  = self.fig.add_subplot(gs[2])
        self.fig.suptitle('Entrenamiento en vivo', fontsize=13, fontweight='bold')

        # Botón pausa
        ax_btn   = self.fig.add_axes([0.44, 0.01, 0.13, 0.04])
        self.btn = Button(ax_btn, 'Pausa', color='#f0f0f0', hovercolor='#d0d0d0')
        self.btn.on_clicked(self._toggle_pause)

        plt.show(block=False)
        plt.pause(0.1)

    def _toggle_pause(self, event):
        self.paused = not self.paused
        self.btn.label.set_text('Reanudar' if self.paused else 'Pausa')
        self.fig.canvas.draw_idle()

    def reset_episode(self, fold, episode):
        self.fold    = fold
        self.episode = episode
        self.prices.clear()
        self.equity.clear()
        self.buy_markers.clear()
        self.sell_markers.clear()
        self.trade_lines.clear()
        self.open_long  = None
        self.open_short = None
        self.t_global   = 0

    def update(self, t, price, current_equity, action, inventory, inventory_sell,
               epsilon=1.0, reward_episode=0.0, trades=0, wins=0, losses=0,
               random_count=0, model_count=0, loss=0.0):

        self.t_global     = t
        self.epsilon      = epsilon
        self.reward_ep    = reward_episode
        self.trades       = trades
        self.wins         = wins
        self.losses       = losses
        self.random_count = random_count
        self.model_count  = model_count
        self.loss_val     = loss

        self.prices.append(price)
        self.equity.append(current_equity)

        # Registrar acción
        inv      = list(inventory)
        inv_sell = list(inventory_sell)

        if action == 1 and len(inv) == 1:
            self.open_long = (t, price)
            self.buy_markers.append((t, price))
        elif action == 2 and len(inv) == 0:
            self.sell_markers.append((t, price))
            if self.open_long:
                self.trade_lines.append((self.open_long[0], self.open_long[1],
                                         t, price, price > self.open_long[1]))
                self.open_long = None
        elif action == 3 and len(inv_sell) == 1:
            self.open_short = (t, price)
            self.sell_markers.append((t, price))
        elif action == 4 and len(inv_sell) == 0:
            self.buy_markers.append((t, price))
            if self.open_short:
                self.trade_lines.append((self.open_short[0], self.open_short[1],
                                         t, price, price < self.open_short[1]))
                self.open_short = None

        # Dibujar cada N pasos (si no está pausado)
        if not self.paused and t % self.update_every == 0 and t > 0:
            self._draw()

    def _draw(self):
        px = list(self.prices)
        n  = len(px)
        if n == 0:
            return

        start   = self.t_global - n + 1
        x_range = range(start, start + n)

        # Markers visibles
        bx = [t for t, p in self.buy_markers  if t >= start]
        by = [p for t, p in self.buy_markers  if t >= start]
        sx = [t for t, p in self.sell_markers if t >= start]
        sy = [p for t, p in self.sell_markers if t >= start]

        # Trades cerrados visibles
        visible = [(t0, p0, t1, p1, ok) for t0, p0, t1, p1, ok in self.trade_lines
                   if t1 >= start or t0 >= start]

        # --- Panel precio ---
        self.ax_price.clear()
        self.ax_price.plot(x_range, px, color='#4a90d9', linewidth=1, label='Precio')

        for t0, p0, t1, p1, ok in visible:
            self.ax_price.plot([t0, t1], [p0, p1],
                               color='#2ecc71' if ok else '#e74c3c',
                               linewidth=1.5, linestyle='--', alpha=0.75, zorder=3)

        # Línea en vivo para posición abierta
        curr = px[-1]
        if self.open_long and self.open_long[0] >= start:
            ok = curr >= self.open_long[1]
            self.ax_price.plot([self.open_long[0], self.t_global],
                               [self.open_long[1], curr],
                               color='#2ecc71' if ok else '#e74c3c',
                               linewidth=1.5, linestyle=':', alpha=0.9, zorder=3)
        if self.open_short and self.open_short[0] >= start:
            ok = curr <= self.open_short[1]
            self.ax_price.plot([self.open_short[0], self.t_global],
                               [self.open_short[1], curr],
                               color='#2ecc71' if ok else '#e74c3c',
                               linewidth=1.5, linestyle=':', alpha=0.9, zorder=3)

        if bx:
            self.ax_price.scatter(bx, by, marker='^', color='#2ecc71',
                                  s=100, zorder=5, label='Compra / Cierre short')
        if sx:
            self.ax_price.scatter(sx, sy, marker='v', color='#e74c3c',
                                  s=100, zorder=5, label='Venta / Apertura short')

        estado = '  ⏸ PAUSADO' if self.paused else ''
        self.ax_price.set_title(
            f'Fold {self.fold} | Episodio {self.episode} | Paso {self.t_global}{estado}',
            fontsize=10)
        self.ax_price.set_ylabel('Precio')
        self.ax_price.legend(loc='upper left', fontsize=8)
        self.ax_price.grid(True, alpha=0.3)

        # --- Panel equity ---
        self.ax_equity.clear()
        eq = list(self.equity)
        if eq:
            color = '#2ecc71' if eq[-1] >= eq[0] else '#e74c3c'
            self.ax_equity.plot(eq, color=color, linewidth=1)
            self.ax_equity.axhline(y=eq[0], color='gray', linestyle='--', linewidth=0.8)
        self.ax_equity.set_ylabel('Equity ($)')
        self.ax_equity.set_xlabel('Paso')
        self.ax_equity.grid(True, alpha=0.3)

        # --- Panel stats ---
        self.ax_stats.clear()
        self.ax_stats.axis('off')

        total   = self.random_count + self.model_count
        pct_r   = (self.random_count / total * 100) if total > 0 else 0
        pct_m   = (self.model_count  / total * 100) if total > 0 else 0
        win_pct = (self.wins / self.trades * 100) if self.trades > 0 else 0
        eq_val  = eq[-1] if eq else 0.0
        pnl_col = '#2ecc71' if (len(eq) < 2 or eq[-1] >= eq[0]) else '#e74c3c'

        stats = [
            ('Epsilon',    f'{self.epsilon:.3f}',                   '#e67e22'),
            ('Reward ep',  f'{self.reward_ep:.2f}',                 '#2ecc71' if self.reward_ep >= 0 else '#e74c3c'),
            ('Loss',       f'{self.loss_val:.5f}',                  '#9b59b6'),
            ('Trades',     f'{self.trades}',                        '#4a90d9'),
            ('Wins',       f'{self.wins}  ({win_pct:.0f}%)',        '#2ecc71'),
            ('Losses',     f'{self.losses}',                        '#e74c3c'),
            ('Random',     f'{self.random_count}  ({pct_r:.0f}%)', '#e67e22'),
            ('Modelo',     f'{self.model_count}  ({pct_m:.0f}%)',  '#3498db'),
            ('Equity',     f'${eq_val:.2f}',                       pnl_col),
        ]

        n_cols = len(stats)
        for i, (label, value, color) in enumerate(stats):
            x = (i + 0.5) / n_cols
            self.ax_stats.text(x, 0.72, label, ha='center', va='center',
                               fontsize=8, color='#888888',
                               transform=self.ax_stats.transAxes)
            self.ax_stats.text(x, 0.28, value, ha='center', va='center',
                               fontsize=10, fontweight='bold', color=color,
                               transform=self.ax_stats.transAxes)
            if i < n_cols - 1:
                xv = (i + 1) / n_cols
                self.ax_stats.plot([xv, xv], [0, 1], color='#dddddd',
                                   linewidth=0.8, transform=self.ax_stats.transAxes)

        self.ax_stats.set_facecolor('#f8f8f8')

        self.plt.pause(0.001)

    def close(self):
        self.plt.ioff()
        self.plt.close(self.fig)
