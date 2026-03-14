# -*- coding: utf-8 -*-
"""
Visualización en tiempo real de compras/ventas durante el entrenamiento.
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
        from matplotlib.collections import LineCollection
        from matplotlib.widgets import Button

        self.plt          = plt
        self.update_every = update_every
        self._LC          = LineCollection

        # --- Buffers ---
        self.prices       = deque(maxlen=window_prices)
        self.equity       = deque(maxlen=5000)
        self.buy_markers  = []   # (t, price)
        self.sell_markers = []   # (t, price)
        self.win_segs     = []   # [(t0,p0),(t1,p1)] trades ganados
        self.loss_segs    = []   # [(t0,p0),(t1,p1)] trades perdidos
        self.open_long    = None # (t, price, is_random)
        self.open_short   = None # (t, price, is_random)
        self.t_global     = 0
        self.fold         = 1
        self.episode      = 1
        self.paused       = False

        # Stats
        self.epsilon      = 1.0
        self.reward_ep    = 0.0
        self.trades       = 0
        self.loss_val     = 0.0
        self.random_count = 0
        self.model_count  = 0
        self.model_wins   = 0
        self.model_losses = 0
        self.rand_wins    = 0
        self.rand_losses  = 0

        # --- Figura ---
        plt.ion()
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[4, 1.6, 1.4],
                               hspace=0.45, top=0.93, bottom=0.10)
        self.ax_p = self.fig.add_subplot(gs[0])
        self.ax_e = self.fig.add_subplot(gs[1])
        self.ax_s = self.fig.add_subplot(gs[2])
        self.fig.suptitle('Entrenamiento en vivo', fontsize=13, fontweight='bold')

        # Objetos persistentes — precio
        self.line_price,      = self.ax_p.plot([], [], color='#4a90d9', lw=1, label='Precio')
        self.line_live,       = self.ax_p.plot([], [], lw=1.5, ls=':', alpha=0.9)
        self.lc_wins          = LineCollection([], colors='#2ecc71', lw=1.5, ls='--', alpha=0.75, zorder=3)
        self.lc_losses        = LineCollection([], colors='#e74c3c', lw=1.5, ls='--', alpha=0.75, zorder=3)
        self.ax_p.add_collection(self.lc_wins)
        self.ax_p.add_collection(self.lc_losses)
        self.sc_buy  = self.ax_p.scatter([], [], marker='^', c='#2ecc71', s=100, zorder=5, label='Compra/Cierre short')
        self.sc_sell = self.ax_p.scatter([], [], marker='v', c='#e74c3c', s=100, zorder=5, label='Venta/Apertura short')
        self.ax_p.set_ylabel('Precio')
        self.ax_p.legend(loc='upper left', fontsize=8)
        self.ax_p.grid(True, alpha=0.3)

        # Objetos persistentes — equity
        self.line_eq,  = self.ax_e.plot([], [], color='#2ecc71', lw=1)
        self.hline_eq  = self.ax_e.axhline(y=0, color='gray', ls='--', lw=0.8)
        self.ax_e.set_ylabel('Equity ($)')
        self.ax_e.set_xlabel('Paso')
        self.ax_e.grid(True, alpha=0.3)

        # Panel stats — textos persistentes
        self.ax_s.axis('off')
        self.ax_s.set_facecolor('#f8f8f8')
        self._stat_labels = []
        self._stat_values = []
        self._stat_divs   = []
        self._stats_init  = False   # se crean en el primer draw

        # Botón pausa
        ax_btn   = self.fig.add_axes([0.44, 0.01, 0.13, 0.04])
        self.btn = Button(ax_btn, 'Pausa', color='#f0f0f0', hovercolor='#d0d0d0')
        self.btn.on_clicked(self._toggle_pause)

        plt.show(block=False)
        plt.pause(0.1)

    # ------------------------------------------------------------------
    def _toggle_pause(self, event):
        self.paused = not self.paused
        self.btn.label.set_text('Reanudar' if self.paused else 'Pausa')
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def reset_episode(self, fold, episode):
        self.fold    = fold
        self.episode = episode
        self.prices.clear()
        self.equity.clear()
        self.buy_markers.clear()
        self.sell_markers.clear()
        self.win_segs.clear()
        self.loss_segs.clear()
        self.open_long   = None
        self.open_short  = None
        self.t_global    = 0
        self.model_wins  = 0
        self.model_losses= 0
        self.rand_wins   = 0
        self.rand_losses = 0
        self.trades      = 0
        self.reward_ep   = 0.0

    # ------------------------------------------------------------------
    def update(self, t, price, current_equity, action, inventory, inventory_sell,
               epsilon=1.0, reward_episode=0.0, trades=0, wins=0, losses=0,
               random_count=0, model_count=0, loss=0.0, is_random=False):

        self.t_global     = t
        self.epsilon      = epsilon
        self.reward_ep    = reward_episode
        self.trades       = trades
        self.loss_val     = loss
        self.random_count = random_count
        self.model_count  = model_count

        self.prices.append(price)
        self.equity.append(current_equity)

        inv      = list(inventory)
        inv_sell = list(inventory_sell)

        if action == 1 and len(inv) == 1:
            self.open_long = (t, price, is_random)
            self.buy_markers.append((t, price))

        elif action == 2 and len(inv) == 0:
            self.sell_markers.append((t, price))
            if self.open_long:
                t0, p0, was_rand = self.open_long
                profit = price > p0
                seg = [(t0, p0), (t, price)]
                if profit:
                    self.win_segs.append(seg)
                    if was_rand: self.rand_wins  += 1
                    else:        self.model_wins  += 1
                else:
                    self.loss_segs.append(seg)
                    if was_rand: self.rand_losses += 1
                    else:        self.model_losses+= 1
                self.open_long = None

        elif action == 3 and len(inv_sell) == 1:
            self.open_short = (t, price, is_random)
            self.sell_markers.append((t, price))

        elif action == 4 and len(inv_sell) == 0:
            self.buy_markers.append((t, price))
            if self.open_short:
                t0, p0, was_rand = self.open_short
                profit = price < p0
                seg = [(t0, p0), (t, price)]
                if profit:
                    self.win_segs.append(seg)
                    if was_rand: self.rand_wins  += 1
                    else:        self.model_wins  += 1
                else:
                    self.loss_segs.append(seg)
                    if was_rand: self.rand_losses += 1
                    else:        self.model_losses+= 1
                self.open_short = None

        if not self.paused and t % self.update_every == 0 and t > 0:
            self._draw()

    # ------------------------------------------------------------------
    def _draw(self):
        px = list(self.prices)
        n  = len(px)
        if n == 0:
            return

        start   = self.t_global - n + 1
        x_range = list(range(start, start + n))

        # --- Precio ---
        self.line_price.set_data(x_range, px)

        # Markers visibles
        bx = np.array([t for t, p in self.buy_markers  if t >= start], dtype=float)
        by = np.array([p for t, p in self.buy_markers  if t >= start], dtype=float)
        sx = np.array([t for t, p in self.sell_markers if t >= start], dtype=float)
        sy = np.array([p for t, p in self.sell_markers if t >= start], dtype=float)

        self.sc_buy.set_offsets(np.c_[bx, by] if len(bx) else np.empty((0, 2)))
        self.sc_sell.set_offsets(np.c_[sx, sy] if len(sx) else np.empty((0, 2)))

        # Trades cerrados
        def _filter(segs):
            return [s for s in segs if s[1][0] >= start or s[0][0] >= start]

        self.lc_wins.set_segments(_filter(self.win_segs))
        self.lc_losses.set_segments(_filter(self.loss_segs))

        # Línea en vivo (posición abierta)
        curr = px[-1]
        if self.open_long and self.open_long[0] >= start:
            ok = curr >= self.open_long[1]
            self.line_live.set_data([self.open_long[0], self.t_global],
                                    [self.open_long[1], curr])
            self.line_live.set_color('#2ecc71' if ok else '#e74c3c')
        elif self.open_short and self.open_short[0] >= start:
            ok = curr <= self.open_short[1]
            self.line_live.set_data([self.open_short[0], self.t_global],
                                    [self.open_short[1], curr])
            self.line_live.set_color('#2ecc71' if ok else '#e74c3c')
        else:
            self.line_live.set_data([], [])

        self.ax_p.set_title(
            f'Fold {self.fold} | Episodio {self.episode} | Paso {self.t_global}'
            + ('  ⏸ PAUSADO' if self.paused else ''), fontsize=10)
        self.ax_p.relim()
        self.ax_p.autoscale_view()

        # --- Equity ---
        eq = list(self.equity)
        if eq:
            xs = list(range(len(eq)))
            self.line_eq.set_data(xs, eq)
            self.line_eq.set_color('#2ecc71' if eq[-1] >= eq[0] else '#e74c3c')
            self.hline_eq.set_ydata([eq[0], eq[0]])
            self.ax_e.relim()
            self.ax_e.autoscale_view()

        # --- Stats ---
        m_tot = self.model_wins + self.model_losses
        r_tot = self.rand_wins  + self.rand_losses
        m_wr  = self.model_wins / m_tot * 100 if m_tot > 0 else 0
        r_wr  = self.rand_wins  / r_tot * 100 if r_tot > 0 else 0
        tot   = self.random_count + self.model_count
        pct_r = self.random_count / tot * 100 if tot > 0 else 0
        pct_m = self.model_count  / tot * 100 if tot > 0 else 0
        eq_v  = eq[-1] if eq else 0.0

        stats = [
            ('Epsilon',       f'{self.epsilon:.3f}',                         '#e67e22'),
            ('Reward ep',     f'{self.reward_ep:.2f}',                       '#2ecc71' if self.reward_ep >= 0 else '#e74c3c'),
            ('Loss',          f'{self.loss_val:.5f}',                        '#9b59b6'),
            ('Trades',        f'{self.trades}',                              '#4a90d9'),
            ('Modelo W/L',    f'{self.model_wins} / {self.model_losses}',    '#2ecc71'),
            ('Modelo Win%',   f'{m_wr:.0f}%',                               '#2ecc71' if m_wr >= 50 else '#e74c3c'),
            ('Random W/L',    f'{self.rand_wins} / {self.rand_losses}',      '#e67e22'),
            ('Random Win%',   f'{r_wr:.0f}%',                               '#e67e22'),
            ('Acciones R/M',  f'{pct_r:.0f}% / {pct_m:.0f}%',              '#888888'),
            ('Equity',        f'${eq_v:.2f}',                               '#2ecc71' if (len(eq) < 2 or eq[-1] >= eq[0]) else '#e74c3c'),
        ]

        n_cols = len(stats)

        if not self._stats_init:
            for i, (label, value, color) in enumerate(stats):
                x = (i + 0.5) / n_cols
                lbl = self.ax_s.text(x, 0.75, label, ha='center', va='center',
                                     fontsize=7.5, color='#888888',
                                     transform=self.ax_s.transAxes)
                val = self.ax_s.text(x, 0.28, value, ha='center', va='center',
                                     fontsize=9.5, fontweight='bold', color=color,
                                     transform=self.ax_s.transAxes)
                self._stat_labels.append(lbl)
                self._stat_values.append(val)
                if i < n_cols - 1:
                    xv = (i + 1) / n_cols
                    div = self.ax_s.plot([xv, xv], [0, 1], color='#dddddd',
                                         lw=0.8, transform=self.ax_s.transAxes)
                    self._stat_divs.append(div)
            self._stats_init = True
        else:
            for txt, (_, value, color) in zip(self._stat_values, stats):
                txt.set_text(value)
                txt.set_color(color)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ------------------------------------------------------------------
    def close(self):
        self.plt.ioff()
        self.plt.close(self.fig)
