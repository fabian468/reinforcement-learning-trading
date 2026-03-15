# -*- coding: utf-8 -*-
"""
Backtesting: carga el modelo entrenado y lo evalúa sobre datos históricos.

Modos disponibles (variable BACKTEST_MODE):
  'test'   -> último 20% del CSV (mismo split que entrenamiento)
  'full'   -> dataset completo
  'custom' -> rango de fechas personalizado (CUSTOM_START / CUSTOM_END)

Salida:
  - Resumen de métricas en consola
  - Gráfico 4 paneles guardado en resultados_cv/backtesting_resultado.png
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from numba import jit

from parametros import (
    ConfigEntorno, ConfigTrading, ConfigAgente,
    ConfigModelo, get_state_size, get_config_trader,
)
from dueling_dqn_con_per import AI_Trader_per
from state_creator import create_all_states_ohcl, create_all_states_advanced
from indicadores import add_ema200_distance


# ==============================================================================
# CONFIGURACIÓN DEL BACKTESTING  ← modificar aquí
# ==============================================================================

BACKTEST_MODE  = 'test'           # 'test' | 'full' | 'custom'
CUSTOM_START   = '2023-01-01'     # solo para BACKTEST_MODE == 'custom'
CUSTOM_END     = '2024-05-31'     # solo para BACKTEST_MODE == 'custom'

GUARDAR_GRAFICO = True
MOSTRAR_TRADES  = False           # True = imprime cada trade en consola


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def price_format(n):
    n = float(n)
    return "- {0:.3f}".format(abs(n)) if n < 0 else "{0:.3f}".format(abs(n))


def calculate_sharpe_ratio(returns):
    arr = np.array(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    std = np.std(arr)
    return float(np.mean(arr) / std) if std != 0 else 0.0


def calculate_sortino_ratio(returns):
    arr = np.array(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    neg = arr[arr < 0]
    downside_std = np.std(neg) if len(neg) > 1 else 1e-9
    return float(np.mean(arr) / downside_std) if downside_std != 0 else 0.0


@jit(nopython=True)
def calc_profit_long(buy_price, sell_price, pip_value, commission, lot_size):
    profit_pips   = sell_price - buy_price
    profit_dollars = (profit_pips * pip_value) - (commission * lot_size)
    return profit_pips, profit_dollars


@jit(nopython=True)
def calc_profit_short(sell_price, buy_price, pip_value, commission, lot_size):
    profit_pips    = sell_price - buy_price
    profit_dollars = (profit_pips * pip_value) - (commission * lot_size)
    return profit_pips, profit_dollars


def get_full_state(base_state, inventory, inventory_sell, current_price, pip_value, initial_balance):
    has_long  = 1.0 if inventory      else 0.0
    has_short = 1.0 if inventory_sell else 0.0

    if inventory:
        raw_upnl = (current_price - inventory[0]) * pip_value
    elif inventory_sell:
        raw_upnl = (inventory_sell[0] - current_price) * pip_value
    else:
        raw_upnl = 0.0

    upnl_norm = float(np.tanh(raw_upnl / (initial_balance * 0.02)))
    return np.concatenate([base_state, [has_long, has_short, upnl_norm]]).astype(np.float32)


def dataset_loader_csv(csv_path):
    df = pd.read_csv(f"data/{csv_path}", sep='\t')
    df.columns = [col.strip('<>').lower() for col in df.columns]

    ema_200_diferencia, _ = add_ema200_distance(df)
    if len(ema_200_diferencia) > 0:
        df['ema_diference_close'] = ema_200_diferencia

    if 'time' in df.columns and df['time'].notnull().all():
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str))
    else:
        df['datetime'] = pd.to_datetime(df['date'])

    df.set_index('datetime', inplace=True)

    if 'tickvol' in df.columns:
        df.rename(columns={'tickvol': 'tick_volume'}, inplace=True)

    return df


# ==============================================================================
# GRAFICACIÓN
# ==============================================================================

def plot_results(data, trades, equity_curve, equity_timestamps,
                 balance_first, symbol, save_path):
    """
    4 paneles:
      1. Precio + señales de compra/venta
      2. Curva de equity
      3. Drawdown
      4. Distribución P&L  |  cuadro de estadísticas
    """
    closed_pips = [pnl for _, _, typ, pnl in trades
                   if pnl is not None and typ in ('long_close', 'short_close')]

    equity = np.array(equity_curve, dtype=float)
    peak   = np.maximum.accumulate(equity)
    dd_pct = (peak - equity) / np.where(peak != 0, peak, 1.0) * 100.0

    fig = plt.figure(figsize=(18, 15))
    gs  = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[3, 1.5, 1.5, 2],
                            hspace=0.45, wspace=0.3)

    # ── Panel 1: Precio ───────────────────────────────────────────────────────
    ax_price = fig.add_subplot(gs[0, :])
    ax_price.plot(data.index, data['close'].values,
                  color='#333333', linewidth=0.7, label='Close')

    scatter_cfg = {
        'long_open':   ('^', '#2ecc71', 'Compra (Long)'),
        'long_close':  ('v', '#e74c3c', 'Cierre Long'),
        'short_open':  ('v', '#e67e22', 'Short (Venta)'),
        'short_close': ('^', '#9b59b6', 'Cierre Short'),
    }
    by_type = {k: [] for k in scatter_cfg}
    for ts, price, typ, _ in trades:
        if typ in by_type:
            by_type[typ].append((ts, price))

    for typ, points in by_type.items():
        if points:
            marker, color, label = scatter_cfg[typ]
            ts_arr = [p[0] for p in points]
            px_arr = [p[1] for p in points]
            ax_price.scatter(ts_arr, px_arr, marker=marker, color=color,
                             s=55, zorder=5, label=label, alpha=0.85)

    ax_price.set_title(f'Backtesting — {symbol}   ({len(data)} velas)',
                       fontsize=13, fontweight='bold')
    ax_price.set_ylabel('Precio')
    ax_price.legend(loc='upper left', fontsize=8, ncol=2)
    ax_price.grid(True, alpha=0.3)

    # ── Panel 2: Equity ───────────────────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[1, :])
    eq_ts = list(equity_timestamps)
    ax_eq.plot(eq_ts, equity, color='#2980b9', linewidth=1.1, label='Equity')
    ax_eq.axhline(balance_first, color='gray', linestyle='--',
                  linewidth=0.8, label=f'Balance inicial ${balance_first}')
    ax_eq.fill_between(eq_ts, balance_first, equity,
                        where=equity >= balance_first, alpha=0.2, color='#2ecc71')
    ax_eq.fill_between(eq_ts, balance_first, equity,
                        where=equity < balance_first,  alpha=0.2, color='#e74c3c')
    ax_eq.set_ylabel('Equity ($)')
    ax_eq.legend(fontsize=8)
    ax_eq.grid(True, alpha=0.3)

    # ── Panel 3: Drawdown ─────────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[2, :])
    ax_dd.fill_between(eq_ts, 0, -dd_pct, color='#e74c3c', alpha=0.65)
    ax_dd.set_ylabel('Drawdown (%)')
    ax_dd.grid(True, alpha=0.3)

    # ── Panel 4a: Distribución P&L ────────────────────────────────────────────
    ax_pnl = fig.add_subplot(gs[3, 0])
    if closed_pips:
        wins_pips = [p for p in closed_pips if p > 0]
        loss_pips = [p for p in closed_pips if p <= 0]
        if wins_pips:
            ax_pnl.hist(wins_pips, bins=30, color='#2ecc71', alpha=0.75,
                        label=f'Wins ({len(wins_pips)})')
        if loss_pips:
            ax_pnl.hist(loss_pips, bins=30, color='#e74c3c', alpha=0.75,
                        label=f'Losses ({len(loss_pips)})')
        ax_pnl.axvline(0, color='black', linewidth=0.8)
        ax_pnl.set_xlabel('P&L (pips)')
        ax_pnl.set_ylabel('Frecuencia')
        ax_pnl.set_title('Distribución de P&L por trade')
        ax_pnl.legend(fontsize=8)
        ax_pnl.grid(True, alpha=0.3)

    # ── Panel 4b: Estadísticas ────────────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[3, 1])
    ax_stats.axis('off')

    total_trades  = len(closed_pips)
    wins_count    = sum(1 for p in closed_pips if p > 0)
    loss_count    = total_trades - wins_count
    accuracy      = wins_count / total_trades if total_trades > 0 else 0
    total_pnl_pip = sum(closed_pips)
    final_equity  = float(equity[-1])
    total_pnl_usd = final_equity - balance_first
    sharpe        = calculate_sharpe_ratio(closed_pips)
    sortino       = calculate_sortino_ratio(closed_pips)
    max_dd        = float(np.max(dd_pct)) if len(dd_pct) > 0 else 0
    avg_win       = np.mean([p for p in closed_pips if p > 0]) if wins_count > 0 else 0
    avg_loss      = np.mean([p for p in closed_pips if p <= 0]) if loss_count > 0 else 0
    gross_w       = sum(p for p in closed_pips if p > 0)
    gross_l       = abs(sum(p for p in closed_pips if p < 0))
    pf            = (gross_w / gross_l) if gross_l > 0 else float('inf')

    stats_text = (
        f"{'RESUMEN BACKTESTING':^32}\n"
        f"{'─'*32}\n"
        f"Trades totales:     {total_trades}\n"
        f"Wins / Losses:      {wins_count} / {loss_count}\n"
        f"Accuracy:           {accuracy:.1%}\n"
        f"{'─'*32}\n"
        f"P&L total (pips):   {total_pnl_pip:.2f}\n"
        f"P&L total (USD):    {price_format(total_pnl_usd)}\n"
        f"Balance inicial:    ${balance_first:.2f}\n"
        f"Balance final:      ${final_equity:.2f}\n"
        f"{'─'*32}\n"
        f"Sharpe ratio:       {sharpe:.3f}\n"
        f"Sortino ratio:      {sortino:.3f}\n"
        f"Max Drawdown:       {max_dd:.2f}%\n"
        f"Profit Factor:      {pf:.3f}\n"
        f"{'─'*32}\n"
        f"Avg Win  (pips):    {avg_win:.2f}\n"
        f"Avg Loss (pips):    {avg_loss:.2f}\n"
    )

    ax_stats.text(0.05, 0.97, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='#f4f4f4', alpha=0.9))

    if save_path:
        out_path = os.path.join(save_path, 'backtesting_resultado.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nGráfico guardado en: {out_path}")

    plt.show()


# ==============================================================================
# BUCLE PRINCIPAL
# ==============================================================================

def run_backtest():
    # ── Parámetros ─────────────────────────────────────────────────────────────
    nombre_csv      = ConfigEntorno.NOMBRE_CSV
    symbol          = ConfigEntorno.SYMBOL
    window_size     = ConfigEntorno.WINDOW_SIZE
    balance_first   = ConfigTrading.BALANCE_INICIAL
    lot_size        = ConfigTrading.LOT_SIZE
    commission      = ConfigTrading.COMMISSION_PER_TRADE
    pip_value       = 10.0 * lot_size          # pip value en USD

    modelo_path     = ConfigModelo.MODELO_EXISTENTE
    resultados_dir  = ConfigModelo.DIRECTORIO_RESULTADOS
    os.makedirs(resultados_dir, exist_ok=True)

    state_size = get_state_size()

    # ── Cargar datos ───────────────────────────────────────────────────────────
    print(f"Cargando datos: {nombre_csv}")
    df = dataset_loader_csv(nombre_csv)
    if df is None:
        print("Error al cargar datos.")
        return
    print(f"Dataset cargado: {len(df)} velas  ({df.index[0]} → {df.index[-1]})")

    # ── Split train / test (mismo que en entrenamiento) ────────────────────────
    split_idx  = int(len(df) * (1.0 - ConfigEntorno.TEST_SIZE_RATIO))
    train_data = df.iloc[:split_idx].copy()

    if BACKTEST_MODE == 'test':
        backtest_data = df.iloc[split_idx:].copy()
        print(f"Modo: TEST SET  ({len(backtest_data)} velas, "
              f"{backtest_data.index[0]} → {backtest_data.index[-1]})")
    elif BACKTEST_MODE == 'full':
        backtest_data = df.copy()
        print(f"Modo: DATASET COMPLETO  ({len(backtest_data)} velas)")
    elif BACKTEST_MODE == 'custom':
        backtest_data = df.loc[CUSTOM_START:CUSTOM_END].copy()
        print(f"Modo: RANGO PERSONALIZADO  {CUSTOM_START} → {CUSTOM_END}  "
              f"({len(backtest_data)} velas)")
    else:
        raise ValueError(f"BACKTEST_MODE desconocido: '{BACKTEST_MODE}'")

    if len(backtest_data) < window_size + 10:
        print("Error: datos insuficientes para backtesting.")
        return

    # ── Scaler ajustado sobre train (sin data leakage) ─────────────────────────
    print("Ajustando scaler en datos de entrenamiento...")
    scaler = StandardScaler()
    scaler.fit(train_data[['open', 'high', 'low', 'close', 'tick_volume']].values)

    # ── Extraer hora ───────────────────────────────────────────────────────────
    has_time = 'time' in backtest_data.columns and backtest_data['time'].notnull().all()
    if has_time:
        hora_int = np.array(
            [int(h.split(":")[0]) for h in backtest_data['time'].values],
            dtype=np.int32
        )
    else:
        hora_int = np.zeros(len(backtest_data), dtype=np.int32)

    # ── Crear estados ──────────────────────────────────────────────────────────
    print(f"Generando estados ({ConfigEntorno.TIPO_ESTADO})...")
    if ConfigEntorno.TIPO_ESTADO == 'advanced':
        states = create_all_states_advanced(backtest_data, window_size, scaler, hora_int)
    else:
        states = create_all_states_ohcl(backtest_data, window_size, scaler, hora_int)
    print(f"Estados generados: {states.shape}")

    # ── Cargar modelo ──────────────────────────────────────────────────────────
    print(f"\nCargando modelo: {modelo_path}.pth")
    trader = AI_Trader_per(
        state_size=state_size,
        action_space=ConfigAgente.ACTION_SPACE,
        **get_config_trader()
    )
    trader.load_model(modelo_path)
    trader.model.eval()
    trader.epsilon = 0.0    # greedy puro: siempre toma la acción de mayor Q-value
    print("Modelo en modo evaluación (epsilon=0, dropout desactivado).\n")

    # ── Variables de simulación ────────────────────────────────────────────────
    inventory       = []    # precio de entrada longs abiertos
    inventory_sell  = []    # precio de entrada shorts abiertos
    current_equity  = float(balance_first)
    peak_equity     = float(balance_first)

    equity_curve      = [balance_first]
    equity_timestamps = [backtest_data.index[0]]

    # trades: lista de (timestamp, price, tipo, pnl|None)
    # tipo: 'long_open' | 'long_close' | 'short_open' | 'short_close'
    trades     = []
    closed_pips = []
    wins = losses = 0
    total_pips  = 0.0

    close_prices = backtest_data['close'].values.astype(np.float64)
    low_prices   = (backtest_data['low'].values.astype(np.float64)
                    if 'low' in backtest_data.columns else close_prices)
    high_prices  = (backtest_data['high'].values.astype(np.float64)
                    if 'high' in backtest_data.columns else close_prices)
    spreads      = (backtest_data['spread'].values.astype(np.float64)
                    if 'spread' in backtest_data.columns
                    else np.zeros(len(backtest_data)))
    timestamps   = backtest_data.index.values

    N = len(backtest_data)
    action_labels = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'SHORT', 4: 'COVER'}
    action_counts = {i: 0 for i in range(5)}

    print(f"Iniciando backtesting ({N} pasos)...\n")

    for t in range(N):
        price    = close_prices[t]
        spread   = spreads[t]
        buy_exec = price + spread * 0.5
        sell_exec = price - spread * 0.5
        ts       = timestamps[t]

        state_full = get_full_state(
            states[t], inventory, inventory_sell,
            price, pip_value, balance_first
        )
        action = trader.trade(state_full)
        action_counts[action] += 1

        profit_pips  = 0.0
        profit_usd   = 0.0
        trade_closed = False

        # ── Acción 1: COMPRAR (abrir long) ────────────────────────────────────
        if action == 1 and not inventory:
            inventory.append(buy_exec)
            trades.append((ts, buy_exec, 'long_open', None))
            if MOSTRAR_TRADES:
                print(f"[{t:6d}] COMPRA   @ {buy_exec:.5f}")

        # ── Acción 2: CERRAR LONG ─────────────────────────────────────────────
        elif action == 2 and inventory:
            entry = inventory.pop(0)
            profit_pips, profit_usd = calc_profit_long(
                entry, sell_exec, pip_value, commission, lot_size)
            trades.append((ts, sell_exec, 'long_close', profit_pips))
            trade_closed = True
            if MOSTRAR_TRADES:
                print(f"[{t:6d}] CIERRE L @ {sell_exec:.5f} | "
                      f"entry {entry:.5f} | pips {profit_pips:+.2f} | "
                      f"USD {profit_usd:+.2f}")

        # ── Acción 3: ABRIR SHORT ─────────────────────────────────────────────
        elif action == 3 and not inventory_sell:
            inventory_sell.append(sell_exec)
            trades.append((ts, sell_exec, 'short_open', None))
            if MOSTRAR_TRADES:
                print(f"[{t:6d}] SHORT    @ {sell_exec:.5f}")

        # ── Acción 4: CERRAR SHORT ────────────────────────────────────────────
        elif action == 4 and inventory_sell:
            entry = inventory_sell.pop(0)
            profit_pips, profit_usd = calc_profit_short(
                entry, buy_exec, pip_value, commission, lot_size)
            trades.append((ts, buy_exec, 'short_close', profit_pips))
            trade_closed = True
            if MOSTRAR_TRADES:
                print(f"[{t:6d}] CUBRIR S @ {buy_exec:.5f} | "
                      f"entry {entry:.5f} | pips {profit_pips:+.2f} | "
                      f"USD {profit_usd:+.2f}")

        # ── Cierre forzado a las 23:00 ────────────────────────────────────────
        elif has_time and hora_int[t] == 23:
            if inventory:
                entry = inventory.pop(0)
                profit_pips, profit_usd = calc_profit_long(
                    entry, sell_exec, pip_value, commission, lot_size)
                trades.append((ts, sell_exec, 'long_close', profit_pips))
                trade_closed = True
                if MOSTRAR_TRADES:
                    print(f"[{t:6d}] FORZADO LONG  @ {sell_exec:.5f} | "
                          f"pips {profit_pips:+.2f}")
            elif inventory_sell:
                entry = inventory_sell.pop(0)
                profit_pips, profit_usd = calc_profit_short(
                    entry, buy_exec, pip_value, commission, lot_size)
                trades.append((ts, buy_exec, 'short_close', profit_pips))
                trade_closed = True
                if MOSTRAR_TRADES:
                    print(f"[{t:6d}] FORZADO SHORT @ {buy_exec:.5f} | "
                          f"pips {profit_pips:+.2f}")

        # ── Actualizar equity ─────────────────────────────────────────────────
        if trade_closed:
            total_pips     += profit_pips
            current_equity += profit_usd
            closed_pips.append(profit_pips)
            if profit_pips > 0:
                wins += 1
            else:
                losses += 1
            if current_equity > peak_equity:
                peak_equity = current_equity

        equity_curve.append(current_equity)
        equity_timestamps.append(ts)

    # ── Cerrar posiciones abiertas al final del período ───────────────────────
    last_price = close_prices[-1]
    last_ts    = timestamps[-1]

    for entry in inventory:
        pp, pu = calc_profit_long(entry, last_price, pip_value, commission, lot_size)
        trades.append((last_ts, last_price, 'long_close', pp))
        total_pips += pp;  current_equity += pu
        closed_pips.append(pp)
        wins += 1 if pp > 0 else 0
        losses += 0 if pp > 0 else 1
        print(f"Posición long abierta cerrada al fin del período | pips {pp:+.2f}")

    for entry in inventory_sell:
        pp, pu = calc_profit_short(entry, last_price, pip_value, commission, lot_size)
        trades.append((last_ts, last_price, 'short_close', pp))
        total_pips += pp;  current_equity += pu
        closed_pips.append(pp)
        wins += 1 if pp > 0 else 0
        losses += 0 if pp > 0 else 1
        print(f"Posición short abierta cerrada al fin del período | pips {pp:+.2f}")

    # ── Métricas finales ──────────────────────────────────────────────────────
    total_trades  = len(closed_pips)
    accuracy      = wins / total_trades if total_trades > 0 else 0
    sharpe        = calculate_sharpe_ratio(closed_pips)
    sortino       = calculate_sortino_ratio(closed_pips)
    avg_win       = (np.mean([p for p in closed_pips if p > 0])  if wins   > 0 else 0)
    avg_loss      = (np.mean([p for p in closed_pips if p <= 0]) if losses > 0 else 0)
    gross_w       = sum(p for p in closed_pips if p > 0)
    gross_l       = abs(sum(p for p in closed_pips if p < 0))
    profit_factor = (gross_w / gross_l) if gross_l > 0 else float('inf')

    eq_arr  = np.array(equity_curve, dtype=float)
    peak_a  = np.maximum.accumulate(eq_arr)
    dd_arr  = (peak_a - eq_arr) / np.where(peak_a != 0, peak_a, 1.0) * 100.0
    max_dd  = float(np.max(dd_arr))

    total_pnl_usd = current_equity - balance_first

    # ── Resumen en consola ────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("             RESULTADOS DEL BACKTESTING")
    print("=" * 58)
    print(f"  Símbolo:            {symbol}")
    print(f"  Período:            {backtest_data.index[0]}  →  {backtest_data.index[-1]}")
    print(f"  Velas:              {N}")
    print(f"  Modelo:             {modelo_path}")
    print("-" * 58)
    print(f"  Trades totales:     {total_trades}")
    print(f"  Wins / Losses:      {wins} / {losses}")
    print(f"  Accuracy:           {accuracy:.1%}")
    print("-" * 58)
    print(f"  P&L total (pips):   {total_pips:.2f}")
    print(f"  P&L total (USD):    {price_format(total_pnl_usd)}")
    print(f"  Balance inicial:    ${balance_first:.2f}")
    print(f"  Balance final:      ${current_equity:.2f}")
    print("-" * 58)
    print(f"  Sharpe ratio:       {sharpe:.3f}")
    print(f"  Sortino ratio:      {sortino:.3f}")
    print(f"  Max Drawdown:       {max_dd:.2f}%")
    print(f"  Profit Factor:      {profit_factor:.3f}")
    print("-" * 58)
    print(f"  Avg Win  (pips):    {avg_win:.2f}")
    print(f"  Avg Loss (pips):    {avg_loss:.2f}")
    print("-" * 58)
    print("  Distribución de acciones:")
    for a, cnt in action_counts.items():
        pct = cnt / N * 100
        print(f"    {action_labels[a]:6s}: {cnt:6d}  ({pct:.1f}%)")
    print("=" * 58)

    # ── Gráfico ───────────────────────────────────────────────────────────────
    if GUARDAR_GRAFICO:
        plot_results(
            data=backtest_data,
            trades=trades,
            equity_curve=equity_curve,
            equity_timestamps=equity_timestamps,
            balance_first=balance_first,
            symbol=symbol,
            save_path=resultados_dir,
        )


if __name__ == "__main__":
    run_backtest()
