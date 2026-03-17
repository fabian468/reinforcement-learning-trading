# -*- coding: utf-8 -*-
"""
live_trading.py — Trading en vivo con MetaTrader 5 usando el modelo Dueling DQN.

Uso:
    python live_trading.py

Requisitos adicionales:
    pip install MetaTrader5

El script espera el cierre de cada vela, construye el estado, consulta el modelo
y envía la orden a MT5. Completamente independiente de parametros.py.

Balance, profit y lot size vienen directamente de MT5 en tiempo real.
"""

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
import MetaTrader5 as mt5

from model_pytorch import DuelingDQN
from indicadores import rsi, macd


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — solo lo que el modelo necesita saber y MT5 no puede darte
# ══════════════════════════════════════════════════════════════════════════════
SYMBOL           = "GOLD#"
TIMEFRAME        = mt5.TIMEFRAME_M15      # mt5.TIMEFRAME_M15 / H1 / H4 / D1
MODELO_PATH      = "resultados_cv/model_XAUUSD_M15_2025_03_01_2025_03_31.csv"  # sin extensión
MAGIC            = 20250101               # número mágico para identificar órdenes del bot

LOT_SIZE         = 0.01                  # tamaño de lote a operar
WINDOW_SIZE      = 18                    # debe coincidir con el entrenamiento
TIPO_ESTADO      = 'advanced'            # 'ohlc' o 'advanced' — debe coincidir con entrenamiento
BARS_SCALER      = 500                   # barras históricas para ajustar el scaler al inicio
FORCE_CLOSE_HOUR = 23                    # hora UTC a la que cerrar posiciones forzosamente
ACTION_SPACE     = 5
# ══════════════════════════════════════════════════════════════════════════════

ACTION_NAMES = {0: 'HOLD', 1: 'BUY', 2: 'CLOSE LONG', 3: 'SHORT', 4: 'COVER'}

_TF_SECONDS = {
    mt5.TIMEFRAME_M1:  60,
    mt5.TIMEFRAME_M5:  300,
    mt5.TIMEFRAME_M15: 900,
    mt5.TIMEFRAME_M30: 1800,
    mt5.TIMEFRAME_H1:  3600,
    mt5.TIMEFRAME_H4:  14400,
    mt5.TIMEFRAME_D1:  86400,
}


# ──────────────────────────────────────────────────────────────────────────────
# Tamaño del estado
# ──────────────────────────────────────────────────────────────────────────────

def _get_state_size():
    if TIPO_ESTADO == 'ohlc':
        return WINDOW_SIZE * 5 + 2 + 3     # OHLCV*W + hora_sin/cos + pos(3)
    elif TIPO_ESTADO == 'advanced':
        return WINDOW_SIZE * 5 + 8 + 3     # OHLCV*W + RSI+MACD(3)+día(2)+hora(2) + pos(3)
    raise ValueError(f"TIPO_ESTADO desconocido: {TIPO_ESTADO}")


# ──────────────────────────────────────────────────────────────────────────────
# Construcción del estado (idéntico a create_all_states_* en state_creator.py)
# ──────────────────────────────────────────────────────────────────────────────

def _states_ohlc(data, scaler, hora_int):
    from numpy.lib.stride_tricks import sliding_window_view
    N = len(data)
    features_scaled = scaler.transform(
        data[['open', 'high', 'low', 'close', 'tick_volume']].values
    ).astype(np.float32)
    padding = np.tile(features_scaled[0:1], (WINDOW_SIZE - 1, 1))
    windows = sliding_window_view(
        np.vstack([padding, features_scaled]), (WINDOW_SIZE, 5)
    ).reshape(N, WINDOW_SIZE * 5)
    h = hora_int.astype(np.float32)
    return np.hstack([
        windows,
        np.sin(2 * np.pi * h / 24).reshape(-1, 1),
        np.cos(2 * np.pi * h / 24).reshape(-1, 1),
    ]).astype(np.float32)


def _states_advanced(data, scaler, hora_int):
    from numpy.lib.stride_tricks import sliding_window_view
    N = len(data)

    # OHLCV
    ohlc_scaled = scaler.transform(
        data[['open', 'high', 'low', 'close', 'tick_volume']].values
    ).astype(np.float32)
    padding_ohlc = np.tile(ohlc_scaled[0:1], (WINDOW_SIZE - 1, 1))
    windows_ohlc = sliding_window_view(
        np.vstack([padding_ohlc, ohlc_scaled]), (WINDOW_SIZE, 5)
    ).reshape(N, WINDOW_SIZE * 5)

    # RSI
    try:
        rsi_vals = rsi(data, period=14).fillna(50).values
    except Exception:
        rsi_vals = np.full(N, 50.0)
    rsi_norm = (rsi_vals / 100.0).astype(np.float32).reshape(-1, 1)

    # MACD
    try:
        ml, sl = macd(data, fast_period=12, slow_period=26, signal_period=9)
        ml = ml.fillna(0).values
        sl = sl.fillna(0).values
    except Exception:
        ml = sl = np.zeros(N)
    hist = ml - sl
    macd_norm   = (ml   / 10.0).astype(np.float32).reshape(-1, 1)
    signal_norm = (sl   / 10.0).astype(np.float32).reshape(-1, 1)
    hist_norm   = (hist / 10.0).astype(np.float32).reshape(-1, 1)

    # Día de la semana
    dow = data.index.dayofweek.values if hasattr(data.index, 'dayofweek') else np.zeros(N)
    day_sin = np.sin(2 * np.pi * dow / 7).astype(np.float32).reshape(-1, 1)
    day_cos = np.cos(2 * np.pi * dow / 7).astype(np.float32).reshape(-1, 1)

    # Hora
    h = hora_int.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * h / 24).reshape(-1, 1)
    hour_cos = np.cos(2 * np.pi * h / 24).reshape(-1, 1)

    return np.hstack([
        windows_ohlc,
        rsi_norm, macd_norm, signal_norm, hist_norm,
        day_sin, day_cos, hour_sin, hour_cos,
    ]).astype(np.float32)


def build_current_state(df, scaler, has_long, has_short, position_profit, account_balance):
    """
    Construye el estado completo para el último timestep de df.
    position_profit y account_balance vienen directamente de MT5.
    """
    hora_int = np.array([ts.hour for ts in df.index], dtype=np.int32)

    if TIPO_ESTADO == 'advanced':
        all_states = _states_advanced(df, scaler, hora_int)
    else:
        all_states = _states_ohlc(df, scaler, hora_int)

    base_state = all_states[-1]

    # Unrealized PnL normalizado con el balance real de MT5
    upnl_norm = float(np.tanh(position_profit / (account_balance * 0.02))) if account_balance > 0 else 0.0

    return np.concatenate([
        base_state,
        [float(has_long), float(has_short), upnl_norm]
    ]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades MT5
# ──────────────────────────────────────────────────────────────────────────────

def get_bars(n_bars):
    """Descarga n_bars barras cerradas de MT5 y devuelve un DataFrame."""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n_bars + 1)
    if rates is None or len(rates) < 2:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']].iloc[:-1].copy()


def get_position_state():
    """
    Consulta MT5 y devuelve (has_long, has_short, position_profit).
    El profit viene en USD directamente de la plataforma.
    """
    positions = mt5.positions_get(symbol=SYMBOL)
    has_long = has_short = False
    profit = 0.0
    if positions:
        for p in positions:
            if p.magic != MAGIC:
                continue
            profit += p.profit
            if p.type == mt5.POSITION_TYPE_BUY:
                has_long = True
            elif p.type == mt5.POSITION_TYPE_SELL:
                has_short = True
    return has_long, has_short, profit


def _send(request):
    result = mt5.order_send(request)
    if result is None:
        print(f"  [ERROR] order_send devolvió None — {mt5.last_error()}")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  [ERROR] retcode={result.retcode}  comment={result.comment}")
        return False
    return True


def execute_action(action):
    """Ejecuta la orden en MT5 según la acción del modelo."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("  [ERROR] No se pudo obtener tick.")
        return False

    ask, bid = tick.ask, tick.bid
    base_req = {
        "symbol":       SYMBOL,
        "magic":        MAGIC,
        "deviation":    20,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "action":       mt5.TRADE_ACTION_DEAL,
    }

    if action == 1:                          # Abrir long
        ok = _send({**base_req, "volume": LOT_SIZE, "type": mt5.ORDER_TYPE_BUY,
                    "price": ask, "comment": "DQN_BUY"})
        if ok:
            print(f"  [BUY]   Long abierto  @ {ask:.5f}")
        return ok

    elif action == 2:                        # Cerrar long
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return False
        for p in positions:
            if p.magic == MAGIC and p.type == mt5.POSITION_TYPE_BUY:
                ok = _send({**base_req, "volume": p.volume, "type": mt5.ORDER_TYPE_SELL,
                             "position": p.ticket, "price": bid, "comment": "DQN_CLOSE_LONG"})
                if ok:
                    print(f"  [CLOSE] Long cerrado @ {bid:.5f}  ticket={p.ticket}")
                return ok

    elif action == 3:                        # Abrir short
        ok = _send({**base_req, "volume": LOT_SIZE, "type": mt5.ORDER_TYPE_SELL,
                    "price": bid, "comment": "DQN_SHORT"})
        if ok:
            print(f"  [SHORT]  Short abierto @ {bid:.5f}")
        return ok

    elif action == 4:                        # Cerrar short
        positions = mt5.positions_get(symbol=SYMBOL)
        if not positions:
            return False
        for p in positions:
            if p.magic == MAGIC and p.type == mt5.POSITION_TYPE_SELL:
                ok = _send({**base_req, "volume": p.volume, "type": mt5.ORDER_TYPE_BUY,
                             "position": p.ticket, "price": ask, "comment": "DQN_COVER"})
                if ok:
                    print(f"  [COVER]  Short cerrado @ {ask:.5f}  ticket={p.ticket}")
                return ok

    return False


def force_close_all():
    """Cierra todas las posiciones abiertas del bot."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return
    tick = mt5.symbol_info_tick(SYMBOL)
    for p in positions:
        if p.magic != MAGIC:
            continue
        if p.type == mt5.POSITION_TYPE_BUY:
            close_type, close_price = mt5.ORDER_TYPE_SELL, tick.bid
        else:
            close_type, close_price = mt5.ORDER_TYPE_BUY, tick.ask
        ok = _send({
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       SYMBOL,
            "volume":       p.volume,
            "type":         close_type,
            "position":     p.ticket,
            "price":        close_price,
            "deviation":    20,
            "magic":        MAGIC,
            "comment":      "DQN_FORCE_CLOSE",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })
        if ok:
            print(f"  [FORCE CLOSE] ticket={p.ticket} cerrado.")


def seconds_to_next_bar():
    tf_sec = _TF_SECONDS.get(TIMEFRAME, 900)
    now_sec = int(datetime.utcnow().timestamp())
    return tf_sec - (now_sec % tf_sec)


# ──────────────────────────────────────────────────────────────────────────────
# Cargar modelo
# ──────────────────────────────────────────────────────────────────────────────

def load_model():
    """Carga solo la red neuronal. Sin agente, sin memoria, sin optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DuelingDQN(state_size=_get_state_size(), action_space=ACTION_SPACE).to(device)
    checkpoint = torch.load(f"{MODELO_PATH}.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Modelo cargado en {device}")
    return model, device


def get_action(model, device, state):
    """Inferencia pura: pasa el estado por la red y devuelve acción + Q-values."""
    with torch.no_grad():
        tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = model(tensor)[0].cpu().numpy()
        return int(np.argmax(q_values)), q_values


def print_state_visual(df, state, q_values, has_long, has_short, upnl_norm, action, valid):
    """Muestra visualmente el estado que recibió el modelo y su decisión."""
    from indicadores import rsi as calc_rsi, macd as calc_macd

    # ── Indicadores sobre los últimos datos ──
    try:
        rsi_val  = float(calc_rsi(df, period=14).iloc[-1])
    except Exception:
        rsi_val  = 50.0
    try:
        ml, sl   = calc_macd(df, fast_period=12, slow_period=26, signal_period=9)
        macd_val = float(ml.iloc[-1])
        hist_val = float((ml - sl).iloc[-1])
    except Exception:
        macd_val = hist_val = 0.0

    close     = float(df['close'].iloc[-1])
    hour      = df.index[-1].hour
    dow_names = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    dow       = dow_names[df.index[-1].dayofweek]

    bar  = lambda v, w=20: '█' * int(abs(v) * w) + '░' * (w - int(abs(v) * w))

    print("  ┌─────────────────────────────────────────────┐")
    print(f"  │  ESTADO DEL MODELO                          │")
    print("  ├─────────────────────────────────────────────┤")
    print(f"  │  Precio   : {close:.5f}   {dow}  {hour:02d}:00          │")
    print(f"  │  RSI(14)  : {rsi_val:6.2f}  {'▲ sobrecompra' if rsi_val > 70 else '▼ sobreventa' if rsi_val < 30 else '  neutral    '}               │")
    print(f"  │  MACD     : {macd_val:+.4f}  Hist: {hist_val:+.4f}          │")
    print("  ├─────────────────────────────────────────────┤")
    print(f"  │  Posición : {'LONG  ✓' if has_long else 'SHORT ✓' if has_short else 'sin posición'}                        │")
    print(f"  │  uPnL norm: {upnl_norm:+.4f}  {'perdiendo' if upnl_norm < 0 else 'ganando  '}                  │")
    print("  ├─────────────────────────────────────────────┤")
    print("  │  Q-VALUES (confianza por acción)            │")

    q_min, q_max = q_values.min(), q_values.max()
    q_range = q_max - q_min if q_max != q_min else 1.0
    for i, (name, q) in enumerate(zip(
        ['HOLD        ', 'BUY         ', 'CLOSE LONG  ', 'SHORT       ', 'COVER SHORT '],
        q_values
    )):
        norm     = (q - q_min) / q_range
        chosen   = ' ◄ ELEGIDA' if i == action else ''
        blocked  = ' [BLOQUEADA]' if i == action and not valid else ''
        print(f"  │  {i} {name} {bar(norm, 16)} {q:+.3f}{chosen}{blocked}  │")

    print("  └─────────────────────────────────────────────┘")


# ──────────────────────────────────────────────────────────────────────────────
# Loop principal
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  LIVE TRADING — {SYMBOL}  ({TIPO_ESTADO}, W={WINDOW_SIZE})")
    print(f"  Modelo  : {MODELO_PATH}")
    print(f"  Lot     : {LOT_SIZE}  |  Magic: {MAGIC}")
    print("=" * 60)

    # ── Conectar a MT5 ──
    if not mt5.initialize():
        print(f"[ERROR] No se pudo inicializar MT5: {mt5.last_error()}")
        return
    print(f"MT5 conectado — build {mt5.version()[0]}")
    acc = mt5.account_info()
    if acc:
        print("=" * 60)
        print(f"  Cuenta    : {acc.login}  ({acc.name})")
        print(f"  Broker    : {acc.company}")
        print(f"  Servidor  : {acc.server}")
        print(f"  Tipo      : {'Demo' if acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else 'Real'}")
        print(f"  Balance   : {acc.balance:.2f} {acc.currency}")
        print(f"  Equity    : {acc.equity:.2f} {acc.currency}")
        print(f"  Margen lib: {acc.margin_free:.2f} {acc.currency}")
        print(f"  Apalanca  : 1:{acc.leverage}")
        print("=" * 60)

    if not mt5.symbol_info(SYMBOL) or not mt5.symbol_info(SYMBOL).visible:
        if not mt5.symbol_select(SYMBOL, True):
            print(f"[ERROR] Símbolo {SYMBOL} no disponible.")
            mt5.shutdown()
            return

    # ── Cargar modelo ──
    try:
        model, device = load_model()
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        mt5.shutdown()
        return

    # ── Ajustar scaler sobre historial largo (una sola vez al inicio) ──
    print(f"Descargando {BARS_SCALER} barras para ajustar el scaler...")
    df_init = get_bars(BARS_SCALER)
    if df_init is None or len(df_init) < WINDOW_SIZE + 1:
        print("[ERROR] No hay suficientes barras históricas para el scaler.")
        mt5.shutdown()
        return
    scaler = StandardScaler()
    scaler.fit(df_init[['open', 'high', 'low', 'close', 'tick_volume']].values)
    print(f"Scaler ajustado sobre {len(df_init)} barras.\n")

    print("Iniciando loop de trading. Ctrl+C para detener.\n")
    last_bar_time = None

    try:
        while True:
            wait = seconds_to_next_bar()
            if wait > 5:
                time.sleep(wait - 3)
                continue
            time.sleep(4)   # margen post-cierre para que MT5 actualice la vela

            # ── Barras recientes ──
            df = get_bars(WINDOW_SIZE + 50)
            if df is None or len(df) < WINDOW_SIZE + 1:
                print("[WARN] Datos insuficientes. Reintentando...")
                time.sleep(10)
                continue

            current_bar_time = df.index[-1]
            if current_bar_time == last_bar_time:
                time.sleep(5)
                continue
            last_bar_time = current_bar_time

            current_price   = float(df['close'].iloc[-1])
            current_hour    = df.index[-1].hour
            now_str         = df.index[-1].strftime('%Y-%m-%d %H:%M')

            # ── Estado de cuenta y posición desde MT5 ──
            account         = mt5.account_info()
            account_balance = account.balance if account else 1000.0
            account_equity  = account.equity  if account else account_balance
            has_long, has_short, position_profit = get_position_state()

            # ── Cierre forzado a hora 23 ──
            if current_hour == FORCE_CLOSE_HOUR and (has_long or has_short):
                print(f"[{now_str}]  Cierre forzado (hora {FORCE_CLOSE_HOUR} UTC)")
                force_close_all()
                continue

            # ── Construir estado ──
            upnl_norm = float(np.tanh(position_profit / (account_balance * 0.02))) if account_balance > 0 else 0.0
            state = build_current_state(
                df, scaler, has_long, has_short, position_profit, account_balance
            )

            # ── Acción del modelo ──
            action, q_values = get_action(model, device, state)
            action_name = ACTION_NAMES.get(action, '?')

            valid = (
                (action == 1 and not has_long  and not has_short) or
                (action == 2 and has_long) or
                (action == 3 and not has_short and not has_long) or
                (action == 4 and has_short) or
                action == 0
            )

            print(f"\n[{now_str}]  balance=${account_balance:.2f}  equity=${account_equity:.2f}  profit=${position_profit:.2f}")
            print_state_visual(df, state, q_values, has_long, has_short, upnl_norm, action, valid)

            if valid and action != 0:
                execute_action(action)

    except KeyboardInterrupt:
        print("\n[INFO] Detenido por el usuario.")

    finally:
        mt5.shutdown()
        print("MT5 desconectado.")


if __name__ == "__main__":
    main()
