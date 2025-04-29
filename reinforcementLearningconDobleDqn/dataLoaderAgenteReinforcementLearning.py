# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:29:50 2025

@author: fabia
"""
from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd

def dataset_loader_mt5(symbol, desde, hasta, timeframe):
    """Carga datos históricos desde MetaTrader 5 incluyendo el volumen"""

    # Convertir parámetros de fecha a formato datetime
    desde_dt = datetime.strptime(desde, "%Y-%m-%d")
    hasta_dt = datetime.strptime(hasta, "%Y-%m-%d")

    # Mapeo de intervalos a constantes MT5
    timeframe_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }

    # Verificar que el intervalo es válido
    if timeframe not in timeframe_map:
        print(f"Intervalo {timeframe} no válido. Opciones disponibles: {list(timeframe_map.keys())}")
        return None

    # Verificar que el símbolo existe en MT5
    symbols = mt5.symbols_get()
    symbol_names = [s.name for s in symbols]
    if symbol not in symbol_names:
        print(f"El símbolo {symbol} no está disponible en MetaTrader 5")
        print("Símbolos disponibles:", symbol_names[:10], "...")
        return None

    # Obtener datos históricos
    rates = mt5.copy_rates_range(symbol,timeframe_map[timeframe], desde_dt, hasta_dt)

    if rates is None or len(rates) == 0:
        print(f"No se pudieron obtener datos para {symbol} en el período especificado")
        return None

    # Crear DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    print(f"Datos cargados: {len(df)} registros para {symbol} desde {desde} hasta {hasta}")

    # Devolver las columnas de cierre y volumen
    return df[['close', 'tick_volume']]



def dataset_loader_csv(filepath):
    """
    Carga datos históricos desde un archivo CSV exportado de MetaTrader 5.
    El archivo debe contener al menos las columnas: 'time', 'close', 'tick_volume'.
    """

    try:
        # Leer el archivo CSV
        df = pd.read_csv(filepath)

        # Verificar columnas mínimas requeridas
        required_cols = {'time', 'close', 'tick_volume'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"El archivo debe contener las columnas: {required_cols}")

        # Convertir 'time' a datetime (formato típico de MT5 es timestamp o ISO8601)
        try:
            # Si es timestamp numérico
            df['time'] = pd.to_datetime(df['time'], unit='s')
        except:
            # Si ya está en formato tipo 'YYYY-MM-DD HH:MM:SS'
            df['time'] = pd.to_datetime(df['time'])

        # Usar 'time' como índice
        df.set_index('time', inplace=True)

        print(f"Datos cargados correctamente desde {filepath} - Registros: {len(df)}")
        return df[['close', 'tick_volume']]

    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return None

