# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:49:04 2025

@author: fabia
"""

import MetaTrader5 as mt5

def initialize_mt5():
    """Inicializa la conexión con MetaTrader 5"""
    if not mt5.initialize():
        print("Error al inicializar MetaTrader 5")
        print(f"Error: {mt5.last_error()}")
        return False
    else:
        print("MetaTrader 5 inicializado correctamente")
        # Mostrar información sobre la versión de MetaTrader 5
        print(mt5.version())
        return True
