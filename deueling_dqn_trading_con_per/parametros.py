# -*- coding: utf-8 -*-
"""
Archivo centralizado de parámetros configurables para el sistema de trading RL.
Todos los parámetros del proyecto se encuentran aquí para fácil modificación.

@author: fabia
"""

# ==============================================================================
# 1. CONFIGURACIÓN DEL ENTORNO Y DATOS
# ==============================================================================

class ConfigEntorno:
    """Parámetros relacionados con los datos y el mercado"""

    # Nombre del archivo CSV con datos históricos
    NOMBRE_CSV = "XAUUSD_H1_2015_01_01_2024_05_31.csv"

    # Símbolo y temporalidad
    SYMBOL = "GOLD"
    INTERVALO = "daily"  # daily, hourly, etc.

    # Tipo de activo
    ES_INDICE = False
    ES_FOREX = False
    ES_METAL = True

    # Valor del pip y multiplicador (para XAUUSD/GOLD: tick_value=5, pip_multiplier=10000)
    TICK_VALUE = 5          # Valor monetario de 1 pip
    PIP_MULTIPLIER = 10000  # Pips decimales (10000 para XAUUSD = 0.01)

    # Tamaño de ventana para crear estados
    WINDOW_SIZE = 18
    VENTANA_DATOS = 4

    # Tipo de estado a usar:
    # 'ohlc'       = solo OHLC + hora (actual, básico)
    # 'advanced'   = OHLC + RSI + MACD + día + hora (nuevo, más completo)
    TIPO_ESTADO = 'advanced'

    # División train/test
    TEST_SIZE_RATIO = 0.2   # 20% para prueba


# ==============================================================================
# 2. CONFIGURACIÓN DEL TRADING
# ==============================================================================

class ConfigTrading:
    """Parámetros de gestión de capital y costos"""

    # Balance inicial
    BALANCE_INICIAL = 100

    # Tamaño de lote (lots)
    LOT_SIZE = 0.01

    # Comisión por operación (en dólares)
    COMMISSION_PER_TRADE = 4.5

    # Spread (usado para cálculos internos)
    SPREAD = 0.20

    # Valor del pip en USD (10 * lot_size para XAUUSD)
    @property
    def PIP_VALUE_EUR_USD(cls):
        return 10 * cls.LOT_SIZE


# ==============================================================================
# 3. CONFIGURACIÓN DEL AGENTE DQN
# ==============================================================================

class ConfigAgente:
    """Parámetros del agente Dueling DQN"""

    # Espacio de acciones (5: Hold, Buy, Sell, Short, Cover)
    ACTION_SPACE = 5

    # Factor de descuento (gamma)
    GAMMA = 0.98

    # Exploración con Noisy Networks (epsilon para backward compatibility)
    EPSILON_INICIO = 1.0
    EPSILON_FINAL = 0.15
    EPSILON_DECAY = 0.995   # Factor de decaimiento por episodio

    # Double DQN
    USE_DOUBLE_DQN = True

    # Frecuencia de actualización del target network
    TARGET_MODEL_UPDATE = 200

    # Learning rate
    LEARNING_RATE = 0.001


# ==============================================================================
# 4. MEMORIA Y EXPERIENCE REPLAY
# ==============================================================================

class ConfigMemoria:
    """Parámetros de la memoria priorizada (PER)"""

    # Tamaño máximo de la memoria
    MEMORY_SIZE = 250000

    # Alpha: grado de priorización (0 = uniform, 1 = totalmente priorizado)
    ALPHA = 0.6

    # Beta: corrección de importancia (0 = sin corrección, 1 = corrección total)
    BETA_START = 0.4
    BETA_FRAMES = 100000  # Frames para alcanzar beta = 1

    # Epsilon pequeño para evitar probabilidad cero
    EPSILON_PRIORITY = 1e-3


# ==============================================================================
# 5. SCHEDULER DEL LEARNING RATE
# ==============================================================================

class ConfigScheduler:
    """Parámetros del scheduler de learning rate"""

    # Tipos disponibles: 'exponential_decay', 'cosine_decay', 'polynomial_decay', 'reduce_on_plateau', 'constant'
    SCHEDULER_TYPE = 'cosine_decay'

    # Parámetros para exponential_decay
    LR_DECAY_RATE = 0.97     # LR se multiplica por este factor cada LR_DECAY_STEPS
    LR_DECAY_STEPS = 1000    # Pasos entre decaimientos

    # Learning rate mínimo
    LR_MIN = 1e-5

    # Parámetros para reduce_on_plateau
    PATIENCE = 10            # Épocas sin mejora antes de reducir LR
    FACTOR = 0.5             # Factor de reducción (nueva_lr = lr * factor)

    # Parámetros para cosine_decay
    COSINE_RESTARTS = True   # Si True, usa CosineAnnealingWarmRestarts


# ==============================================================================
# 6. SISTEMA DE RECOMPENSAS
# ==============================================================================

class ConfigReward:
    """Pesos del sistema de recompensas avanzadas"""

    # Pesos de cada componente de recompensa - valores optimizados para trading
    PESO_PROFIT = 1.0        # Recompensa base por profit
    PESO_SHARPE = 0.3        # Incentiva retornos consistentes
    PESO_DRAWDOWN = 0.3     # Penaliza Drawdown - reducido para evitar sobre-penalización
    PESO_CONSISTENCY = 0.2   # Incentiva estabilidad en los retornos
    PESO_RISK_ADJUSTED = 0.3 # Incentiva buen ratio retorno/riesgo
    PESO_MOMENTUM = 0.15     # Incentiva seguir tendencias
    PESO_TRADE_QUALITY = 0.25 # Recompensa por buenas operaciones

    # Tasa libre de riesgo (anual) - se convierte a diaria internamente
    RISK_FREE_RATE = 0.02

    # Ruido adicional en recompensas
    REWARD_NOISE_STD = 0.01

    @staticmethod
    def get_pesos():
        """Retorna diccionario de pesos"""
        return {
            'profit': ConfigReward.PESO_PROFIT,
            'sharpe': ConfigReward.PESO_SHARPE,
            'drawdown': ConfigReward.PESO_DRAWDOWN,
            'consistency': ConfigReward.PESO_CONSISTENCY,
            'risk_adjusted': ConfigReward.PESO_RISK_ADJUSTED,
            'momentum': ConfigReward.PESO_MOMENTUM,
            'trade_quality': ConfigReward.PESO_TRADE_QUALITY
        }


# ==============================================================================
# 7. ENTRENAMIENTO
# ==============================================================================

class ConfigEntrenamiento:
    """Parámetros del proceso de entrenamiento"""

    # Número de episodios por fold
    EPISODES = 5

    # Número de folds para validación cruzada
    N_FOLDS = 2

    # Batch size para entrenamiento
    BATCH_SIZE = 256

    # Frecuencia de entrenamiento (cada cuántos steps entrenar)
    TRAIN_FREQUENCY = 5     # Entrenar cada 5 steps
    TRAIN_ITERATIONS = 3    # Iteraciones por step

    # Guardar modelo cada cuántos episodios
    GUARDAR_MODELO_CADA = 5


# ==============================================================================
# 8. CARGAR/GUARDAR MODELO
# ==============================================================================

class ConfigModelo:
    """Parámetros para cargar/guardar modelos"""

    # Nombre del modelo a guardar
    NOMBRE_MODELO = "model_" + ConfigEntorno.NOMBRE_CSV

    # Ruta donde guardar resultados
    DIRECTORIO_RESULTADOS = 'resultados_cv'

    # ¿Cargar modelo existente?
    CARGAR_MODELO = False
    MODELO_EXISTENTE = "resultados_cv/model_XAUUSD_M15_2025_03_01_2025_03_31.csv"

    # ¿Cargar memoria previa?
    CARGAR_MEMORIA_BUFFER = True


# ==============================================================================
# 9. BACKEND Y NOTIFICACIONES
# ==============================================================================

class ConfigBackend:
    """Parámetros de integración con backend y notificaciones"""

    # Guardar estadísticas en backend
    GUARDAR_ESTADISTICAS_EN_BACKEND = True

    # Subir a Dropbox
    GUARDAR_EN_DROPBOX = False

    # URL del backend
    URL_BACKEND = 'https://back-para-entrenamiento.onrender.com/api/upload'

    # Mostrar prints durante entrenamiento
    MOSTRAR_PRINTS = False

    # TensorBoard: True = activo, False = desactivado (para entrenar en la nube)
    TENSORBOARD = True


# ==============================================================================
# 10. MÉTRICAS Y LOGGING
# ==============================================================================

class ConfigMetricas:
    """Parámetros para métricas y logging"""

    # Épocas para calcular adaptive weights (si se usa)
    ADAPTIVE_WEIGHTS_EPISODE_1 = 250
    ADAPTIVE_WEIGHTS_EPISODE_2 = 400

    # Guardar gráficos de entrenamiento
    GUARDAR_GRAFICOS = True


# ==============================================================================
# UTILIDADES
# ==============================================================================

def get_state_size():
    """Calcula el tamaño del estado basado en window_size y tipo de estado"""
    window = ConfigEntorno.WINDOW_SIZE
    tipo = ConfigEntorno.TIPO_ESTADO

    if tipo == 'ohlc':
        # OHLC * window + hora_sin/cos + has_long + has_short + upnl_norm
        # 5 features * window + 2 (hora) + 3 (posición)
        base_state = window * 5 + 2
    elif tipo == 'advanced':
        # OHLC * window + RSI + MACD(3) + dia(2) + hora(2)
        # 5 * window + 1 + 3 + 2 + 2 = 5*window + 8
        base_state = window * 5 + 8
    else:
        raise ValueError(f"TIPO_ESTADO desconocido: {tipo}")

    # Agregar features de posición (has_long, has_short, upnl_norm)
    return base_state + 3


def get_config_trader():
    """Retorna diccionario con todos los parámetros para AI_Trader_per"""
    return {
        'epsilon_decay': ConfigAgente.EPSILON_DECAY,
        'commission_per_trade': ConfigTrading.COMMISSION_PER_TRADE,
        'gamma': ConfigAgente.GAMMA,
        'target_model_update': ConfigAgente.TARGET_MODEL_UPDATE,
        'memory_size': ConfigMemoria.MEMORY_SIZE,
        'alpha': ConfigMemoria.ALPHA,
        'beta_start': ConfigMemoria.BETA_START,
        'beta_frames': ConfigMemoria.BETA_FRAMES,
        'epsilon_priority': ConfigMemoria.EPSILON_PRIORITY,
        'scheduler_type': ConfigScheduler.SCHEDULER_TYPE,
        'learning_rate': ConfigAgente.LEARNING_RATE,
        'lr_decay_rate': ConfigScheduler.LR_DECAY_RATE,
        'lr_decay_steps': ConfigScheduler.LR_DECAY_STEPS,
        'lr_min': ConfigScheduler.LR_MIN,
    }


def imprimir_config():
    """Imprime la configuración actual"""
    print("=" * 60)
    print("CONFIGURACIÓN ACTUAL DEL SISTEMA")
    print("=" * 60)

    print("\n[ENTORNO]")
    print(f"  CSV: {ConfigEntorno.NOMBRE_CSV}")
    print(f"  Symbol: {ConfigEntorno.SYMBOL}")
    print(f"  Window Size: {ConfigEntorno.WINDOW_SIZE}")

    print("\n[TRADING]")
    print(f"  Balance Inicial: ${ConfigTrading.BALANCE_INICIAL}")
    print(f"  Lot Size: {ConfigTrading.LOT_SIZE}")
    print(f"  Comisión: ${ConfigTrading.COMMISSION_PER_TRADE}")

    print("\n[AGENTE]")
    print(f"  Gamma: {ConfigAgente.GAMMA}")
    print(f"  Learning Rate: {ConfigAgente.LEARNING_RATE}")
    print(f"  Epsilon Decay: {ConfigAgente.EPSILON_DECAY}")
    print(f"  Double DQN: {ConfigAgente.USE_DOUBLE_DQN}")

    print("\n[MEMORIA]")
    print(f"  Memory Size: {ConfigMemoria.MEMORY_SIZE}")
    print(f"  Alpha (PER): {ConfigMemoria.ALPHA}")
    print(f"  Beta Start: {ConfigMemoria.BETA_START}")

    print("\n[SCHEDULER]")
    print(f"  Tipo: {ConfigScheduler.SCHEDULER_TYPE}")
    print(f"  LR Decay Rate: {ConfigScheduler.LR_DECAY_RATE}")

    print("\n[REWARD]")
    pesos = ConfigReward.get_pesos()
    for k, v in pesos.items():
        print(f"  {k}: {v}")

    print("\n[ENTRENAMIENTO]")
    print(f"  Episodes: {ConfigEntrenamiento.EPISODES}")
    print(f"  N Folds: {ConfigEntrenamiento.N_FOLDS}")
    print(f"  Batch Size: {ConfigEntrenamiento.BATCH_SIZE}")

    print("=" * 60)


# ============================================================================
# CLASE LEGACY PARA COMPATIBILIDAD (usar las clases acima)
# ============================================================================

# Variables sueltas para compatibilidad hacia atrás
NOMBRE_CSV = ConfigEntorno.NOMBRE_CSV
SYMBOL = ConfigEntorno.SYMBOL
INTERVALO = ConfigEntorno.INTERVALO
ES_INDICE = ConfigEntorno.ES_INDICE
ES_FOREX = ConfigEntorno.ES_FOREX
ES_METAL = ConfigEntorno.ES_METAL
TICK_VALUE = ConfigEntorno.TICK_VALUE
PIP_MULTIPLIER = ConfigEntorno.PIP_MULTIPLIER
WINDOW_SIZE = ConfigEntorno.WINDOW_SIZE
VENTANA_DATOS = ConfigEntorno.VENTANA_DATOS
TEST_SIZE_RATIO = ConfigEntorno.TEST_SIZE_RATIO

BALANCE_FIRST = ConfigTrading.BALANCE_INICIAL
LOT_SIZE = ConfigTrading.LOT_SIZE
COMMISSION_PER_TRADE = ConfigTrading.COMMISSION_PER_TRADE

GAMMA = ConfigAgente.GAMMA
EPSILON_DECAY = ConfigAgente.EPSILON_DECAY
LEARNING_RATE = ConfigAgente.LEARNING_RATE

N_FOLDS = ConfigEntrenamiento.N_FOLDS
EPISODES = ConfigEntrenamiento.EPISODES
BATCH_SIZE = ConfigEntrenamiento.BATCH_SIZE

MEMORY_SIZE = ConfigMemoria.MEMORY_SIZE
ALPHA = ConfigMemoria.ALPHA
BETA_START = ConfigMemoria.BETA_START
BETA_FRAMES = ConfigMemoria.BETA_FRAMES
EPSILON_PRIORITY = ConfigMemoria.EPSILON_PRIORITY

SCHEDULER_TYPE = ConfigScheduler.SCHEDULER_TYPE
LR_DECAY_RATE = ConfigScheduler.LR_DECAY_RATE
LR_DECAY_STEPS = ConfigScheduler.LR_DECAY_STEPS
LR_MIN = ConfigScheduler.LR_MIN
PATIENCE = ConfigScheduler.PATIENCE
FACTOR = ConfigScheduler.FACTOR
COSINE_RESTARTS = ConfigScheduler.COSINE_RESTARTS

CADA_CUANTO_ACTUALIZAR = ConfigAgente.TARGET_MODEL_UPDATE

CADA_CUANTOS_EPISODES_GUARDAR_EL_MODELO = ConfigEntrenamiento.GUARDAR_MODELO_CADA