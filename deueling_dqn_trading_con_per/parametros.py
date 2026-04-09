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

    # Archivo CSV de datos históricos (formato MT5, tab-separado).
    # Cambiar este valor y NOMBRE_MODELO juntos para no mezclar checkpoints
    # de diferentes activos o temporalidades.
    NOMBRE_CSV = "GOLD#_M15_202112200200_202412311530.csv"

    SYMBOL = "GOLD"
    INTERVALO = "daily"  # Solo informativo para nombres de plots

    ES_INDICE = False
    ES_FOREX = False
    ES_METAL = True

    # TICK_VALUE y PIP_MULTIPLIER: determinan cuántos dólares vale cada punto de precio.
    # Para XAUUSD con 0.01 lots: ganancia = (precio_salida - precio_entrada) * TICK_VALUE * LOT_SIZE
    # = diferencia_precio * 100 * 0.01 = diferencia_precio * 1.0 $/punto
    # NO cambiar sin revisar todos los cálculos de profit en run_con_per.py.
    TICK_VALUE = 100        # 100 oz por lote estándar XAUUSD
    PIP_MULTIPLIER = 100    # 1 punto de precio = $0.01 × TICK_VALUE × LOT_SIZE

    # WINDOW_SIZE: cuántas velas pasadas ve el agente en cada estado.
    # Con TIPO_ESTADO='advanced': estado = WINDOW_SIZE*5 + 8 + 3 features.
    # Con WINDOW_SIZE=18: estado de 101 features (18 velas × 5 OHLCV + RSI+MACD+dia+hora + posición).
    #
    # Más alto (ej. 30): el agente ve más contexto, puede detectar patrones más largos,
    #   pero el estado crece, la red necesita más parámetros y tarda más en aprender.
    # Más bajo (ej. 10): estado más pequeño, aprendizaje más rápido, pero el agente
    #   ve menos contexto — puede perder señales de tendencia en M15.
    # Valor actual (18) cubre 4.5 horas de M15, suficiente para intraday.
    # IMPORTANTE: cambiar WINDOW_SIZE invalida modelos guardados (cambia el tamaño del estado).
    WINDOW_SIZE = 18

    # VENTANA_DATOS: ventana secundaria usada en algunos indicadores y plots.
    # No afecta el tamaño del estado ni el entrenamiento principal.
    VENTANA_DATOS = 4

    # TIPO_ESTADO: qué información incluye el estado del mercado.
    # 'ohlc'     → OHLCV normalizado + hora. Más simple, menos señal.
    # 'advanced' → OHLCV + RSI(14) + MACD(line/signal/hist) + día_sin/cos + hora_sin/cos.
    #   El RSI y MACD son los indicadores que más información aportan en GOLD M15
    #   porque capturan sobrecompra/sobreventa y momentum — señales que el agente
    #   no puede derivar solo de precios crudos en 18 velas.
    #   El encoding sin/cos del día y hora permite al agente aprender patrones
    #   intradía (apertura Londres, cierre NY) sin tratar las horas como valores ordinales.
    # IMPORTANTE: cambiar esto invalida modelos guardados (cambia tamaño del estado).
    TIPO_ESTADO = 'advanced'

    # TEST_SIZE_RATIO: fracción de datos reservada para evaluación final.
    # El 20% final (cronológicamente) nunca se usa para entrenar — solo para eval().
    # Reducir (ej. 0.1) da más datos de entrenamiento pero menos validación out-of-sample.
    # Aumentar (ej. 0.3) da más confianza en el resultado final pero menos datos para aprender.
    TEST_SIZE_RATIO = 0.2


# ==============================================================================
# 2. CONFIGURACIÓN DEL TRADING
# ==============================================================================

class ConfigTrading:
    """Parámetros de gestión de capital y costos"""

    # BALANCE_INICIAL: capital inicial en USD.
    # Este valor escala directamente el reward:
    #   - profit reward usa tanh(expectancy / (balance * 0.01))
    #   - upnl_norm usa tanh(upnl / (balance * 0.02))
    # Si cambias el balance, los rewards almacenados en memoria quedan en escala
    # diferente → poner CARGAR_MEMORIA_BUFFER = False.
    # Con 1000 USD y 0.01 lots, cada $10 de P&L representa 1% del balance —
    # suficientemente sensible para que el agente detecte diferencia entre trades.
    BALANCE_INICIAL = 1000

    # LOT_SIZE: tamaño de posición en lotes estándar.
    # 0.01 lots en GOLD = 1 oz. Un movimiento de $1 en el precio = $1 de P&L.
    # Aumentar (ej. 0.02) amplifica ganancias Y pérdidas. El reward escala igual
    # porque usa balance como denominador, pero el riesgo real de ruina aumenta.
    # Con balance $1000, 0.01 lots da riesgo razonable (~1% por cada $10 de move).
    LOT_SIZE = 0.01

    # COMMISSION_PER_TRADE: comisión fija por operación en USD.
    # Con XM Ultra Low el spread ya está en los datos del CSV → 0.0.
    # Si se cambia de broker o se agrega comisión explícita, esto afecta
    # directamente al P&L real y al cálculo de expectancy en el reward.
    COMMISSION_PER_TRADE = 0.0

    # SPREAD: spread fallback en puntos de precio (solo si el CSV no lo trae).
    # El spread real viene de la columna 'spread' del CSV de MT5.
    SPREAD = 0.20

    @property
    def PIP_VALUE_EUR_USD(cls):
        return 10 * cls.LOT_SIZE


# ==============================================================================
# 3. CONFIGURACIÓN DEL AGENTE DQN
# ==============================================================================

class ConfigAgente:
    """Parámetros del agente Dueling DQN"""

    ACTION_SPACE = 5  # Hold, Buy, Sell(close long), Short, Cover(close short). No cambiar.

    # GAMMA: factor de descuento — cuánto valora el agente recompensas futuras vs inmediatas.
    # Rango: 0.0 (solo presente) a 1.0 (futuro igual que presente).
    #
    # Para trading, un gamma alto (0.95-0.99) es crítico porque la consecuencia
    # de abrir una posición se realiza steps después (al cerrar).
    # Con gamma bajo (ej. 0.80): el agente no conecta el cierre de un trade
    # con la decisión de entrada → aprende a cerrar rápido sin importar el P&L.
    # Con gamma=0.98: una recompensa en 50 steps vale 0.98^50 ≈ 0.36 hoy.
    #   En M15, 50 velas = ~12 horas → el agente planifica hasta medio día.
    # Subir a 0.99 aumenta el horizonte de planificación pero también la varianza
    # del entrenamiento (los errores se propagan más lejos en el tiempo).
    GAMMA = 0.98

    # EPSILON: exploración epsilon-greedy. Con NoisyLinear activo, este valor
    # tiene efecto secundario porque NoisyLinear ya provee exploración intrínseca.
    # Se mantiene por compatibilidad — el decay reduce la chance de acción aleatoria
    # pura pero la red sigue explorando via ruido paramétrico incluso con epsilon bajo.
    EPSILON_INICIO = 1.0

    # EPSILON_FINAL: valor mínimo de epsilon al que se detiene el decay.
    # Con NoisyLinear, 0.05 significa que el 5% de las acciones son completamente
    # aleatorias (uniforme entre 5 acciones) — útil para evitar que el agente
    # quede atascado en una política local.
    EPSILON_FINAL = 0.05

    # EPSILON_DECAY: multiplicador por episodio.
    # Con 0.985 y 50 episodios: epsilon_final = 1.0 × 0.985^50 ≈ 0.47 (aún alto).
    # Con 0.985 y 100 episodios: ≈ 0.22.
    # Con 0.985 y 200 episodios: ≈ 0.047 ≈ EPSILON_FINAL.
    # Si EPISODES = 50, bajar a 0.960 para alcanzar 0.13 al final del fold,
    # o a 0.940 para alcanzar 0.05 exactamente al ep 50.
    # Decaer demasiado rápido hace que el agente explote prematuramente antes
    # de haber visto suficiente variedad de situaciones de mercado.
    EPSILON_DECAY = 0.985

    # USE_DOUBLE_DQN: evita sobreestimar Q-values.
    # Sin Double DQN, el agente tiende a ser demasiado optimista → abre
    # más posiciones de las que debería. Siempre True para este modelo.
    USE_DOUBLE_DQN = True

    # TARGET_MODEL_UPDATE: cada cuántos pasos del optimizer se sincronizan
    # los pesos del target network con la red online.
    # Demasiado bajo (ej. 50): el target se mueve rápido → entrenamiento inestable,
    #   los Q-values objetivo oscilan y la red no converge.
    # Demasiado alto (ej. 1000): el target queda obsoleto → el agente aprende
    #   contra información vieja, convergencia lenta.
    # Con TRAIN_FREQUENCY=20 y TRAIN_ITERATIONS=3, hay ~(14000/20)*3 = 2100
    #   pasos de optimizer por fold × episodio. Update cada 200 → ~10 syncs
    #   por fold, suficiente para estabilidad sin obsolescencia.
    TARGET_MODEL_UPDATE = 200

    # LEARNING_RATE: tasa de aprendizaje inicial del optimizer Adam.
    # El scheduler lo reduce durante el entrenamiento (ver ConfigScheduler).
    # 0.001 es el default de Adam y funciona bien para este modelo.
    # Bajar a 0.0005 si el entrenamiento es muy inestable (loss oscila mucho).
    # Subir a 0.003 si la convergencia es muy lenta, pero aumenta riesgo de divergencia.
    LEARNING_RATE = 0.001


# ==============================================================================
# 4. MEMORIA Y EXPERIENCE REPLAY
# ==============================================================================

class ConfigMemoria:
    """Parámetros de la memoria priorizada (PER)"""

    # MEMORY_SIZE: máximo de transiciones almacenadas en el SumTree.
    # Con ~2500 trades/episodio × 50 episodios = 125k transiciones por fold.
    # 250k cubre ~2 folds completos — transiciones más viejas se descartan.
    # Subir (ej. 500k): más diversidad de experiencias, menor correlación entre
    #   batches, pero consume más RAM (~500k × estado_size × 4 floats).
    # Bajar (ej. 100k): memoria se llena rápido, el agente olvida experiencias
    #   tempranas del fold → puede ser beneficioso si el mercado drift mucho.
    MEMORY_SIZE = 250000

    # ALPHA: grado de priorización del PER.
    # 0.0 → muestreo uniforme (igual que DQN estándar, ignora prioridades).
    # 1.0 → muestreo completamente guiado por TD error (solo replaya los peores casos).
    # 0.6 es el valor estándar de la literatura PER (Schaul et al. 2015).
    # Para trading: subir ALPHA (ej. 0.8) hace que el agente se enfoque más en
    #   los trades que más erró (grandes pérdidas inesperadas), pero puede causar
    #   overfitting a situaciones extremas raras.
    # Bajar (ej. 0.4) = más diversidad en el replay, aprendizaje más suave.
    ALPHA = 0.6

    # BETA_START: corrección de importance sampling al inicio del entrenamiento.
    # El PER introduce sesgo al muestrear no-uniformemente — BETA corrige ese sesgo.
    # 0.4 → corrección parcial al inicio (permite aprendizaje más agresivo).
    # Se anneala automáticamente hasta 1.0 (corrección completa) en BETA_FRAMES pasos.
    # No cambiar sin cambiar también BETA_FRAMES.
    BETA_START = 0.4

    # BETA_FRAMES: steps de optimizer para annealar BETA de 0.4 a 1.0.
    # Con ~2100 steps/fold × 4 folds × 50 episodes = ~420k steps totales.
    # 100k significa que BETA llega a 1.0 en el primer fold.
    # Aumentar (ej. 300k) para que la corrección sea más gradual a lo largo
    # de todo el entrenamiento.
    BETA_FRAMES = 100000

    # EPSILON_PRIORITY: constante pequeña que se suma al TD error antes de calcular
    # la prioridad: priority = (|TD_error| + EPSILON_PRIORITY)^ALPHA.
    # Evita que experiencias con TD error=0 tengan probabilidad cero de ser muestreadas.
    # 1e-3 es seguro. Bajar (ej. 1e-5) hace la priorización más extrema.
    # No hay razón para cambiar este valor en la práctica.
    EPSILON_PRIORITY = 1e-3


# ==============================================================================
# 5. SCHEDULER DEL LEARNING RATE
# ==============================================================================

class ConfigScheduler:
    """Parámetros del scheduler de learning rate"""

    # SCHEDULER_TYPE: estrategia de decaimiento del LR durante el entrenamiento.
    #
    # 'cosine_decay'      → baja suavemente de LR a LR_MIN siguiendo una curva coseno
    #                       en LR_DECAY_STEPS pasos, luego sube de nuevo.
    #                       ADVERTENCIA: con LR_DECAY_STEPS=1000 y ~600 steps/episodio,
    #                       el LR completa un ciclo cada ~1.7 episodios — demasiado rápido.
    #                       Para evitar esto, usar LR_DECAY_STEPS = total_steps_entrenamiento.
    #
    # 'reduce_on_plateau' → reduce LR a la mitad si no mejora en PATIENCE episodios.
    #                       Más conservador, reacciona a señales reales del entrenamiento.
    #                       Recomendado si 'cosine_decay' produce varianza alta entre episodios.
    #
    # 'exponential_decay' → multiplica LR × LR_DECAY_RATE cada LR_DECAY_STEPS pasos.
    #                       Simple pero irreversible — el LR solo baja y nunca sube.
    #
    # 'constant'          → LR fijo, sin scheduler. Útil para debugging o fine-tuning
    #                       con LR muy bajo desde el inicio.
    SCHEDULER_TYPE = 'reduce_on_plateau'

    # LR_DECAY_RATE: factor de multiplicación por step para 'exponential_decay'.
    # Solo se usa si SCHEDULER_TYPE = 'exponential_decay'.
    LR_DECAY_RATE = 0.97

    # LR_DECAY_STEPS: parámetro T_max para 'cosine_decay' (steps hasta LR_MIN),
    # o steps entre decaimientos para 'exponential_decay'.
    # Para 'cosine_decay': idealmente = folds × episodios × steps_optimizer_por_episodio.
    # Estimación actual: 4 folds × 50 eps × ~700 steps/ep ≈ 140k steps.
    # Con 1000, el ciclo coseno se repite ~140 veces durante el entrenamiento completo.
    LR_DECAY_STEPS = 1000

    # LR_MIN: learning rate mínimo que el scheduler no puede cruzar hacia abajo.
    # Con LR inicial = 0.001 y LR_MIN = 1e-5: rango de 100x.
    # No bajar de 1e-6 — por debajo de eso los gradientes son prácticamente cero
    # y el modelo deja de aprender.
    LR_MIN = 1e-5

    # PATIENCE: episodios sin mejora en la métrica de validación antes de reducir LR.
    # Solo se usa si SCHEDULER_TYPE = 'reduce_on_plateau'.
    # Con 50 episodios por fold, PATIENCE=15 significa que el agente tiene
    # 15 episodios para mejorar antes de que el LR se reduzca.
    # Bajar (ej. 8) hace el scheduler más reactivo pero puede reducir el LR
    # prematuramente durante fases normales de exploración.
    PATIENCE = 15

    # FACTOR: multiplicador del LR cuando se activa 'reduce_on_plateau'.
    # nueva_lr = lr_actual × FACTOR. Con 0.5 → se divide a la mitad.
    # Bajar (ej. 0.3) hace reducciones más agresivas.
    # Subir (ej. 0.7) hace reducciones más conservadoras.
    FACTOR = 0.5

    # COSINE_RESTARTS: si True, usa CosineAnnealingWarmRestarts (el LR sube bruscamente
    # cada T_0 steps). Si False, usa CosineAnnealingLR (curva suave sin saltos).
    # SIEMPRE False. Los warm restarts causaron episodios catástrofe (-250 a -308 pips)
    # en fold 3 porque cada salto de LR destruía la política de salida aprendida.
    # La habilidad de entrada (accuracy 65-69%) sobrevivía, pero la gestión de salida no.
    COSINE_RESTARTS = False


# ==============================================================================
# 6. SISTEMA DE RECOMPENSAS
# ==============================================================================

class ConfigReward:
    """Pesos del sistema de recompensas v2.0 (2026-04-08)

    Cada componente genera un valor en [-1, 1] vía tanh y se multiplica por su
    peso antes de sumarse al reward total del trade cerrado.

    NOTA: los nombres internos de los atributos acumuladores NO cambiaron para
    mantener compatibilidad con run_con_per.py y TensorBoard. Los componentes
    risk_adjusted y momentum fueron reemplazados conceptualmente:
      sumaRecompensaRiskAdjusted → loss_severity
      sumaRecompensaMomentum     → win_retention

    Guía de ajuste: verificar [REWARD BREAKDOWN] en el print del episodio.
    Si un componente acumula > 2× los demás, reducir su peso.
    """

    # PESO_PROFIT: expectancy real del episodio.
    # Fórmula: tanh(expectancy / (balance * 0.01))
    #   expectancy = (accuracy × avg_win) - ((1-accuracy) × avg_loss)
    # Si avg_loss > avg_win → negativo aunque accuracy > 50%. Señal correcta.
    # Es el componente más importante. No bajar de 0.8.
    PESO_PROFIT = 1.0

    # PESO_SHARPE: rolling Sharpe sobre últimos 50 trades cerrados.
    # Incentiva ganancias consistentes en relación al riesgo.
    # Mínimo 15 trades para activarse (v2, antes 10).
    PESO_SHARPE = 0.3

    # PESO_DRAWDOWN: penaliza el INCREMENTO de drawdown entre cierres de trades.
    # v2: ya NO acumula cuando el drawdown es persistente. Solo penaliza cuando
    # el drawdown empeora en un cierre. Elimina el problema de acumulación de v1
    # que llegaba a -106 por episodio en fold 3.
    PESO_DRAWDOWN = 0.3

    # PESO_CONSISTENCY: coeficiente de variación de retornos.
    # Reducido de 0.2 a 0.1 — señal secundaria, no debe dominar.
    PESO_CONSISTENCY = 0.1

    # PESO_RISK_ADJUSTED: en v2 = loss_severity.
    # Penaliza cuando un loser específico es peor que el promedio histórico del buffer.
    # PROBLEMA DETECTADO (2026-04-08 ep1-8): con ~1500 losses por episodio, exactamente
    # ~50% son peores que la media del buffer por definición estadística → ~750 trades
    # × penalización fija ≈ -100 constante en TODOS los episodios. No discrimina entre
    # un episodio de -133 pips y uno de -548 pips. Ruido, no señal.
    # Reducido de 0.5 a 0.02 para que el total sea ~-4 en lugar de ~-100.
    # Mantiene la presión direccional correcta pero no domina el reward total.
    PESO_RISK_ADJUSTED = 0.02

    # PESO_MOMENTUM: en v2 = win_retention.
    # Premia cuando un winner supera el promedio histórico del buffer.
    # PROBLEMA DETECTADO (2026-04-08 ep1-8): mismo problema que loss_severity en positivo.
    # ~50% de winners superan el buffer promedio → +37 constante en todos los episodios,
    # independiente del P&L real. Da señal positiva mientras el agente pierde -400 pips.
    # Reducido de 0.2 a 0.02 para que el total sea ~+0.75 en lugar de ~+37.
    PESO_MOMENTUM = 0.02

    # PESO_TRADE_QUALITY: delta del ratio avg_win/avg_loss entre trades consecutivos.
    # v2.1: usa delta, no ratio absoluto. Ratio estable → ~0 (verificado: +0.05 a +2.65
    # en eps 1-8, correcto). Solo dispara cuando el ratio cambia.
    # 0.3 funciona bien — no acumula, no domina.
    PESO_TRADE_QUALITY = 0.3

    # RISK_FREE_RATE: tasa libre de riesgo anual para el Sharpe (/ 252 = diaria).
    RISK_FREE_RATE = 0.02

    @staticmethod
    def get_pesos():
        """Retorna diccionario de pesos para AdvancedRewardSystem."""
        return {
            'profit':        ConfigReward.PESO_PROFIT,
            'sharpe':        ConfigReward.PESO_SHARPE,
            'drawdown':      ConfigReward.PESO_DRAWDOWN,
            'consistency':   ConfigReward.PESO_CONSISTENCY,
            'risk_adjusted': ConfigReward.PESO_RISK_ADJUSTED,
            'momentum':      ConfigReward.PESO_MOMENTUM,
            'trade_quality': ConfigReward.PESO_TRADE_QUALITY,
        }


# ==============================================================================
# 7. ENTRENAMIENTO
# ==============================================================================

class ConfigEntrenamiento:
    """Parámetros del proceso de entrenamiento"""

    # EPISODES: episodios de entrenamiento por fold.
    # En fold 2 el agente seguía convergiendo al ep 50 — 50 es insuficiente.
    # El breakthrough (primer episodio fuertemente positivo) ocurrió en ep 37 de fold 2,
    # ep 16 de fold 3, ep 8 de fold 4 — la convergencia acelera entre folds.
    # Recomendado: 100-150 para dar tiempo de consolidación post-breakthrough.
    # Cada episodio en fold de ~14k velas M15 tarda ~X minutos (medir con cProfile).
    EPISODES = 500

    # N_FOLDS: número de folds en la validación cruzada walk-forward.
    # Cada fold es una porción cronológica del 80% de datos de entrenamiento.
    # Con 3 años de datos M15 (~100k velas) y 4 folds: cada fold tiene ~20k velas.
    # Aumentar (ej. 6): más diversidad de condiciones de mercado, mejor validación,
    #   pero cada fold es más corto → menos episodios útiles por fold.
    # Bajar (ej. 2): folds más largos, más velas por episodio, pero menos validación.
    N_FOLDS = 4

    # BATCH_SIZE: número de transiciones por batch en cada paso de optimizer.
    # 256 es estándar y equilibra velocidad vs. estabilidad del gradiente.
    # Subir (ej. 512): gradientes más estables, menos ruido en updates, pero
    #   más lento por paso y más uso de VRAM.
    # Bajar (ej. 128): updates más rápidos pero gradientes más ruidosos.
    # No cambiar sin evidencia de que la convergencia mejora.
    BATCH_SIZE = 256

    # TRAIN_FREQUENCY: cada cuántos steps del episodio se llama a batch_train.
    # Con M15 y TRAIN_FREQUENCY=20: se entrena cada 20 velas = cada 5 horas de mercado.
    # Bajar (ej. 5): más updates por episodio → aprendizaje más intensivo pero
    #   mayor correlación entre batches consecutivos (el mercado no cambió mucho).
    # Subir (ej. 50): menos updates, más variedad entre batches, pero el agente
    #   aprende más lento por episodio.
    # El comentario en el código dice "cada 5 steps" pero el valor es 20 — revisar.
    TRAIN_FREQUENCY = 20

    # TRAIN_ITERATIONS: cuántas veces se llama a batch_train por cada TRAIN_FREQUENCY steps.
    # Con 3: por cada 20 velas se hacen 3 gradient updates.
    # Subir (ej. 5): aprendizaje más intensivo por episodio, útil si EPISODES es bajo,
    #   pero aumenta riesgo de sobreajustar a las últimas experiencias en memoria.
    # Bajar a 1: más conservador, equivalente a DQN estándar.
    TRAIN_ITERATIONS = 3

    # GUARDAR_MODELO_CADA: frecuencia de guardado del checkpoint en episodios.
    # Con 10: se guarda en ep 10, 20, 30... El último episodio siempre guarda.
    # Bajar (ej. 5) da más granularidad para recuperar el mejor punto si el
    # entrenamiento colapsa. Aumentar (ej. 20) reduce I/O en disco.
    GUARDAR_MODELO_CADA = 10


# ==============================================================================
# 8. CARGAR/GUARDAR MODELO
# ==============================================================================

class ConfigModelo:
    """Parámetros para cargar/guardar modelos"""

    NOMBRE_MODELO = "model_" + ConfigEntorno.NOMBRE_CSV

    DIRECTORIO_RESULTADOS = 'resultados_cv'

    # CARGAR_MODELO: si True, carga el checkpoint de MODELO_EXISTENTE antes de entrenar.
    # Los parámetros del checkpoint (epsilon, LR, scheduler state, etc.) pisan
    # los de parametros.py — ver load_model() en dueling_dqn_con_per.py.
    # Poner False para empezar desde cero con los parámetros actuales de este archivo.
    CARGAR_MODELO = False
    MODELO_EXISTENTE = "resultados_cv/model_GOLD#_M15_202112200200_202412311530.csv"

    # CARGAR_MEMORIA_BUFFER: si True, restaura el SumTree con las transiciones previas.
    # Ventaja: el agente retoma el entrenamiento con experiencias acumuladas → menos
    #   episodios en "frío" al inicio del nuevo fold.
    # Desventaja: si se cambió BALANCE_INICIAL, LOT_SIZE o la función de reward,
    #   las experiencias antiguas tienen rewards en escala diferente → gradientes corruptos.
    # Regla: False si se cambió cualquier cosa que afecte el valor numérico del reward.
    CARGAR_MEMORIA_BUFFER = True


# ==============================================================================
# 9. BACKEND Y NOTIFICACIONES
# ==============================================================================

class ConfigBackend:
    """Parámetros de integración con backend y notificaciones"""

    GUARDAR_ESTADISTICAS_EN_BACKEND = True

    URL_BACKEND = 'https://back-para-entrenamiento.onrender.com/api/upload'

    # MOSTRAR_PRINTS: activa prints detallados por step durante el episodio.
    # False para entrenamiento normal (reduce I/O y acelera el loop).
    # True solo para debugging de comportamiento del agente en steps específicos.
    MOSTRAR_PRINTS = False

    # TENSORBOARD: activa el logger de TensorBoard.
    # False en entornos sin display o al entrenar en la nube.
    TENSORBOARD = True

    # LIVE_PLOT: muestra el gráfico de compras/ventas en tiempo real durante training.
    # Útil para ver visualmente qué hace el agente, pero agrega overhead de matplotlib.
    # Desactivar si el entrenamiento es lento o se corre en servidor sin display.
    LIVE_PLOT = False
    LIVE_PLOT_WINDOW = 150   # Cuántas velas mostrar en el gráfico
    LIVE_PLOT_UPDATE = 100   # Actualizar cada N steps del episodio


# ==============================================================================
# 10. MÉTRICAS Y LOGGING
# ==============================================================================

class ConfigMetricas:
    """Parámetros para métricas y logging"""

    # Estos valores eran para get_adaptive_weights() — función eliminada (2026-03-18).
    # Se mantienen por si se reimplementa lógica de pesos adaptativos.
    ADAPTIVE_WEIGHTS_EPISODE_1 = 250
    ADAPTIVE_WEIGHTS_EPISODE_2 = 400

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
        # Exploración
        'epsilon': ConfigAgente.EPSILON_INICIO,
        'epsilon_final': ConfigAgente.EPSILON_FINAL,
        'epsilon_decay': ConfigAgente.EPSILON_DECAY,
        # Agente
        'gamma': ConfigAgente.GAMMA,
        'use_double_dqn': ConfigAgente.USE_DOUBLE_DQN,
        'target_model_update': ConfigAgente.TARGET_MODEL_UPDATE,
        'learning_rate': ConfigAgente.LEARNING_RATE,
        # Trading
        'commission_per_trade': ConfigTrading.COMMISSION_PER_TRADE,
        # Memoria PER
        'memory_size': ConfigMemoria.MEMORY_SIZE,
        'alpha': ConfigMemoria.ALPHA,
        'beta_start': ConfigMemoria.BETA_START,
        'beta_frames': ConfigMemoria.BETA_FRAMES,
        'epsilon_priority': ConfigMemoria.EPSILON_PRIORITY,
        # Scheduler
        'scheduler_type': ConfigScheduler.SCHEDULER_TYPE,
        'cosine_restarts': ConfigScheduler.COSINE_RESTARTS,
        'lr_decay_rate': ConfigScheduler.LR_DECAY_RATE,
        'lr_decay_steps': ConfigScheduler.LR_DECAY_STEPS,
        'lr_min': ConfigScheduler.LR_MIN,
        'patience': ConfigScheduler.PATIENCE,
        'factor': ConfigScheduler.FACTOR,
        # Entrenamiento
        'batch_size': ConfigEntrenamiento.BATCH_SIZE,
        'train_frequency': ConfigEntrenamiento.TRAIN_FREQUENCY,
        'train_iterations': ConfigEntrenamiento.TRAIN_ITERATIONS,
        # Reward weights
        'reward_weights': ConfigReward.get_pesos(),
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
