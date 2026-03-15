# Contexto del Proyecto — Trading RL con Dueling DQN + PER

Estoy desarrollando un agente de trading con Reinforcement Learning para operar XAUUSD (Oro) en el broker XM, cuenta Ultra Low. Necesito ayuda para mejorar el sistema.

---

## Stack técnico

- **Python**, PyTorch, Numba (JIT)
- **Algoritmo**: Dueling DQN con Prioritized Experience Replay (PER)
- **Red neuronal**: LayerNorm → FC(512→256→128) + LeakyReLU + Dropout(0.02) → streams Value y Advantage
- **Exploración**: NoisyLinear (reemplaza epsilon-greedy)
- **Entrenamiento**: K-Fold Walk-Forward Cross-Validation (3 folds, 15 episodios por fold)
- **Datos**: CSV de MetaTrader 5, XAUUSD H1, 2015–2024

---

## Arquitectura del sistema

### Espacio de acciones (5 acciones)
| Acción | Significado |
|--------|-------------|
| 0 | Hold |
| 1 | Buy (abrir long) |
| 2 | Sell (cerrar long) |
| 3 | Short sell (abrir short) |
| 4 | Cover (cerrar short) |

Solo una posición abierta a la vez (long o short). Las posiciones se cierran a la fuerza a las 23h.

### Estado del agente
Tipo `advanced`: OHLCV × window_size(18) + RSI(14) + MACD(línea/señal/histograma) + día sin/cos + hora sin/cos + [has_long, has_short, upnl_norm]
- Tamaño del estado: 18×5 + 8 + 3 = **101 features**

### Sistema de recompensas (AdvancedRewardSystem)
Recompensa calculada al cerrar trade:
- **profit**: `tanh(profit_dollars)`
- **sharpe**: Sharpe rolling últimos 50 trades × peso 0.3
- **drawdown**: penalización si equity cae × peso 0.2
- consistency, risk_adjusted, momentum, trade_quality: peso 0 (desactivados)

Entre trades: step reward = `tanh(upnl / (balance × 2%)) × 0.01`

---

## Parámetros clave (configurados para XM Ultra Low)

```python
# Broker / Trading
BALANCE_INICIAL = 100       # USD
LOT_SIZE = 0.01             # micro lote = 1 oz de oro
COMMISSION_PER_TRADE = 0.0  # XM Ultra Low: sin comisión, costo en spread
SPREAD = 0.20               # fallback; el spread real viene del CSV MT5

# P&L correcto para XAUUSD
pip_value = 100 * lot_size  # = 1.0 → $1 movimiento de precio = $1 P&L para 0.01 lots
# Fórmula: profit_dollars = (sell_price - buy_price) * 1.0

# Agente
GAMMA = 0.98
LEARNING_RATE = 0.001
EPSILON_DECAY = 0.995       # por episodio
TARGET_MODEL_UPDATE = 200   # pasos
BATCH_SIZE = 256

# PER (Prioritized Experience Replay)
MEMORY_SIZE = 250000
ALPHA = 0.6
BETA_START = 0.4

# Entrenamiento
EPISODES = 15
N_FOLDS = 3
TRAIN_FREQUENCY = 5         # entrenar cada 5 steps
TRAIN_ITERATIONS = 3        # iteraciones por step
```

---

## Bugs ya corregidos

1. **pip_value incorrecto**: estaba `10 * lot_size = 0.1` → corregido a `100 * lot_size = 1.0` para XAUUSD
2. **Comisión ficticia**: estaba $4.50/trade → corregido a $0.0 (XM Ultra Low)
3. **Spread inconsistente train vs test**: el test no multiplicaba por `lot_size` → corregido
4. **Pesos de recompensa ignorados**: `AdvancedRewardSystem` tenía pesos hardcodeados → ahora lee `ConfigReward.get_pesos()`
5. **Scaler del test = scaler del último fold**: ahora usa un `global_scaler` ajustado en todos los datos de entrenamiento

---

## Resultado actual (episodio 5/15, antes de correcciones)

```
Beneficio (pips) = -92.52
Beneficio (usd)  = -$90.34
Trades           = 1802         ← overtrading
Wins             = 920 (51.05%)
Sharpe           = -0.01
Drawdown         = 91.25%       ← cuenta casi destruida
Equity           = $9.66 / $100
```

El problema principal identificado: **overtrading masivo** (1802 trades por episodio) y **sin selectividad** (el agente no aprende cuándo NO operar).

---

## Problemas pendientes / áreas de mejora

1. **Overtrading**: el agente hace demasiadas operaciones. ¿Cómo penalizar el exceso de trades?
2. **Convergencia lenta**: solo 15 episodios × 3 folds = 45 episodios totales. ¿Es suficiente?
3. **Balance de $100 con pip_value=1.0**: un movimiento adverso de $100 en oro destruye la cuenta. ¿Cómo manejar esto en el reward para que el agente aprenda gestión de riesgo?
4. **NoisyLinear + epsilon decay**: se usan ambos mecanismos de exploración. ¿Son redundantes?
5. **Recompensa escasa (sparse)**: la mayoría de los pasos tienen reward=0. El step reward mitiga esto pero es muy pequeño (0.01).

---

## Pregunta inicial sugerida

"Dado este contexto, ¿qué cambios concretos harías para reducir el overtrading y mejorar la convergencia del agente? El agente está haciendo 1802 trades por episodio en datos H1 de XAUUSD con $100 de balance."
