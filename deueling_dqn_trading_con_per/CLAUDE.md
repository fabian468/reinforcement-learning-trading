# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
# Train the agent (entry point)
python run_con_per.py
```

The script runs with cProfile enabled by default and prints the 20 slowest functions after training finishes.

## Installing dependencies

```bash
pip install -r requirements.txt
```

PyTorch is configured for CUDA 11.8 (`cu118`). For CPU-only, replace the torch lines in `requirements.txt` with the standard PyTorch packages.

## Environment variables

El `.env` ya no es obligatorio. Dropbox fue eliminado completamente del proyecto.

## Architecture overview

### Training loop (`run_con_per.py`)

The main function implements **k-fold walk-forward cross-validation** on time-series trading data:

1. Load CSV from `data/` → split 80/20 into train/test
2. Divide train data into `N_FOLDS` sequential folds
3. For each fold, pre-compute all states at once (`create_all_states_*`) and run `EPISODES` episodes
4. Every 5 steps during an episode, call `batch_train` 3 times
5. After all folds, run a final evaluation on the test set in `eval()` mode (epsilon=0)

The state passed to the agent is `market_state + [has_long, has_short, upnl_norm]` (see `get_full_state`).

Positions are force-closed at hour 23 of each trading day to avoid overnight holds.

### Action space (5 actions)

| Action | Meaning |
|--------|---------|
| 0 | Hold |
| 1 | Buy (open long) |
| 2 | Sell (close long) |
| 3 | Short sell (open short) |
| 4 | Cover (close short) |

Only one position (long or short) can be open at a time.

### Neural network (`model_pytorch.py` + `NoisyLinear.py`)

`DuelingDQN`: LayerNorm → 3 FC layers (512→256→128) with LeakyReLU and Dropout(0.02) → two streams:
- **Value stream**: Linear(128→64) → NoisyLinear(64→1)
- **Advantage stream**: Linear(128→128) → NoisyLinear(128→action_space)

NoisyLinear provides intrinsic exploration, replacing pure epsilon-greedy. `reset_noise()` is called after each `batch_train`.

### Agent (`dueling_dqn_con_per.py`)

`AI_Trader_per` implements:
- **Double DQN**: online model selects actions, target model evaluates them; target synced every `TARGET_MODEL_UPDATE` steps
- **Prioritized Experience Replay (PER)**: `SumTree` stores transitions with priority = `(|TD_error| + ε)^α`; importance-sampling weights correct bias
- **Mixed precision** (AMP): auto-enabled when CUDA is available
- **LR schedulers**: configurable via `ConfigScheduler.SCHEDULER_TYPE` (`cosine_decay` default with warm restarts)
- **Save/load**: `save_model` saves `.pth` (model + optimizer + scheduler), `_target.pth`, `_params.txt`, `_memory.pkl`

### State representations (`state_creator.py`)

Two modes controlled by `ConfigEntorno.TIPO_ESTADO`:

- **`'ohlc'`**: `create_all_states_ohcl` — OHLCV normalized (StandardScaler) × window_size + hour sin/cos. State size = `window_size * 5 + 2 + 3`
- **`'advanced'`**: `create_all_states_advanced` — OHLCV × window_size + RSI(14) + MACD line/signal/histogram + day sin/cos + hour sin/cos. State size = `window_size * 5 + 8 + 3`

Both use vectorized `sliding_window_view` for performance instead of per-timestep loops.

### Reward system (`AdvancedRewardSystem.py`)

Multi-component reward calculated only on trade close:
- **profit**: basado en expectancy de los últimos 50 wins/losses — `tanh(expectancy / (balance * 0.01))`. Si `avg_loss > avg_win`, es negativo aunque la accuracy sea alta.
- **sharpe**: rolling Sharpe sobre los últimos 50 trades cerrados
- **drawdown**: penalización cuando el drawdown de equity supera 5%
- **consistency**, **risk_adjusted**, **momentum**, **trade_quality**: componentes opcionales configurables desde `ConfigReward`

Between trade events, a small step reward based on unrealized P&L (`tanh(upnl / 2% balance) * 0.01`) reduces reward sparsity.

#### Interpretación del reward total negativo durante entrenamiento

Es **normal y esperado** que el reward acumulado del episodio sea negativo, especialmente en episodios tempranos. Lo que importa es la tendencia entre componentes:

- **Reward negativo con P&L negativo = señal correcta.** El sistema detecta el problema real: el agente puede ganar en accuracy (más trades positivos) pero si los trades perdedores son más grandes en dólares que los ganadores, el resultado neto es pérdida. Eso es el error clásico de trading — *dejar correr los losers y cortar los winners rápido*. La recompensa negativa es la presión correcta para que el agente aprenda a recortar pérdidas.

- **Señales de aprendizaje real a observar** (independientemente del reward total):
  - `trade_quality` y `risk_adjusted` subiendo → el agente mejora la relación ganancia/riesgo por trade
  - Accuracy subiendo con P&L que también mejora → el agente está aprendiendo bien
  - Accuracy subiendo pero P&L sigue negativo → el agente gana más seguido pero pierde más por trade (problema de gestión de pérdidas, no de dirección)

- **El reward total acumula sobre todos los trades del episodio**, por eso los números crecen en magnitud con más trades. Lo relevante no es el valor absoluto sino la tendencia episodio a episodio.

### Configuration (`parametros.py`)

**All hyperparameters live here.** The key classes:

| Class | Purpose |
|-------|---------|
| `ConfigEntorno` | CSV file, symbol, window size, state type, train/test split |
| `ConfigTrading` | Balance, lot size, commission |
| `ConfigAgente` | Gamma, epsilon decay, learning rate, Double DQN toggle |
| `ConfigMemoria` | PER memory size, alpha, beta |
| `ConfigScheduler` | LR scheduler type and params |
| `ConfigReward` | Reward component weights |
| `ConfigEntrenamiento` | Episodes, folds, batch size, save frequency |
| `ConfigModelo` | Model save/load paths |
| `ConfigBackend` | Backend URL, debug prints, LivePlot config |

Use `imprimir_config()` to print current configuration.

### Data format

Tab-separated CSV files exported from MetaTrader 5, stored in `data/`. Required columns: `<DATE>`, `<TIME>`, `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`, `<TICKVOL>`, `<SPREAD>`. The loader strips `<>` from headers and lowercases them.

### Output

Models and plots are saved to `resultados_cv/`:
- `model_<csv_name>.pth` — main model checkpoint
- `model_<csv_name>_target.pth` — target network
- `model_<csv_name>_memory.pkl` — PER memory buffer
- `training_metrics_fold_N.png` — per-fold training metrics (10 subplots)
- `trading_session_*.png` — buy/sell points plotted on price chart

---

## Historial de cambios

> **INSTRUCCIÓN PARA CLAUDE:** Cada vez que se realice un cambio en el código, debe agregarse una entrada en esta sección con fecha, archivo(s) modificado(s) y descripción del cambio. Esto es obligatorio, no opcional.



### 2026-03-30

**Fix: `reset_episode()` movido al inicio de cada episodio (`run_con_per.py`)**
- `reward_system.reset_episode()` se llamaba al **final** del episodio (línea ~670), lo que limpiaba los buffers después de que el episodio ya terminó — completamente inútil.
- Movido al **inicio** de cada episodio, justo después de resetear los `sumaRecompensa*`.
- Efecto: `wins_buffer`, `losses_buffer`, `returns_buffer`, `equity_buffer` y `drawdown_buffer` ahora contienen solo trades del episodio actual. Anteriormente acumulaban historia de episodios y folds anteriores, contaminando todos los componentes de reward (profit, sharpe, consistency, risk_adjusted, momentum, trade_quality).
- El `reset_episode()` del final fue eliminado (redundante y destructivo — tiraba datos válidos recién generados).
- Impacto en entrenamiento: los primeros ~50 trades de cada episodio tienen buffers en "warm-up" (algunos componentes retornan 0 por debajo del mínimo de datos). Con ~2000+ trades por episodio esto representa el 2.5% inicial — el resto del episodio recibe señal limpia del episodio actual.

**Ampliar parámetros guardados en checkpoint (`dueling_dqn_con_per.py`, `parametros.py`)**
- `__init__` de `AI_Trader_per`: agregados parámetros `batch_size`, `train_frequency`, `train_iterations`, `reward_weights` con sus respectivos `self.*`
- `save_model`: ahora guarda en `_params.txt` los campos `gamma`, `target_model_update`, `memory_size`, `epsilon_final`, `epsilon_decay`, `batch_size`, `train_frequency`, `train_iterations`, y cada peso de reward como `reward_weight_<nombre>`
- `load_model`: parsea y restaura todos los campos nuevos; los `reward_weights` se reconstruyen iterando las líneas con prefijo `reward_weight_`
- `get_config_trader()` en `parametros.py`: ahora pasa `batch_size`, `train_frequency`, `train_iterations` y `reward_weights` al constructor

**Documentación detallada de parámetros (`parametros.py`)**
- Cada parámetro tiene comentario explicando qué hace, qué pasa al subirlo/bajarlo, y por qué el valor actual es el correcto para este modelo (GOLD M15, Dueling DQN + PER)
- Se incorporó contexto del historial de entrenamiento (episodios catástrofe, bug 50/50, saturación de trade_quality) directamente en los comentarios relevantes

### 2026-03-17

**Eliminación de Dropbox (`run_con_per.py`, `parametros.py`)**
- Quitado `import dropbox`
- Quitados `DROPBOX_ACCESS_TOKEN` y `dbx = dropbox.Dropbox(...)` que crasheaban al iniciar aunque `GUARDAR_EN_DROPBOX = False`
- Quitada variable `guardar_en_dropbox` y el bloque de upload completo
- Quitado `GUARDAR_EN_DROPBOX` de `ConfigBackend` en `parametros.py`

**Bug `best_high` para shorts (`run_con_per.py`)**
- Corregida condición de tracking: `len(trader.inventory)` → `len(trader.inventory_sell)`
- `best_high` ahora se trackea y resetea correctamente según si hay posición short abierta
- Afectaba al cálculo de `pip_drawdrow_real` para operaciones short (drawdown real incorrecto)

### 2026-03-18 (6)

**Performance: vectorizar RSI loop Python → ewm (`indicadores.py`, `backtesting.py`)**
- `indicadores.py`: eliminado `for i in range(period, len(data))` con `.iloc` que iteraba 14K+ filas en Python puro
- Reemplazado por `up.ewm(alpha=1/period, adjust=False).mean()` — equivalente matemático al suavizado de Wilder, totalmente vectorizado en C (pandas)
- `backtesting.py`: misma corrección aplicada a su copia local de `rsi()`
- `live_trading.py`: ya importa `rsi` de `indicadores.py`, cubierto automáticamente
- Impacto estimado: ~500-1000ms ahorrados por fold × 4 folds

### 2026-03-18 (5)

**Fix: drawdown_real roto → MAE (Maximum Adverse Excursion) (`run_con_per.py`)**
- `current_drawdown_real` acumulaba `profit_drawdrow_real` de todos los trades, dando valores >100% imposibles (ej: 643%)
- Eliminadas variables: `current_drawdown_real`, `peak_equity_drawdrown_real`, `drawdown_real_history_episode`, `max_drawdown_real`
- Reemplazado por `worst_mae_pips` / `worst_mae_usd`: el peor precio no realizado alcanzado en cualquier trade del episodio (MAE real)
- El print ahora muestra `Peor MAE (pips)` y su equivalente en USD

### 2026-03-18 (4)

**Mejora print de episodio + contador force_closes (`run_con_per.py`)**
- Agregado `force_closes = 0` en inicialización de episodio; se incrementa en cada cierre forzado a hora 23
- Print reorganizado en secciones: `[P&L]`, `[TRADES]`, `[RIESGO]`, `[AGENTE]`, `[REWARD BREAKDOWN]`
- Nuevas métricas mostradas: `losses`, `avg_win`, `avg_loss`, `profit_factor`, `expectancy`, `force_closes`, `max_drawdown_real`, `trader.epsilon`, `trader.learning_rate`, `current_loss`
- `expectancy` y `profit_factor` se calculan inline antes del print

### 2026-03-18 (3)

**Fix: `torch.ger()` deprecated → `torch.outer()` (`NoisyLinear.py`)**
- Línea 38: `epsilon_out.ger(epsilon_in)` reemplazado por `torch.outer(epsilon_out, epsilon_in)`
- `torch.ger()` fue deprecado en PyTorch 1.9 y puede lanzar `AttributeError` en versiones modernas
- Funcionalidad idéntica

### 2026-03-18 (2)

**Eliminación de código muerto**
- `state_creator.py`: eliminada `state_creator_vectorized` (función per-timestep con RSI/MACD/EMA200, reemplazada por `create_all_states_advanced`)
- `AdvancedRewardSystem.py`: eliminado método `get_adaptive_weights` (llamada estaba comentada; lógica con `elif % 400` nunca se ejecutaba)
- `run_con_per.py`: quitado `state_creator_vectorized` del import

### 2026-03-18

**Fix: parámetros faltantes en `get_config_trader()` (`parametros.py`)**
- `get_config_trader()` no pasaba varios parámetros al constructor de `AI_Trader_per`, que usaba sus defaults hardcodeados ignorando `parametros.py`.
- Parámetros añadidos: `epsilon_final` (0.15 default → 0.05 real), `cosine_restarts` (True default → False real), `epsilon`, `use_double_dqn`, `patience`, `factor`.
- `epsilon_final` hacía que el decay se detuviera en 0.15 en lugar de 0.05.
- `cosine_restarts` hacía que el fix de warm restarts nunca llegara al scheduler real.

**Fix: `PESO_TRADE_QUALITY` reducido de 0.25 a 0.03 (`parametros.py`)**
- `_calculate_trade_quality_reward` divide `profit_dollars` entre `abs(avg_recent) * balance` donde `avg_recent` viene del `returns_buffer` (valores en escala `profit/balance`, ej. 0.0003). El denominador resulta ~$0.30, haciendo que el tanh sature en ±0.95 en casi todos los trades.
- Con 67% accuracy y ~3000 trades, la suma acumulada era siempre ~+200 independientemente del P&L real del episodio. Dominaba el total y daba reward positivo con equity negativo.
- Reducir a 0.03 baja su contribución de ~200 a ~24, permitiendo que `profit` (con accuracy real, ya corregido) y `drawdown` dominen correctamente.
- **Pendiente futuro:** reformular `_calculate_trade_quality_reward` para que use magnitud real (avg_win vs avg_loss de los buffers) en lugar del avg_recent normalizado. El peso podrá subir una vez que la fórmula no sature.

### 2026-03-17 (3)

**Fix: bug accuracy 50/50 en profit reward (`AdvancedRewardSystem.py`, `run_con_per.py`)**
- `wins_buffer` y `losses_buffer` tienen `maxlen=50`. Cuando ambos están llenos, `accuracy = 50/100 = 0.5` siempre, independientemente de la accuracy real del episodio (que llegó a 69%).
- `calculate_reward` ahora acepta `episode_wins` y `episode_losses` opcionales. Cuando se pasan, usa esos valores para calcular la accuracy real; `avg_win` y `avg_loss` siguen viniendo del buffer (son estimaciones de magnitud válidas).
- `calculate_advanced_reward` acepta y propaga los nuevos parámetros.
- Los 3 call sites en `run_con_per.py` (cierre long, cierre short, force-close hora 23) ahora pasan `episode_wins=wins + (1 if profit_pips > 0 else 0)` y `episode_losses=losses + (1 if profit_pips <= 0 else 0)` — el +1 es porque `wins`/`losses` se incrementan DESPUÉS del call en el código original.
- Efecto esperado: el componente `profit` ahora verá la accuracy real del episodio (66-69%) en lugar de 50% fijo. El reward de profit pasará a ser positivo cuando la accuracy real supere el breakeven de la expectancy.

### 2026-03-17 (2)

**Fix: pesos de reward ignorados (`AdvancedRewardSystem.py`, `run_con_per.py`)**
- `AdvancedRewardSystem.__init__` ahora acepta parámetro opcional `weights`; si no se pasa, usa los defaults anteriores
- Las dos instanciaciones en `run_con_per.py` ahora pasan `weights=ConfigReward.get_pesos()`, resolviendo el desacople silencioso entre `parametros.py` y la clase de recompensa

### Observaciones del entrenamiento GOLD# M15 — fold 1 desde cero (2026-03-30)

**Modelo:** `GOLD#_M15_202112200200_202412311530.csv` | arranque desde cero (sin checkpoint) | `cosine_decay` con `COSINE_RESTARTS=False`, `LR_DECAY_STEPS=1000`

**Contexto:** el checkpoint anterior no estaba disponible al iniciar. El agente arrancó completamente desde cero, sin conocimiento previo.

**Comportamiento del LR (cosine_decay confirmado):**
El LR oscila con período ~13 episodios medido al final de cada episodio (ciclo completo = 2×T_max = 2000 optimizer steps; ~2148 steps/episodio → 1.074 ciclos por episodio).
- Máximo (~0.001) en ep 13-14, ep 40
- Mínimo (~0.000011) en ep 20, ep 33
- LR visible al final del episodio sigue la envolvente coseno con ese período

**Progresión de métricas clave (fold 1, eps 1-40):**

| Ep | P&L (pips) | Accuracy | Avg Loss | Expectancy | LR |
|----|-----------|----------|----------|------------|----|
| 1  | -318 | 47.5% | -2.30 | -0.11 | 0.000970 |
| 12 | -109 | 50.7% | -2.46 | -0.03 | 0.000879 |
| 20 | -500 | 50.8% | -2.61 | -0.15 | 0.000011 ← mínimo |
| 31 | -125 | 54.4% | -2.72 | -0.04 | 0.000291 |
| 33 | -149 | 55.2% | -2.81 | -0.05 | 0.000018 ← mínimo |
| **39** | **+62** | **56.7%** | **-2.78** | **+0.02** | 0.000946 |
| 40 | -37  | 57.8% | -2.99 | -0.01 | **0.001000 ← máximo** |

**Episodio 39 — primer breakthrough del fold:**
- Primer P&L positivo (+62 pips), primer Sharpe positivo (+0.005), primer expectancy positiva (+0.02), primer reward de episodio positivo (+32.42), drawdown mínimo del fold (10.97%).
- Ocurrió en fold 1 ep 39 partiendo de cero — equivalente al fold 2 ep 37 del entrenamiento anterior (que tenía checkpoint). El agente aprendió más rápido de lo esperado.

**Episodio 40 — destrucción inmediata por LR máximo:**
- LR llegó a 0.001000 (máximo absoluto del ciclo) justo después del breakthrough.
- avg_loss saltó de 2.78 a 2.99 (peor del fold), P&L volvió a negativo (-37 pips).
- Mismo mecanismo que fold 3 del entrenamiento anterior: el LR alto hace updates grandes que destruyen la política de salida. La accuracy siguió subiendo (57.82%) pero el timing de exit se corrompió.

**Patrón estructural confirmado — accuray sube, avg_loss también:**
- Accuracy: 47.5% (ep1) → 57.8% (ep40). Mejora consistente.
- Avg_loss: -2.30 (ep1) → -2.99 (ep40). Crece junto con la accuracy.
- El agente aprende dirección (entrada) pero deja correr los losers — espera que se recuperen en lugar de cortarlos. La accuracy mejora porque entra en mejores momentos, pero eso no compensa el aumento del avg_loss.
- `risk_adjusted` y `trade_quality` son los únicos componentes consistentemente positivos (igual que en entrenamiento anterior), y son los que están guiando el aprendizaje.

**Conclusión del fold 1:**
El agente está aprendiendo, y más rápido que en el entrenamiento anterior dado que partió de cero. El problema pendiente es el LR oscilante que genera episodios catástrofe cada vez que alcanza el máximo. Cambiar a `reduce_on_plateau` antes del fold 2.

**Cambio pendiente para fold 2+:**
```python
SCHEDULER_TYPE = 'reduce_on_plateau'
CARGAR_MODELO = True   # cargar checkpoint de ep50 del fold 1
```

**Descubrimiento — avg_loss crece con la accuracy:**
Patrón nuevo no observado en el entrenamiento anterior (que arrancó con checkpoint). A medida que el agente mejora la entrada, tolera pérdidas más grandes esperando reversión. Esto refuerza la necesidad del reward de costo de oportunidad (lookahead 10-15 velas en losers) para señalizar que el costo de esperar en una pérdida es alto.

---

### Observaciones del entrenamiento GOLD# M15 (2026-03-17)

**Modelo:** `GOLD#_M15_202112200200_202412311530.csv` | 4 folds × 50 episodios

El agente mostró una convergencia clara a lo largo del entrenamiento:
- **Fold 1**: accuracy subió de 50% a 61%, trades ~3600-3700, P&L mayormente negativo (peor episodio: -490 pips).
- **Fold 2 (eps 1-36)**: arrancó con 60.7% de accuracy (transfirió lo aprendido), trades bajaron progresivamente de ~3600 a 3006 sin ninguna penalización explícita. Aparecieron los primeros episodios rentables (+13, +26 pips) y la accuracy alcanzó 66%.
- **Fold 2, Episodio 37 — hito del entrenamiento**: trades bajaron a **2987** (primera vez bajo 3000), P&L **+260 pips / +$260**, Sharpe **+0.02** (primer Sharpe positivo), Drawdown solo **10.29%**, reward total del episodio **+100.41** (primer reward fuertemente positivo). El agente cruzó tres umbrales a la vez: selectividad, rentabilidad y gestión de riesgo.
- **Fold 2, Episodios 38-40 — confirmación de aprendizaje real**: la accuracy se mantuvo en 66-67% en los tres episodios siguientes al hito (no cayó), confirmando que el ep 37 fue aprendizaje y no suerte. Los episodios negativos post-37 fueron solo -97, -119 y -15 pips — el piso mejoró permanentemente respecto a los -300/-400 de episodios anteriores. Ep 40: -15 pips con 67.10% accuracy, casi breakeven.

**Conclusión clave:** el agente aprendió solo a ser más selectivo — menos trades + mayor accuracy = mejores resultados. La correlación fue consistente y se confirmó sostenida: la accuracy no retrocedió tras el hito, señal de que el cambio de política fue durable. Sin ninguna penalización explícita por overtrading.

**Problema pendiente que limitó la convergencia:** el bug de accuracy en `calculate_reward` (buffers siempre 50/50) hace que el componente de profit no vea la mejora real de accuracy. A pesar de eso, `trade_quality` y `risk_adjusted` compensaron y guiaron el aprendizaje correctamente.

**Fold 3 (eps 1-10):** arrancó con trades en rango 2491-3142 y accuracy 65-68% — mejor punto de partida que fold 2. Primer episodio rentable en ep 8 (+15 pips) vs ep 11 en fold 2. Accuracy alcanzó 68.33% (nuevo máximo). Alta varianza en primeros episodios (-324, -376) es normal al adaptarse a nuevos datos.
- **Fold 3, Episodio 16:** +37.97 pips, reward **+8.13** (positivo), drawdown solo **13.34%**, 2234 trades, 67.41% accuracy. El breakthrough llegó en ep 16 vs ep 37 en fold 2 y ep 8 en fold 3 — la convergencia está acelerando entre folds. El agente encontró el equilibrio entre selectividad y volumen de trades rentables.

**Fold 3 completo (eps 17-44, entrenamiento detenido en ep 44):**

Resumen de episodios clave:

| Ep | P&L (pips) | Trades | Acc% | DD% | Notas |
|----|-----------|--------|------|-----|-------|
| 16 | +37.97 | 2234 | 67.41 | 13.34 | Breakthrough |
| 19 | -32.83 | 2309 | 68.77 | 13.33 | — |
| 21 | +7.32 | 2413 | 68.88 | 10.86 | — |
| 23 | +0.76 | 2427 | 68.27 | 18.47 | — |
| 25 | -16.15 | 2280 | 67.81 | 9.35 | Casi breakeven |
| **26** | **-288.66** | 2064 | 67.30 | **32.13** | **1er catástrofe** |
| 27 | +4.99 | 2234 | 68.62 | 10.22 | Recuperación rápida |
| **31** | **-247.63** | 2221 | 66.68 | **29.74** | **2da catástrofe** |
| **32** | **-234.15** | 2366 | 67.24 | **27.50** | **Catástrofe consecutiva** |
| **36** | **-233.61** | 2062 | 67.85 | **30.64** | **3era catástrofe** |
| **37** | **-296.65** | 2192 | 65.74 | **34.04** | **Peor del fold** |
| 38 | -6.79 | 2291 | 69.27 | 13.65 | Recuperación (accuracy máxima: 69.27%) |
| **40** | **-308.84** | 2205 | 67.76 | **32.83** | **Peor absoluto de fold 3** |
| **41** | **+128.15** | **2017** | 67.77 | **7.61** | **Mejor del fold — mínimo trades y DD** |
| 42 | -141.09 | 2170 | 67.33 | 25.14 | Recaída inmediata post-ep41 |
| 43 | -178.62 | 2005 | 65.79 | 20.98 | — |
| 44 | -41.89 | 2138 | 65.95 | 20.55 | Entrenamiento detenido |

**Patrón de catástrofes confirmado — warm restarts del LR:**
Los episodios con pérdidas mayores a -230 pips (eps 26, 31, 32, 36, 37, 40) aparecen cada ~5-6 episodios, exactamente el período de `T_0=LR_DECAY_STEPS` del `CosineAnnealingWarmRestarts`. Cada vez que el LR se resetea a su valor inicial, el optimizador hace updates grandes que destruyen los pesos aprendidos. La política de trading se corrompe y el agente pierde todo lo aprendido en ese mini-ciclo. El conocimiento subyacente no desaparece del todo (el agente se recupera en 1-2 episodios), pero el daño de P&L en los episodios catástrofe es severo.

**Episodio 41 — mejor del fold pero NO una consolidación:**
+128.15 pips, 2017 trades (mínimo del fold), drawdown 7.61% (mínimo absoluto), Sharpe +0.02. Sin embargo, el ep 42 cayó de inmediato a -141 pips. Ep 41 coincidió con el LR en su punto mínimo del ciclo cosine — la política fue óptima exactamente en ese momento. No fue consolidación, fue un episodio afortunado de timing dentro del ciclo destructivo. Con `COSINE_RESTARTS = False`, este nivel de performance podría sostenerse.

**Accuracy durante catástrofes:** en los peores episodios la accuracy se mantuvo en 65-68% — nunca cayó por debajo de 65.74%. Confirma que la habilidad de dirección (entrada) es robusta y no se destruye con los warm restarts. Lo que se destruye es la gestión de salida y el sizing de pérdidas.

**Tendencia de trades en fold 3:** ep 1-10: ~2491-3142 → ep 19-30: ~2064-2516 → ep 36-44: ~2005-2366. El agente continuó reduciendo trades durante todo el fold, incluyendo en los episodios post-catástrofe. La selectividad aprendida sobrevive a los warm restarts, aunque el P&L no.

**Conclusión — el agente separó dos habilidades distintas:**
- **Habilidad 1 (entrada/dirección):** la accuracy se mantiene en 65-69% independientemente del número de trades o de los warm restarts. La señal de entrada es robusta y estable — no se destruye con LR alto.
- **Habilidad 2 (salida/gestión):** el P&L varía enormemente entre episodios con accuracy similar. La diferencia no está en cuándo entra sino en cuándo y cómo sale. Esta habilidad es frágil y se destruye cada vez que el LR se resetea. El agente sabe hacia dónde va el mercado pero no consolida cuándo cerrar.
- **Implicación directa:** el problema que queda es de timing de salida + estabilidad del LR. Ambos apuntan al mismo fix: `COSINE_RESTARTS = False` + reward de costo de oportunidad.

**Descubrimiento importante — penalización por overtrading NO necesaria:** el agente redujo trades de ~3600 (fold 1) → ~2900 (fold 2) → ~2005-2500 (fold 3) sin ninguna regla explícita. Lo aprendió solo a través de `trade_quality` y `risk_adjusted`. **No implementar penalización por cantidad de trades.**

**Rol real de cada componente de reward en este entrenamiento:**
- `profit` (-144 a -249): siempre negativo, nunca dio señal positiva. Bloqueado por el bug 50/50.
- `drawdown` (-3 a -106): el componente más destructivo en episodios catástrofe; acumula por cada trade con DD >5%.
- `trade_quality` (+97 a +170) y `risk_adjusted` (+36 a +72): los únicos que guiaron el aprendizaje real. Sin ellos el agente no hubiera aprendido nada.
- `sharpe`, `consistencia`, `momentum`: señales de soporte, oscilan según el episodio.

**Cambios obligatorios para el próximo entrenamiento:**
1. **`COSINE_RESTARTS = False`** — elimina los episodios catástrofe que destruyen la política cada ~5-6 eps
2. **Bug accuracy profit reward** — pasar `wins`/`losses` reales del episodio a `calculate_reward`; el componente más pesado lleva todo el fold ciego
3. **Actualizar `MODELO_EXISTENTE`** en `ConfigModelo` al nuevo checkpoint

**Cambios recomendados:**
4. Reward de costo de oportunidad (lookahead 10-15 velas en winners) — ataca el timing de salida
5. `reset_episode()` al inicio de cada episodio para no contaminar buffers entre episodios

---

## Nota del proyecto — 2026-03-30

**Nota actual: 5.5/7**

El proyecto tiene base técnica sólida (Dueling DQN + PER + Double DQN, walk-forward CV, reward multi-componente, configuración centralizada, historial documentado). El agente aprende dirección con accuracy real (55-58%). No es un proyecto de juguete.

**Lo que falta para llegar a 7/7:**

1. **Agente rentable en producción** — el breakthrough ocurrió en fold 1 ep39 pero no se sostuvo. Falta confirmar rentabilidad consistente a lo largo de múltiples folds y en el set de test final.

2. **Reward de costo de oportunidad** — el agente no recibe señal cuando cierra una posición perdedora tarde (dejó correr el loser). El avg_loss > avg_win es el problema central no resuelto. Sin esta señal el agente no puede aprender a cortar pérdidas.

3. ~~**`reset_episode()` sin conectar**~~ — **RESUELTO (2026-03-30)**.

4. **50 episodios por fold insuficiente** — el agente seguía convergiendo al ep 50 en runs anteriores. Con 100-150 episodios por fold habría tiempo de consolidar post-breakthrough.

---

## Posibles mejoras futuras

### Críticas
- ~~**Bug accuracy profit reward**~~ — **RESUELTO (2026-03-17)**. `calculate_reward` ahora acepta `episode_wins`/`episode_losses` y los usa para accuracy real en lugar de los buffers 50/50.
- **Reward de costo de oportunidad** — el agente no recibe señal de que cerró un winner antes de tiempo. Implementar lookahead de 10-15 velas al cerrar un trade ganador: si el precio siguió a favor, reward negativo proporcional a lo que dejó sobre la mesa. Solo aplicar en trades ganadores. Esto ataca directamente el patrón de "cortar winners rápido".

### Importantes
- **`COSINE_RESTARTS = False` (ya aplicado)** — cambiado en `parametros.py`. Elimina los saltos bruscos de LR que causaban episodios catástrofe (-250 a -308 pips) en fold 3. Usa `CosineAnnealingLR` en lugar de `CosineAnnealingWarmRestarts`. **Ojo:** `CosineAnnealingLR` con `T_max=1000` igual oscila — baja de LR_max a LR_min en 1000 pasos y luego sube de nuevo suavemente. La diferencia clave es que no hay saltos bruscos, todo es curva coseno suave. Si en el próximo entrenamiento sigue habiendo varianza alta entre episodios, la solución definitiva es poner `T_max` igual al total de pasos del entrenamiento completo (folds × episodios × ~600 optimizer steps/episodio) para que el LR solo baje y nunca suba.
- ~~**`reset_episode()` no se llama entre episodios**~~ — **RESUELTO (2026-03-30)**. Movido al inicio de cada episodio en `run_con_per.py`. Los buffers ahora se limpian correctamente al comenzar cada episodio.
- **Aumentar episodios por fold** — 50 episodios es insuficiente; el agente seguía convergiendo al ep 50 del fold 2. Considerar 100-150 episodios por fold.
- **Drawdown penalty acumula por trade** — con 2500 trades por episodio y drawdown persistente >5%, la penalización acumulada domina el reward total. Considerar calcularla una vez por episodio o normalizarla por número de trades.

### Menores
- **Force-close a hora 23** — heurística arbitraria. Con un agente bien entrenado y gestión de riesgo propia podría no ser necesaria. Evaluar eliminarla en etapas avanzadas.
- **Más folds** — con 4 folds sobre 3 años de datos M15, cada fold tiene ~14.000 velas. Aumentar a 6-8 folds daría más diversidad de condiciones de mercado y validación más robusta.
- **Código muerto** — ~~`state_creator_vectorized`~~ y ~~`get_adaptive_weights`~~ ya eliminados. `state_creator_ohcl_vectorized` se mantiene (aún referenciada en imports).
- **No agregar penalización por cantidad de trades** — el agente redujo trades de ~3600 a ~2100 solo, sin ninguna regla explícita. Una penalización artificial limitaría estrategias ganadoras no anticipadas.

---

### Problemas conocidos pendientes
- ~~`state_creator_vectorized`~~ — eliminado (2026-03-18)
- ~~`get_adaptive_weights`~~ — eliminado (2026-03-18)
- Epsilon con valor 1.0 inicial es redundante dado que NoisyLinear ya provee exploración intrínseca
- `backtesting.py` tiene CSV path hardcodeado en `main()` — actualizar manualmente antes de correr
- `MODELO_EXISTENTE` en `ConfigModelo` — actualizado a `resultados_cv/model_GOLD#_M15_202112200200_202412311530.csv` (2026-03-17)
