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

A `.env` file is required at the project root with:
- `ACCESS_TOKEN_DROPBOX` — Dropbox token for optional model uploads

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
- **profit**: `tanh(profit_dollars)`
- **sharpe**: rolling Sharpe over last 50 trades
- **drawdown**: penalty when equity drawdown exceeds 5%
- **consistency**, **risk_adjusted**, **momentum**, **trade_quality**: optional components (weights set to 0 by default)

Between trade events, a small step reward based on unrealized P&L (`tanh(upnl / 2% balance) * 0.01`) reduces reward sparsity.

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
| `ConfigBackend` | Backend URL, Dropbox toggle, debug prints |

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
