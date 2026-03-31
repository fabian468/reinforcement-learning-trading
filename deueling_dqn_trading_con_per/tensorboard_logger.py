# -*- coding: utf-8 -*-
"""
TensorBoard logger para el sistema de trading RL.
Para desactivar: ConfigBackend.TENSORBOARD = False
Para eliminar: borrar este archivo y quitar las llamadas en run_con_per.py
"""

from torch.utils.tensorboard import SummaryWriter
import os


class TBLogger:
    def __init__(self, log_dir='runs'):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard activo → ejecuta: tensorboard --logdir={log_dir}")

    def log_episode(self, fold, episode, profit_usd, profit_pips, equity,
                    sharpe, drawdown, accuracy, trades, epsilon, reward_episode,
                    avg_win=0, avg_loss=0, expectancy=0, profit_factor=0,
                    worst_mae_pips=0, force_closes=0, learning_rate=0, loss_td=0,
                    reward_profit=0, reward_sharpe=0, reward_drawdown=0,
                    reward_consistency=0, reward_risk_adjusted=0,
                    reward_momentum=0, reward_trade_quality=0):
        """Métricas por episodio."""
        global_step = (fold - 1) * 10000 + episode  # step único por fold+episodio

        self.writer.add_scalar(f'Fold_{fold}/Profit_USD',      profit_usd,      global_step)
        self.writer.add_scalar(f'Fold_{fold}/Profit_Pips',     profit_pips,     global_step)
        self.writer.add_scalar(f'Fold_{fold}/Equity',          equity,          global_step)
        self.writer.add_scalar(f'Fold_{fold}/Sharpe',          sharpe,          global_step)
        self.writer.add_scalar(f'Fold_{fold}/Drawdown',        drawdown,        global_step)
        self.writer.add_scalar(f'Fold_{fold}/Accuracy',        accuracy,        global_step)
        self.writer.add_scalar(f'Fold_{fold}/Trades',          trades,          global_step)
        self.writer.add_scalar(f'Fold_{fold}/Epsilon',         epsilon,         global_step)
        self.writer.add_scalar(f'Fold_{fold}/Reward_Episodio', reward_episode,  global_step)
        self.writer.add_scalar(f'Fold_{fold}/Avg_Win_Pips',    avg_win,         global_step)
        self.writer.add_scalar(f'Fold_{fold}/Avg_Loss_Pips',   avg_loss,        global_step)
        self.writer.add_scalar(f'Fold_{fold}/Expectancy',      expectancy,      global_step)
        self.writer.add_scalar(f'Fold_{fold}/Profit_Factor',   profit_factor,   global_step)
        self.writer.add_scalar(f'Fold_{fold}/Worst_MAE_Pips',  worst_mae_pips,  global_step)
        self.writer.add_scalar(f'Fold_{fold}/Force_Closes',    force_closes,    global_step)
        self.writer.add_scalar(f'Fold_{fold}/Learning_Rate',   learning_rate,   global_step)
        self.writer.add_scalar(f'Fold_{fold}/Loss_TD',         loss_td,         global_step)

        self.writer.add_scalar(f'Fold_{fold}/Reward_Profit',        reward_profit,        global_step)
        self.writer.add_scalar(f'Fold_{fold}/Reward_Sharpe',        reward_sharpe,        global_step)
        self.writer.add_scalar(f'Fold_{fold}/Reward_Drawdown',      reward_drawdown,      global_step)
        self.writer.add_scalar(f'Fold_{fold}/Reward_Consistency',   reward_consistency,   global_step)
        self.writer.add_scalar(f'Fold_{fold}/Reward_RiskAdjusted',  reward_risk_adjusted, global_step)
        self.writer.add_scalar(f'Fold_{fold}/Reward_Momentum',      reward_momentum,      global_step)
        self.writer.add_scalar(f'Fold_{fold}/Reward_TradeQuality',  reward_trade_quality, global_step)

    def log_train_step(self, step, loss, learning_rate):
        """Métricas por batch_train (loss y LR)."""
        self.writer.add_scalar('Train/Loss',          loss,          step)
        self.writer.add_scalar('Train/Learning_Rate', learning_rate, step)

    def log_fold_summary(self, fold, profit_usd, profit_pips, sharpe,
                         drawdown, accuracy, avg_win, avg_loss, trades):
        """Resumen final de cada fold (para comparar folds entre sí)."""
        self.writer.add_scalar('Resumen_Folds/Profit_USD',  profit_usd,  fold)
        self.writer.add_scalar('Resumen_Folds/Profit_Pips', profit_pips, fold)
        self.writer.add_scalar('Resumen_Folds/Sharpe',      sharpe,      fold)
        self.writer.add_scalar('Resumen_Folds/Drawdown',    drawdown,    fold)
        self.writer.add_scalar('Resumen_Folds/Accuracy',    accuracy,    fold)
        self.writer.add_scalar('Resumen_Folds/Avg_Win',     avg_win,     fold)
        self.writer.add_scalar('Resumen_Folds/Avg_Loss',    avg_loss,    fold)
        self.writer.add_scalar('Resumen_Folds/Trades',      trades,      fold)

    def log_test_results(self, profit_usd, profit_pips, sharpe, drawdown,
                         accuracy, equity, trades):
        """Resultados del conjunto de prueba final."""
        self.writer.add_scalar('Test/Profit_USD',  profit_usd,  0)
        self.writer.add_scalar('Test/Profit_Pips', profit_pips, 0)
        self.writer.add_scalar('Test/Sharpe',      sharpe,      0)
        self.writer.add_scalar('Test/Drawdown',    drawdown,    0)
        self.writer.add_scalar('Test/Accuracy',    accuracy,    0)
        self.writer.add_scalar('Test/Equity',      equity,      0)
        self.writer.add_scalar('Test/Trades',      trades,      0)

    def close(self):
        self.writer.flush()
        self.writer.close()
