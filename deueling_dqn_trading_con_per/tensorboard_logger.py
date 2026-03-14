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
                    sharpe, drawdown, accuracy, trades, epsilon, reward_episode):
        """Métricas por episodio."""
        global_step = (fold - 1) * 10000 + episode  # step único por fold+episodio

        self.writer.add_scalar(f'Fold_{fold}/Profit_USD',     profit_usd,     global_step)
        self.writer.add_scalar(f'Fold_{fold}/Profit_Pips',    profit_pips,    global_step)
        self.writer.add_scalar(f'Fold_{fold}/Equity',         equity,         global_step)
        self.writer.add_scalar(f'Fold_{fold}/Sharpe',         sharpe,         global_step)
        self.writer.add_scalar(f'Fold_{fold}/Drawdown',       drawdown,       global_step)
        self.writer.add_scalar(f'Fold_{fold}/Accuracy',       accuracy,       global_step)
        self.writer.add_scalar(f'Fold_{fold}/Trades',         trades,         global_step)
        self.writer.add_scalar(f'Fold_{fold}/Epsilon',        epsilon,        global_step)
        self.writer.add_scalar(f'Fold_{fold}/Reward_Episodio', reward_episode, global_step)

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
