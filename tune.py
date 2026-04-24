import os
import optuna
import lightning as L
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from models.itransformer import iTransformer
from data_utils import load_and_preprocess, split_data
from sklearn.metrics import mean_absolute_error, mean_squared_error
import wandb
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)
DB_PATH = 'sqlite:///optuna_trials.db'
DATASETS = ['ETTm1','ETTm2','ETTh1','ETTh2']
DATA_PATH = './ETTl/'
BATCH_SIZE = 32
SEQ_LEN = 96
PRED_LEN = 24
N_TRIALS = 10

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len, pred_len):
        self.data = df[feature_cols].values
        self.target = df[target_col].values
        self.seq_len = seq_len
        self.pred_len = pred_len
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.target[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class ETTDataModule(L.LightningDataModule):
    def __init__(self, df, feature_cols, target_col, batch_size, seq_len, pred_len):
        super().__init__()
        train_df, valid_df, test_df = split_data(df)
        self.train_ds = TimeSeriesDataset(train_df, feature_cols, target_col, seq_len, pred_len)
        self.valid_ds = TimeSeriesDataset(valid_df, feature_cols, target_col, seq_len, pred_len)
        self.batch_size = batch_size
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size)

def objective(trial: optuna.Trial, df, feature_cols, target_col):
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dm = ETTDataModule(df, feature_cols, target_col, BATCH_SIZE, SEQ_LEN, PRED_LEN)
    model = iTransformer(len(feature_cols), d_model, nhead, num_layers, dropout, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 训练
    for epoch in range(50):
        model.train()
        for x, y in dm.train_dataloader():
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        # 每5个epoch验证一次
        if (epoch+1) % 5 == 0:
            model.eval()
            val_losses, val_maes, val_mses = [], [], []
            with torch.no_grad():
                for x, y in dm.val_dataloader():
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    val_loss = criterion(y_hat, y).item()
                    val_mae = mean_absolute_error(y.cpu().numpy(), y_hat.cpu().numpy())
                    val_mse = mean_squared_error(y.cpu().numpy(), y_hat.cpu().numpy())
                    val_losses.append(val_loss)
                    val_maes.append(val_mae)
                    val_mses.append(val_mse)
            mean_val_loss = np.mean(val_losses)
            trial.report(mean_val_loss, epoch)
#            if trial.should_prune():
#                raise optuna.TrialPruned()
    return mean_val_loss

def run_optuna():
    for dataset in DATASETS:
        file_path = os.path.join(DATA_PATH, f'{dataset}.csv')
        df = load_and_preprocess(file_path)
        feature_cols = [col for col in df.columns if col not in ['date', 'target']]
        target_col = 'target' if 'target' in df.columns else df.columns[-1]
        db_path = f"sqlite:///{dataset}.db"
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(),
            storage=db_path,
            study_name=f'{dataset}_study',
            load_if_exists=True
        )
        remain_trials = N_TRIALS - len(study.trials)
        if remain_trials > 0:
            study.optimize(lambda trial: objective(trial, df, feature_cols, target_col), n_trials=remain_trials)
        print(f'{dataset} 最优参数: {study.best_params}')
        # 可视化
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f'{dataset}_parallel_coordinate.png')
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(f'{dataset}_param_importance.png')

if __name__ == '__main__':
    run_optuna()
    print('所有数据集调参完成，可用 optuna-dashboard sqlite:///optuna_trials.db 查看过程。')
