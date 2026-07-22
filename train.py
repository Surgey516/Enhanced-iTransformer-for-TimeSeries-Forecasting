import os
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from models.itransformer import iTransformer
from data.data_utils import load_and_preprocess, split_data, scale_data
from sklearn.metrics import mean_absolute_error, mean_squared_error
import wandb

L.seed_everything(42)

DATASETS = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']
DATA_PATH = './data/ETTl/'
BATCH_SIZE = 32
SEQ_LEN = 96
PRED_LEN = 24

# 最优超参数（来自 Optuna 调参结果）
BEST_PARAMS = {
    'ETTm1': {
        'd_model': 32,
        'nhead': 4,
        'num_layers': 4,
        'dropout': 0.052378253465880396,
        'lr': 0.001204170583167675
    },
    'ETTm2': {
        'd_model': 64,
        'nhead': 2,
        'num_layers': 4,
        'dropout': 0.05024491425799964,
        'lr': 0.00232937983871852
    },
    'ETTh1': {
        'd_model': 64,
        'nhead': 2,
        'num_layers': 3,
        'dropout': 0.05129616944847602,
        'lr': 0.0010274708369170011
    },
    'ETTh2': {
        'd_model': 128,
        'nhead': 2,
        'num_layers': 3,
        'dropout': 0.10974827437780539,
        'lr': 0.0001919114539435852
    }
}

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
        self.test_ds = TimeSeriesDataset(test_df, feature_cols, target_col, seq_len, pred_len)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

class LitiTransformer(L.LightningModule):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, lr, output_dim, seq_len, pred_len):
        super().__init__()
        self.save_hyperparameters()
        self.model = iTransformer(input_dim, d_model, nhead, num_layers, dropout, output_dim)
        self.criterion = torch.nn.MSELoss()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lr = lr
        self.train_mae = []
        self.train_mse = []
        self.valid_mae = []
        self.valid_mse = []
        self.train_loss = []
        self.valid_loss = []
        # 临时存储每个batch的指标
        self._train_epoch_losses = []
        self._train_epoch_mae = []
        self._train_epoch_mse = []
        self._valid_epoch_losses = []
        self._valid_epoch_mae = []
        self._valid_epoch_mse = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = mean_absolute_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())
        mse = mean_squared_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True)
        self.log('train_mse', mse, on_step=False, on_epoch=True)
        # 只在每个batch临时存储
        self._train_epoch_losses.append(loss.item())
        self._train_epoch_mae.append(mae)
        self._train_epoch_mse.append(mse)
        return loss

    def on_train_epoch_end(self):
        # 每个epoch结束时求均值并append到最终列表
        if self._train_epoch_losses:
            self.train_loss.append(np.mean(self._train_epoch_losses))
            self.train_mae.append(np.mean(self._train_epoch_mae))
            self.train_mse.append(np.mean(self._train_epoch_mse))
        # 清空临时列表
        self._train_epoch_losses.clear()
        self._train_epoch_mae.clear()
        self._train_epoch_mse.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = mean_absolute_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())
        mse = mean_squared_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        self.log('val_mse', mse, on_step=False, on_epoch=True)
        # 只在每个batch临时存储
        self._valid_epoch_losses.append(loss.item())
        self._valid_epoch_mae.append(mae)
        self._valid_epoch_mse.append(mse)
        return {'val_loss': loss, 'val_mae': mae, 'val_mse': mse}

    def on_validation_epoch_end(self):
        # 每个epoch结束时求均值并append到最终列表
        if self._valid_epoch_losses:
            self.valid_loss.append(np.mean(self._valid_epoch_losses))
            self.valid_mae.append(np.mean(self._valid_epoch_mae))
            self.valid_mse.append(np.mean(self._valid_epoch_mse))
        # 清空临时列表
        self._valid_epoch_losses.clear()
        self._valid_epoch_mae.clear()
        self._valid_epoch_mse.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = mean_absolute_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())
        mse = mean_squared_error(y.cpu().numpy(), y_hat.cpu().detach().numpy())
        mape = np.mean(np.abs((y.cpu().numpy() - y_hat.cpu().detach().numpy()) / (y.cpu().numpy() + 1e-8)))
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        self.log('test_mse', mse)
        self.log('test_mape', mape)
        return {'test_loss': loss, 'test_mae': mae, 'test_mse': mse, 'test_mape': mape}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == '__main__':
    for dataset in DATASETS:
        # 每个数据集单独 wandb.init，设置 run name
        wandb.init(project='iTransformer-ETT', name=dataset, config={'seed': 42}, reinit=True)
        file_path = os.path.join(DATA_PATH, f'{dataset}.csv')
        df = load_and_preprocess(file_path)
        feature_cols = [col for col in df.columns if col not in ['date', 'target']]
        target_col = 'target' if 'target' in df.columns else df.columns[-1]
        dm = ETTDataModule(df, feature_cols, target_col, BATCH_SIZE, SEQ_LEN, PRED_LEN)

        # 获取当前数据集的最优参数
        params = BEST_PARAMS[dataset]

        model = LitiTransformer(
            input_dim=len(feature_cols),
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            lr=params['lr'],
            output_dim=1,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN
        )
        # 设置 wandb logger，name=dataset，log_model=False
        wandb_logger = WandbLogger(project='iTransformer-ETT', name=dataset, log_model=False)
        trainer = L.Trainer(
            max_epochs=50,
            logger=wandb_logger,
            accelerator='gpu',
            check_val_every_n_epoch=5,
            enable_checkpointing=True,
            callbacks=[ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)]
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)
        # 保存loss/mae/mse到csv，补齐所有列表为相同长度
        max_len = max(len(model.train_loss), len(model.valid_loss), len(model.train_mae), len(model.valid_mae), len(model.train_mse), len(model.valid_mse))
        def pad(lst):
            return lst + [None] * (max_len - len(lst))
        hist = pd.DataFrame({
            'train_loss': pad(model.train_loss),
            'valid_loss': pad(model.valid_loss),
            'train_mae': pad(model.train_mae),
            'valid_mae': pad(model.valid_mae),
            'train_mse': pad(model.train_mse),
            'valid_mse': pad(model.valid_mse)
        })
        hist.to_csv(f'{dataset}_metrics.csv', index=False)
        print(f'{dataset} 训练完成，结果已保存。')
        wandb.finish()
