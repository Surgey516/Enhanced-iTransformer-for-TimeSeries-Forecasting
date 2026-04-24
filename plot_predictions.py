import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.itransformer import iTransformer
from data_utils import load_and_preprocess, split_data
from torch.utils.data import DataLoader, Dataset

DATASETS = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']
DATA_PATH = './ETTl/'
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

def generate_predictions(dataset_name):
    """
    为指定数据集生成预测结果
    """
    print(f'\n生成 {dataset_name} 预测结果...')
    
    # 加载数据
    file_path = os.path.join(DATA_PATH, f'{dataset_name}.csv')
    df = load_and_preprocess(file_path)
    feature_cols = [col for col in df.columns if col not in ['date', 'target']]
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    
    # 数据集划分
    _, _, test_df = split_data(df)
    test_ds = TimeSeriesDataset(test_df, feature_cols, target_col, SEQ_LEN, PRED_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 加载模型
    params = BEST_PARAMS[dataset_name]
    model = iTransformer(
        input_dim=len(feature_cols),
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        output_dim=1,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN
    )
    
    # 尝试加载训练好的模型权重
    checkpoint_path = f'lightning_logs/{dataset_name}/checkpoints/best.ckpt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f'成功加载权重: {checkpoint_path}')
    else:
        print(f'警告: 未找到权重文件 {checkpoint_path}，使用未训练模型')
    
    model.eval()
    
    # 生成预测
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            y_hat = model(x)
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # 拼接所有预测结果
    all_preds = np.concatenate(all_preds, axis=0)  # [N, pred_len]
    all_targets = np.concatenate(all_targets, axis=0)  # [N, pred_len]
    
    # 展平为一维序列（取前100个预测窗口用于绘图）
    num_plot = min(100, len(all_preds))
    pred_flat = []
    target_flat = []
    
    for i in range(num_plot):
        if i == 0:
            pred_flat.extend(all_preds[i])
            target_flat.extend(all_targets[i])
        else:
            # 只取每个窗口的最后一个点，避免重复
            pred_flat.append(all_preds[i][-1])
            target_flat.append(all_targets[i][-1])
    
    return np.array(pred_flat), np.array(target_flat)

def plot_comparison():
    """
    绘制四个数据集的预测对比图（2x2布局）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(DATASETS):
        print(f'\n处理 {dataset}...')
        
        # 生成预测
        preds, targets = generate_predictions(dataset)
        
        # 绘制对比曲线
        ax = axes[idx]
        time_steps = np.arange(len(targets))
        
        # 真实值（实线，黑色）
        ax.plot(time_steps, targets, label='Ground Truth', color='black', linewidth=2)
        
        # iTransformer 预测（虚线，蓝色）
        ax.plot(time_steps, preds, label='iTransformer', color='blue', linestyle='--', linewidth=1.5)
        
        # SARIMAX 和 Foundation Model 预测（需要队员提供数据）
        # 这里用随机数据作为占位符，实际使用时替换为真实预测
        sarimax_preds = targets + np.random.randn(len(targets)) * 0.1  # 占位符
        foundation_preds = targets + np.random.randn(len(targets)) * 0.08  # 占位符
        
        ax.plot(time_steps, sarimax_preds, label='SARIMAX', color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot(time_steps, foundation_preds, label='Foundation Model', color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 设置图表
        ax.set_title(f'{dataset} 预测对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('OT (油温)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Results/predictions_comparison.png', dpi=300, bbox_inches='tight')
    print('\n预测对比图已保存: Results/predictions_comparison.png')
    plt.show()

if __name__ == '__main__':
    # 创建输出目录
    os.makedirs('Results', exist_ok=True)
    
    # 绘制对比图
    plot_comparison()
    
    print('\n完成！')
    print('\n注意：')
    print('1. SARIMAX 和 Foundation Model 的预测曲线目前使用占位符数据')
    print('2. 请让队员提供真实预测结果，替换 plot_comparison() 函数中的占位符')
    print('3. 如需加载训练好的模型，请确保 lightning_logs/{dataset}/checkpoints/best.ckpt 存在')
