import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import numpy as np
from models.itransformer import iTransformer
from data_utils import load_and_preprocess, split_data

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

# 1. 训练/验证 loss、MAE、MSE 曲线可视化
def plot_metrics(metrics_csv, dataset):
    df = pd.read_csv(metrics_csv)
    plt.figure(figsize=(8,6))
    plt.plot(df['train_loss'], label='Train Loss', color='blue')
    plt.plot(df['valid_loss'], label='Valid Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{dataset} Loss Curve')
    plt.savefig(f'{dataset}_loss.png')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(df['train_mae'], label='Train MAE', color='blue')
    plt.plot(df['valid_mae'], label='Valid MAE', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title(f'{dataset} MAE Curve')
    plt.savefig(f'{dataset}_mae.png')
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(df['train_mse'], label='Train MSE', color='blue')
    plt.plot(df['valid_mse'], label='Valid MSE', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title(f'{dataset} MSE Curve')
    plt.savefig(f'{dataset}_mse.png')
    plt.close()

# 2. SHAP特征重要性分析
def plot_shap(model, df, feature_cols, dataset, seq_len):
    """
    对时序模型进行SHAP分析
    注意：由于iTransformer期望输入是[batch, seq_len, features]，
    我们需要构造合适的时序样本
    """
    model.eval()
    
    # 准备时序数据样本（取前100个时序窗口）
    num_samples = min(100, len(df) - seq_len)
    X_samples = []
    for i in range(num_samples):
        sample = df[feature_cols].values[i:i+seq_len]
        X_samples.append(sample)
    
    X_samples = np.array(X_samples)  # [num_samples, seq_len, features]
    X_tensor = torch.tensor(X_samples, dtype=torch.float32)
    
    # 使用GradientExplainer代替DeepExplainer（更适合时序模型）
    # 选择部分背景样本
    background = X_tensor[:20]
    test_samples = X_tensor[20:40]
    
    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_samples)
        # 对时序维度求平均，得到每个特征的重要性
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values_mean = np.mean(shap_values, axis=1)  # [samples, features]
        test_samples_mean = np.mean(test_samples.numpy(), axis=1)  # [samples, features]
        # shape校验：特征数量必须一致
        if shap_values_mean.shape[1] == len(feature_cols):
            shap.summary_plot(shap_values_mean, test_samples_mean, 
                             feature_names=feature_cols, show=False)
            plt.savefig(f'{dataset}_SHAP.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f'{dataset} SHAP分析完成')
        else:
            print(f'{dataset} SHAP shape不匹配，仅绘制分布直方图')
            plt.figure(figsize=(8,6))
            plt.hist(shap_values_mean.flatten(), bins=30)
            plt.title(f'{dataset} SHAP值分布')
            plt.savefig(f'{dataset}_SHAP_hist.png', dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f'{dataset} SHAP分析失败: {str(e)}')
        print(f'跳过{dataset}的SHAP分析')

if __name__ == '__main__':
    for dataset in DATASETS:
        print(f'\n处理 {dataset}...')
        
        # 1. 绘制训练指标曲线
        metrics_csv = f'{dataset}_metrics.csv'
        try:
            plot_metrics(metrics_csv, dataset)
            print(f'{dataset} 指标曲线绘制完成')
        except Exception as e:
            print(f'{dataset} 指标曲线绘制失败: {str(e)}')
        
        # 2. SHAP特征重要性分析
        try:
            file_path = f'{DATA_PATH}{dataset}.csv'
            df = load_and_preprocess(file_path)
            feature_cols = [col for col in df.columns if col not in ['date', 'target']]
            target_col = 'target' if 'target' in df.columns else df.columns[-1]
            
            # 使用最优参数初始化模型
            params = BEST_PARAMS[dataset]
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
            
            # 注意：这里使用的是未训练的模型进行SHAP分析
            # 如果需要加载训练好的模型权重，请在这里添加模型加载代码
            # 例如: model.load_state_dict(torch.load(f'{dataset}_best_model.pt'))
            
            model.eval()
            plot_shap(model, df, feature_cols, dataset, SEQ_LEN)
            
        except Exception as e:
            print(f'{dataset} SHAP分析失败: {str(e)}')
        
        print(f'{dataset} 处理完成\n')
