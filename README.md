# Enhanced iTransformer for Time Series Forecasting

基于 iTransformer（倒置 Transformer）的多变量时间序列预测，用于电力变压器油温 (OT) 预测。

## 目录结构

```
├── data/
│   ├── data_utils.py    # 数据加载、预处理、划分
│   └── ETTl/            # ETT 数据集 (CSV)
├── models/
│   └── itransformer.py  # iTransformer 模型定义
├── train.py             # 训练脚本 (PyTorch Lightning)
├── tune.py              # 超参数搜索 (Optuna)
├── plot_predictions.py  # 预测结果可视化
├── visualize.py         # 训练曲线 + SHAP 分析
├── visualize_optuna.py  # Optuna 调参可视化
├── requirements.txt
└── .gitignore
```

## 安装

```bash
pip install -r requirements.txt
wandb login
```

## 使用方法

```bash
# 训练
python train.py

# 超参数搜索
python tune.py

# 预测可视化
python plot_predictions.py

# 训练曲线 + SHAP 特征重要性
python visualize.py

# Optuna 可视化
python visualize_optuna.py
```

## 模型

iTransformer 将传统 Transformer 的注意力机制从**时间维度**翻转到**变量维度**，每个特征作为 token，捕捉多变量间的复杂关系。

- 输入: `[batch, 96 步, N 个特征]`
- 输出: `[batch, 24 步]`
- 训练/验证/测试: 60% / 20% / 20%

## 数据集

ETT (Electricity Transformer Temperature) 包含四个子集:
- **ETTh1/h2**: 小时级采样
- **ETTm1/m2**: 15 分钟级采样
- 特征: HUFL, HULL, MUFL, MULL, LUFL, LULL (负荷) + OT (油温)
