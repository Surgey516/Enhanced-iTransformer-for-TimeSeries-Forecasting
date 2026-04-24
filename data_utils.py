import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df = df.ffill()  # 缺失值前向填充
    df = add_time_features(df)
    return df

def split_data(df):
    train_clip = int(len(df) * 0.6)
    valid_clip = int(len(df) * 0.8)
    train_df = df[:train_clip]
    valid_df = df[train_clip:valid_clip]
    test_df = df[valid_clip:]
    return train_df, valid_df, test_df

def add_time_features(df):
    if 'date' in df.columns:
        time_col = pd.to_datetime(df['date'])
        df['month_sin'] = np.sin(2 * np.pi * time_col.dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * time_col.dt.month / 12)
        df['day_sin'] = np.sin(2 * np.pi * time_col.dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * time_col.dt.day / 31)
        df['hour_sin'] = np.sin(2 * np.pi * time_col.dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * time_col.dt.hour / 24)
    return df

def scale_data(train, valid, test, feature_cols):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[feature_cols])
    valid_scaled = scaler.transform(valid[feature_cols])
    test_scaled = scaler.transform(test[feature_cols])
    return train_scaled, valid_scaled, test_scaled, scaler
