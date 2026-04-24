import torch
import torch.nn as nn
import torch.nn.functional as F

# 参考官方实现：https://github.com/thuml/iTransformer
class DataEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Embedding(5000, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x = self.value_embedding(x) + self.position_embedding(pos)
        return self.dropout(x)

class iTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        # src: [batch, seq_len, d_model]
        attn_output, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(ff_output))
        return src

class iTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, output_dim=1, seq_len=96, pred_len=24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding = DataEmbedding(input_dim, d_model, dropout)
        self.encoder_layers = nn.ModuleList([
            iTransformerLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.projection = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        # 只取最后pred_len步做预测
        out = self.projection(x[:, -self.pred_len:, :])
        return out.squeeze(-1)  # [batch, pred_len]
