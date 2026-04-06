import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, seq_len=96, pred_len=96,
                 n_features=7, hidden=256):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(seq_len * n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, pred_len * n_features)
        )

    def forward(self, x):
        b = x.shape[0]
        out = self.net(x.reshape(b, -1))
        return out.reshape(b, self.pred_len, self.n_features)
