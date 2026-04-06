import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, n_features=7, hidden=64,
                 layers=2, pred_len=96, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, n_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -self.pred_len:, :]
        return self.fc(out)
