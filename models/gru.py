
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # DEBUG
        print("Input shape to GRU:", x.shape)

        out, _ = self.gru(x)

        print("Output shape from GRU:", out.shape)

        out = out[:, -1, :]  # safe now

        return self.fc(out)
