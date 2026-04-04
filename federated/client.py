
import torch

class Client:
    def __init__(self, model, data, lr=0.001):
        self.model = model
        self.data = data
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, global_weights, epochs=1):
        # load global weights
        self.model.load_state_dict(global_weights)

        # DEBUG: check one sample
        sample_x, sample_y = self.data[0]
        print("Sample X shape:", sample_x.shape)
        print("Sample Y shape:", sample_y.shape)

        self.model.train()

        for _ in range(epochs):
            for x, y in self.data:
                x = torch.tensor(x).float()
                y = torch.tensor(y).float()

                # ✅ FORCE correct shape
                if len(x.shape) == 2:
                    x = x.unsqueeze(0)  # (seq, features) → (1, seq, features)

                elif len(x.shape) == 1:
                    x = x.unsqueeze(0).unsqueeze(0)

                y = y.view(1, -1)

                pred = self.model(x)

                loss = ((pred - y)**2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()
