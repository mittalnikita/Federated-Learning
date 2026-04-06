def fedprox_train(model, dataloader, global_state_dict,
                  local_epochs=5, lr=0.001, mu=0.1, device='cpu'):
    import torch
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    global_params = [p.clone().detach()
                     for p in model.parameters()]
    for epoch in range(local_epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            prox = sum(
                (p - g).norm(2) ** 2
                for p, g in zip(model.parameters(), global_params)
            )
            loss = loss + (mu / 2) * prox
            loss.backward()
            optimizer.step()
    return model.state_dict(), len(dataloader.dataset)
