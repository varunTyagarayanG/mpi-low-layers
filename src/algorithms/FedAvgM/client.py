import torch
from copy import deepcopy

class Client:
    """
    One client per MPI rank.
    Server sets client.x (global model) each round in communicate().
    client_update() trains a local copy and computes delta_y = y - x.
    """

    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr):
        self.id = client_id
        self.data = local_data
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion
        self.x = None           # set by server before training
        self.y = None           # local trained model
        self.delta_y = None     # list of parameter deltas (CPU tensors)

    def client_update(self):
        if self.x is None:
            raise RuntimeError(f"Client {self.id}: global model x is None. Server must call communicate() first.")
        # local copy of global
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()

        optimizer = torch.optim.SGD(self.y.parameters(), lr=self.lr)

        for _ in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)
                optimizer.zero_grad()
                outputs = self.y(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # compute delta_y = y - x on CPU (so server can Allreduce numpy buffers)
        with torch.no_grad():
            deltas = []
            for p_y, p_x in zip(self.y.parameters(), self.x.parameters()):
                # detach, move to CPU, and store difference
                deltas.append((p_y.detach().cpu() - p_x.detach().cpu()))
            self.delta_y = deltas
