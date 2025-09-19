import torch
from copy import deepcopy

class Client:
    """
    Each process holds exactly one Client object.
    The client trains the global model (x) on its local dataset (data),
    and keeps a local copy (y) to send updates back to the server.
    """

    def __init__(self, x, batch_size, num_epochs, criterion, device, lr, data):
        self.x = deepcopy(x)              # Global model received from server (refreshed each round)
        self.y = None                     # Local trained model after training
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.device = device
        self.lr = lr
        self.data = data                  # Local dataset (DataLoader)

    def train(self):
        """
        Train the local copy (y) starting from the current global model (x).
        """
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()

        optimizer = torch.optim.SGD(self.y.parameters(), lr=self.lr)

        for _ in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
                optimizer.zero_grad()
                outputs = self.y(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.y
