import torch
from copy import deepcopy

class Client:
    """
    Each process holds exactly one Client object.
    The client trains the global model (x) on its local dataset (data),
    and keeps a local copy (y) to send updates back to the server.
    """

    def __init__(self, x, batch_size, num_epochs, criterion, device, lr, data):
        self.x = deepcopy(x)              # Global model copy
        self.y = None                     # Local model after training
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.device = device
        self.lr = lr
        self.data = data                  # Local dataset (already DataLoader)

    def train(self):
        """
        Train the local copy (y) starting from global model (x).
        """
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()

        optimizer = torch.optim.SGD(self.y.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
                optimizer.zero_grad()
                outputs = self.y(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Return local model (optional, not used because server pulls params directly)
        return self.y
