import torch
from copy import deepcopy

class Client:
    """
    Local client in federated learning.
    """
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr):
        self.id = client_id
        self.data = local_data
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion
        self.x = None
        self.y = None
        self.delta_y = None

    def client_update(self):
        self.y = deepcopy(self.x).to(self.device)

        for epoch in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)

                outputs = self.y(inputs)
                loss = self.criterion(outputs, labels)

                self.y.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for param in self.y.parameters():
                        param.data -= self.lr * param.grad

        with torch.no_grad():
            self.delta_y = [
                param_y.data.detach() - param_x.data.detach()
                for param_y, param_x in zip(self.y.parameters(), self.x.parameters())
            ]
