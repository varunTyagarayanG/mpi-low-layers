import torch
from copy import deepcopy

class Client:
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
        self.y = deepcopy(self.x)
        self.y.to(self.device)

        for epoch in range(self.num_epochs):
            for inputs, labels in self.data:  # full local dataset
                inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
                output = self.y(inputs)
                loss = self.criterion(output, labels)

                grads = torch.autograd.grad(loss, self.y.parameters())
                with torch.no_grad():
                    for param, grad in zip(self.y.parameters(), grads):
                        param.data = param.data - self.lr * grad.data

        # compute delta_y
        delta_y = [p_y.data.detach() - p_x.data.detach() for p_y, p_x in zip(self.y.parameters(), self.x.parameters())]
        self.delta_y = delta_y
