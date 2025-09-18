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

        for epoch in range(self.num_epochs):
            inputs, labels = next(iter(self.data))
            inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
            output = self.y(inputs)
            loss = self.criterion(output, labels)

            grads = torch.autograd.grad(loss, self.y.parameters())
            with torch.no_grad():
                for param, grad in zip(self.y.parameters(), grads):
                    param.data = param.data - self.lr * grad.data

        with torch.no_grad():
            delta_y = [torch.zeros_like(param) for param in self.y.parameters()]
            for d_param, param_y, param_x in zip(delta_y, self.y.parameters(), self.x.parameters()):
                d_param.data += param_y.data.detach() - param_x.data.detach()
        self.delta_y = delta_y
