import torch
from copy import deepcopy

class Client:
    """
    One client per MPI rank.
    Server sets client.x (global model) each round in communicate().
    client_update() trains a local copy and computes delta_y = y - x (CPU tensors).
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
        self.delta_y = None     # list[Tensor(cpu)] parameter deltas

    def client_update(self):
        if self.x is None:
            raise RuntimeError(f"Client {self.id}: global model x is None. Server must call communicate() first.")

        # local copy of global
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()

        # --- your manual gradient step loop preserved ---
        for _ in range(self.num_epochs):
            for inputs, labels in self.data:  # iterate full local dataset
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)

                outputs = self.y(inputs)
                loss = self.criterion(outputs, labels)

                grads = torch.autograd.grad(loss, self.y.parameters())
                with torch.no_grad():
                    for p, g in zip(self.y.parameters(), grads):
                        p.data = p.data - self.lr * g.data

        # compute delta_y = y - x on CPU so server can Allreduce numpy buffers
        deltas = []
        with torch.no_grad():
            for p_y, p_x in zip(self.y.parameters(), self.x.parameters()):
                d = (p_y.detach() - p_x.detach()).float().cpu().contiguous()
                deltas.append(d)
        self.delta_y = deltas
