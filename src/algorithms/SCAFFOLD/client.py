import torch
from copy import deepcopy
from math import ceil

class Client():
    """
    SCAFFOLD-style client with control variates.
    Changes vs your original:
      - device normalized to torch.device
      - deltas kept as CPU tensors (contiguous) for MPI Allreduce
      - no algorithmic change to update rules
    """

    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr, client_c):
        self.id = client_id
        self.data = local_data
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion

        self.x = None               # set by server each round
        self.server_c = None        # set by server each round
        self.client_c = client_c    # persistent per-client control variate

        self.y = None
        self.delta_y = None
        self.delta_c = None

    def client_update(self):
        """
        Train local model 'y' from global 'x' with control variates, then compute:
            delta_y = y - x
            new_client_c = client_c - server_c - delta_y / a
            delta_c = new_client_c - client_c
        """
        if self.x is None or self.server_c is None:
            raise RuntimeError(f"Client {self.id}: x/server_c not initialized by server.")

        self.y = deepcopy(self.x).to(self.device)

        for _ in range(self.num_epochs):
            # Your original code used one mini-batch per epoch via next(iter(self.data))
            # Preserving that behavior to keep the algorithm intact.
            inputs, labels = next(iter(self.data))
            inputs = inputs.float().to(self.device)
            labels = labels.long().to(self.device)

            output = self.y(inputs)
            loss = self.criterion(output, labels)

            grads = torch.autograd.grad(loss, self.y.parameters())

            with torch.no_grad():
                for param, grad, s_c, c_c in zip(self.y.parameters(), grads, self.server_c, self.client_c):
                    s_c, c_c = s_c.to(self.device), c_c.to(self.device)
                    param.data = param.data - self.lr * (grad.data + (s_c.data - c_c.data))

        # Compute deltas on CPU (contiguous) so server can Allreduce numpy buffers
        with torch.no_grad():
            delta_y = [torch.zeros_like(p, device="cpu") for p in self.y.parameters()]
            new_client_c = [torch.zeros_like(c, device=self.device) for c in self.client_c]
            delta_c = [torch.zeros_like(c, device="cpu") for c in self.client_c]

            # delta_y = y - x
            for d, p_y, p_x in zip(delta_y, self.y.parameters(), self.x.parameters()):
                d.copy_((p_y.detach() - p_x.detach()).float().cpu().contiguous())

            # a = (#mini-batches per epoch) * num_epochs * lr
            a = (ceil(len(self.data.dataset) / self.data.batch_size) * self.num_epochs * self.lr)

            # new_client_c = client_c - server_c - delta_y / a
            for n_c, c_l, c_g, d in zip(new_client_c, self.client_c, self.server_c, delta_y):
                n_c.data = c_l.to(self.device).data - c_g.to(self.device).data - (d.to(self.device) / a)

            # delta_c = new_client_c - client_c
            for d_c, n_c_l, c_l in zip(delta_c, new_client_c, self.client_c):
                d_c.copy_((n_c_l.detach() - c_l.detach()).float().cpu().contiguous())

        # Persist updates
        self.client_c = [t.detach().to(self.device) for t in new_client_c]
        self.delta_y = delta_y               # list of CPU tensors
        self.delta_c = delta_c               # list of CPU tensors
