import torch
from copy import deepcopy

class Client():
    """
    Server uses Client class to create multiple client objects.
    Client trains the model on its local data. Then the client updates its local model (y)
    and sends updates back to the server.
    
    Attributes:
        id: Acts as an identifier for a particular client
        data: Local dataset which resides on the client (as a DataLoader)
        device: Specifies which device (cpu or gpu) to use for training
        num_epochs: Number of epochs to train the local model
        lr: Local stepsize
        criterion: Measures the disagreement between model's prediction and ground truth
        x: Global model sent by server
        y: Local model initialized using x
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

    def client_update(self):
        """
        Trains the model on its local data. Updates the local copy (y) using the gradients
        and stores it for sending back to the server.
        """
        self.y = deepcopy(self.x)  # Initialize local model from global model

        for epoch in range(self.num_epochs):
            for inputs, labels in self.data:  # Iterate over all batches in local dataset
                inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
                output = self.y(inputs)
                loss = self.criterion(output, labels)

                # Compute gradients of the loss w.r.t. model parameters
                grads = torch.autograd.grad(loss, self.y.parameters())

                # Update local model parameters using gradient descent
                with torch.no_grad():
                    for param, grad in zip(self.y.parameters(), grads):
                        param.data -= self.lr * grad.data

        # Optional: Clear cache if using GPU
        # if self.device == "cuda":
        #     torch.cuda.empty_cache()
