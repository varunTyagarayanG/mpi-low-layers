import torch
from copy import deepcopy
import logging
import numpy as np
from mpi4py import MPI

from .client import Client
from src.models import *
from src.load_data_for_clients import dist_data_per_client
from src.util_functions import set_seed, evaluate_fn

class Server:
    """
    MPI-compatible FedAdam server
    """
    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, comm=None, rank=None, size=None):
        set_seed(global_config["seed"])
        
        # MPI
        self.comm = comm if comm else MPI.COMM_WORLD
        self.rank = rank if rank is not None else self.comm.Get_rank()
        self.size = size if size is not None else self.comm.Get_size()

        self.device = global_config["device"]

        self.data_path = data_config["dataset_path"]
        self.dataset_name = data_config["dataset_name"]
        self.non_iid_per = data_config["non_iid_per"]

        self.num_clients = fed_config["num_clients"]
        self.num_rounds = fed_config["num_rounds"]
        self.num_epochs = fed_config["num_epochs"]
        self.batch_size = fed_config["batch_size"]
        self.fraction = fed_config["fraction_clients"]
        self.criterion = eval(fed_config["criterion"])()
        self.lr = fed_config["global_stepsize"]
        self.lr_l = fed_config["local_stepsize"]

        self.x = eval(model_config["name"])().to(self.device)
        self.m = [torch.zeros_like(p, device=self.device) for p in self.x.parameters()]
        self.v = [torch.zeros_like(p, device=self.device) for p in self.x.parameters()]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6
        self.timestep = 1

        self.clients = []
        self.test_data = None

    def setup(self):
        """Split dataset across ranks and initialize clients"""
        if self.rank == 0:
            local_datasets, test_dataset = dist_data_per_client(
                self.data_path, self.dataset_name, self.num_clients,
                self.batch_size, self.non_iid_per, self.device
            )
            chunks = np.array_split(local_datasets, self.size)
        else:
            chunks = None
            test_dataset = None

        local_data = self.comm.scatter(chunks, root=0)
        self.test_data = self.comm.bcast(test_dataset, root=0)

        for idx, dataset in enumerate(local_data):
            client_id = self.rank * len(local_data) + idx
            self.clients.append(Client(
                client_id=client_id,
                local_data=dataset,
                device=self.device,
                num_epochs=self.num_epochs,
                criterion=self.criterion,
                lr=self.lr_l
            ))

        logging.info(f"Rank {self.rank}: Initialized {len(self.clients)} clients")

    def communicate(self):
        """Broadcast global model to all clients"""
        x_params = [p.detach().cpu().numpy() for p in self.x.parameters()]
        x_params = self.comm.bcast(x_params, root=0)

        for param, new_param in zip(self.x.parameters(), x_params):
            param.data = torch.tensor(new_param, device=self.device)

        for client in self.clients:
            client.x = deepcopy(self.x)

    def update_clients(self):
        """Local client updates"""
        for client in self.clients:
            client.client_update()

    def server_update(self):
        """Aggregate delta_y from all clients and perform Adam update"""
        local_grads = [torch.zeros_like(p, device=self.device) for p in self.x.parameters()]

        for client in self.clients:
            for g, delta in zip(local_grads, client.delta_y):
                g.data += delta.data / self.num_clients

        # Aggregate across all ranks
        global_grads = []
        for g in local_grads:
            total = np.zeros_like(g.cpu().numpy())
            self.comm.Allreduce(g.detach().cpu().numpy(), total, op=MPI.SUM)
            global_grads.append(torch.tensor(total, device=self.device))

        with torch.no_grad():
            for p, g, m, v in zip(self.x.parameters(), global_grads, self.m, self.v):
                m.data = self.beta1 * m.data + (1 - self.beta1) * g.data
                v.data = self.beta2 * v.data + (1 - self.beta2) * torch.square(g.data)
                m_hat = m / (1 - self.beta1**self.timestep)
                v_hat = v / (1 - self.beta2**self.timestep)
                p.data += self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

        self.timestep += 1

    def evaluate(self):
        if self.rank == 0:
            loss, acc = evaluate_fn(self.test_data, self.x, self.criterion, self.device)
            logging.info(f"Test Loss: {loss:.4f}, Accuracy: {acc:.2f}%")

    def train(self):
        for round in range(self.num_rounds):
            logging.info(f"Rank {self.rank}: Starting round {round+1}")
            self.communicate()
            self.update_clients()
            logging.info(f"Rank {self.rank}: Clients updated")
            self.server_update()
            if self.rank == 0:
                logging.info(f"Round {round+1} completed")
                self.evaluate()
