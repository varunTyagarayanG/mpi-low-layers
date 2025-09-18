import numpy as np
import logging
import torch
from copy import deepcopy
from .client import Client
from src.models import *
from src.load_data_for_clients import dist_data_per_client
from src.util_functions import set_seed, evaluate_fn
from mpi4py import MPI


class Server:
    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={},
                 comm=None, rank=0, size=1):
        set_seed(global_config["seed"])
        self.device = global_config["device"]
        self.model_config = model_config
        self.global_config = global_config
        self.data_config = data_config
        self.fed_config = fed_config
        self.optim_config = optim_config

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.non_iid_per = data_config["non_iid_per"]

        self.fraction = fed_config["fraction_clients"]
        self.num_clients = fed_config["num_clients"]
        self.num_rounds = fed_config["num_rounds"]
        self.num_epochs = fed_config["num_epochs"]
        self.batch_size = fed_config["batch_size"]
        self.criterion = eval(fed_config["criterion"])()
        self.lr = fed_config["global_stepsize"]
        self.lr_l = fed_config["local_stepsize"]

        self.x = eval(model_config["name"])()
        self.clients = None

        # MPI attributes
        self.comm = comm or MPI.COMM_WORLD
        self.rank = rank
        self.size = size
        self.device = torch.device(self.device)
        self.x.to(self.device)

        logging.info(f"Server init on rank {self.rank}/{self.size-1} | device={self.device}")

    def create_clients(self, local_datasets):
        clients = []
        for ld in local_datasets:
            clients.append(Client(x=deepcopy(self.x), batch_size=self.batch_size, num_epochs=self.num_epochs,
                                  criterion=self.criterion, device=self.device, lr=self.lr_l, data=ld))
        return clients

    def setup(self, **init_kwargs):
        """Initializes clients with 1 client per MPI process."""
        if self.rank == 0:
            # Prepare exactly num_clients datasets
            local_datasets, test_dataset = dist_data_per_client(
                self.data_path, self.dataset_name, self.num_clients,
                self.batch_size, self.non_iid_per, self.device
            )
            if len(local_datasets) != self.num_clients:
                raise ValueError(f"dist_data_per_client returned {len(local_datasets)} clients, expected {self.num_clients}.")
            if self.num_clients != self.size:
                raise ValueError(f"num_clients ({self.num_clients}) must equal MPI world size ({self.size}) when running 1 client per process.")
        else:
            local_datasets = None
            test_dataset = None

        # Broadcast shared test dataset to all ranks
        test_dataset = self.comm.bcast(test_dataset, root=0)
        self.data = test_dataset

        # Scatter ONE dataset per rank (length must equal self.size)
        local_dataset = self.comm.scatter(local_datasets, root=0)

        # Create exactly one client on each rank
        self.clients = self.create_clients([local_dataset])
        # Ensure fraction_clients is at least 1 client per round locally
        self.fraction_clients = 1.0
        logging.info(f"Process {self.rank}: Initialized a single client with its own dataset.")

    def sample_clients(self):
        """Selects local participating clients for training on THIS rank.
        In 1-client-per-process mode, this always returns [0]."""
        if len(self.clients) == 1:
            return [0]
        else:
            num_sampled_clients = max(1, int(self.fraction * len(self.clients)))
            return list(np.random.choice(np.arange(0, len(self.clients)), size=num_sampled_clients, replace=False).tolist())

    def communicate(self, op_name, data=None):
        if op_name == "sum":
            # Sum tensors across ranks
            for p in data:
                self.comm.Allreduce(MPI.IN_PLACE, p, op=MPI.SUM)
            return data
        elif op_name == "avg":
            # Average tensors across ranks
            for p in data:
                self.comm.Allreduce(MPI.IN_PLACE, p, op=MPI.SUM)
            for p in data:
                p /= self.size
            return data
        else:
            raise NotImplementedError

    def server_update(self, client_ids):
        # Sum parameters from local client updates first
        avg_params = [torch.zeros_like(param.data) for param in self.x.parameters()]
        for idx in client_ids:
            self.clients[idx].x.to(self.device)
            for param, client_param in zip(avg_params, self.clients[idx].x.parameters()):
                param.data += client_param.data

        # Allreduce to sum params across ranks
        for param in avg_params:
            self.comm.Allreduce(MPI.IN_PLACE, param.data, op=MPI.SUM)

        # Divide by global participating client count
        local_count = len(client_ids)
        total_clients = self.comm.allreduce(local_count, op=MPI.SUM)
        for param, avg in zip(self.x.parameters(), avg_params):
            param.data = (avg / total_clients).to(self.device)

    def train(self):
        for cur_round in range(self.num_rounds):
            # Sample local clients
            client_ids = self.sample_clients()

            # Local training
            for idx in client_ids:
                self.clients[idx].x.to(self.device)
                self.clients[idx].train()

            # Server update (global aggregation via MPI)
            self.server_update(client_ids)

            # Optionally evaluate on root
            if self.rank == 0 and (cur_round % 1 == 0):
                loss, acc = evaluate_fn(self.x, self.data, self.device)
                logging.info(f"Round {cur_round:03d} | Test Loss: {loss:.4f} | Test Acc: {acc:.2f}%")