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
        self.device = torch.device(global_config["device"])
        self.model_config = model_config
        self.global_config = global_config
        self.data_config = data_config
        self.fed_config = fed_config
        self.optim_config = optim_config

        # Accept either data_path or dataset_path
        self.data_path = data_config.get("data_path", data_config.get("dataset_path"))
        if self.data_path is None:
            raise KeyError("data_config must contain 'data_path' (or 'dataset_path').")
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

        self.x = eval(model_config["name"])().to(self.device)  # global model
        self.clients = None

        # MPI
        self.comm = comm or MPI.COMM_WORLD
        self.rank = rank
        self.size = size

        logging.info(f"Server init on rank {self.rank}/{self.size-1} | device={self.device}")

    def create_clients(self, local_datasets):
        clients = []
        for ld in local_datasets:
            clients.append(
                Client(
                    x=deepcopy(self.x),
                    batch_size=self.batch_size,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    device=self.device,
                    lr=self.lr_l,
                    data=ld,
                )
            )
        return clients

    def setup(self, **init_kwargs):
        """Initializes clients with 1 client per MPI process."""
        if self.rank == 0:
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

        # Scatter ONE dataset per rank
        local_dataset = self.comm.scatter(local_datasets, root=0)

        # Create exactly one client on each rank
        self.clients = self.create_clients([local_dataset])
        logging.info(f"Process {self.rank}: Initialized a single client with its own dataset.")

    def sample_clients(self):
        # 1 client per rank -> always [0]
        return [0]

    def server_update(self, client_ids):
        # Sum parameters from local client updates first
        avg_params = [torch.zeros_like(param.data) for param in self.x.parameters()]

        for idx in client_ids:
            # ⬅️ FIX #2: aggregate the **trained** local model (client.y), not client.x
            for param, client_param in zip(avg_params, self.clients[idx].y.parameters()):
                param.data += client_param.data

        # Allreduce to sum params across ranks
        for param in avg_params:
            self.comm.Allreduce(MPI.IN_PLACE, param.data, op=MPI.SUM)

        # Divide by global participating client count
        total_clients = self.comm.allreduce(len(client_ids), op=MPI.SUM)
        for param, summed in zip(self.x.parameters(), avg_params):
            param.data = (summed / total_clients).to(self.device)

    def train(self):
        for cur_round in range(self.num_rounds):
            client_ids = self.sample_clients()

            # ⬅️ FIX #3: sync client global weights from current server model every round
            for idx in client_ids:
                self.clients[idx].x = deepcopy(self.x)

            # Local training
            for idx in client_ids:
                self.clients[idx].train()

            # Global aggregation
            self.server_update(client_ids)

            # Optional evaluation on root
            if self.rank == 0:
                # ⬅️ FIX #1: call evaluate_fn with the correct order (dataloader, model, loss_fn, device)
                loss, acc = evaluate_fn(self.data, self.x, self.criterion, self.device)
                logging.info(f"Round {cur_round:03d} | Test Loss: {loss:.4f} | Test Acc: {acc:.2f}%")
