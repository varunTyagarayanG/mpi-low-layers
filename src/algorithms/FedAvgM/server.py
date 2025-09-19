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

        # Accept both keys for compatibility
        self.data_path = data_config.get("data_path", data_config.get("dataset_path"))
        if self.data_path is None:
            raise KeyError("data_config must contain 'data_path' (or 'dataset_path').")
        self.dataset_name = data_config["dataset_name"]
        self.non_iid_per = data_config["non_iid_per"]

        self.fraction = fed_config.get("fraction_clients", 1.0)  # ignored in 1-client-per-rank mode
        self.num_clients = fed_config["num_clients"]
        self.num_rounds = fed_config["num_rounds"]
        self.num_epochs = fed_config["num_epochs"]
        self.batch_size = fed_config["batch_size"]
        self.criterion = eval(fed_config["criterion"])()
        self.lr = fed_config["global_stepsize"]        # server LR on momentum velocity
        self.lr_l = fed_config["local_stepsize"]       # client LR
        self.beta = fed_config.get("momentum", 0.9)    # momentum factor

        # global model
        self.x = eval(model_config["name"])().to(self.device)
        self.clients = None

        # momentum velocity kept on device
        self.velocity = [torch.zeros_like(p.data, device=self.device) for p in self.x.parameters()]

        # MPI
        self.comm = comm if comm else MPI.COMM_WORLD
        self.rank = rank
        self.size = size

        logging.info(f"Server init on rank {self.rank}/{self.size-1} | device={self.device}")

    def create_clients(self, local_datasets):
        # In 1-client-per-rank mode, local_datasets has length 1
        clients = []
        for i, dataset in enumerate(local_datasets):
            cid = self.rank  # give global id == rank for clear logs
            client = Client(
                client_id=cid,
                local_data=dataset,
                device=self.device,
                num_epochs=self.num_epochs,
                criterion=self.criterion,
                lr=self.lr_l
            )
            clients.append(client)
        return clients

    def setup(self, **init_kwargs):
        """One client per MPI process: build exactly num_clients loaders, scatter one to each rank."""
        if self.rank == 0:
            local_datasets, test_dataset = dist_data_per_client(
                self.data_path, self.dataset_name, self.num_clients,
                self.batch_size, self.non_iid_per, self.device
            )
            if len(local_datasets) != self.num_clients:
                raise ValueError(f"dist_data_per_client returned {len(local_datasets)} clients, expected {self.num_clients}.")
            if self.num_clients != self.size:
                raise ValueError(f"num_clients ({self.num_clients}) must equal MPI world size ({self.size}) "
                                 f"when running 1 client per process.")
        else:
            local_datasets = None
            test_dataset = None

        # share the same test loader to all ranks
        test_dataset = self.comm.bcast(test_dataset, root=0)
        self.data = test_dataset

        # scatter ONE dataset per rank
        local_dataset = self.comm.scatter(local_datasets, root=0)
        self.clients = self.create_clients([local_dataset])
        logging.info(f"Process {self.rank}: Initialized a single client with its own dataset.")

    def sample_clients(self):
        # one client on this rank ⇒ always [0]
        return [0]

    def communicate(self, client_ids):
        # Set each sampled client's x to current global model
        for idx in client_ids:
            self.clients[idx].x = deepcopy(self.x)

    def update_clients(self, client_ids):
        for idx in client_ids:
            self.clients[idx].client_update()

    def server_update(self, client_ids):
        """
        Aggregate delta_y from all ranks with momentum.
        We sum on CPU tensors (numpy buffers for Allreduce), then update velocity and global params on device.
        """
        # sum of deltas across local sampled clients (here only one)
        summed_deltas_cpu = [torch.zeros_like(p.data, device='cpu') for p in self.x.parameters()]
        for idx in client_ids:
            if self.clients[idx].delta_y is None:
                raise RuntimeError(f"Process {self.rank}: delta_y is None for client {self.clients[idx].id}")
            for acc, d in zip(summed_deltas_cpu, self.clients[idx].delta_y):
                acc.add_(d)

        # Allreduce over all ranks (operate on numpy buffers)
        for i in range(len(summed_deltas_cpu)):
            # ensure contiguous float32 numpy array
            buf = summed_deltas_cpu[i].contiguous().numpy()
            self.comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)

        # number of participating clients globally this round
        local_count = len(client_ids)
        total_count = self.comm.allreduce(local_count, op=MPI.SUM)
        if total_count == 0:
            return

        # momentum update on device
        with torch.no_grad():
            for v, delta_cpu, param in zip(self.velocity, summed_deltas_cpu, self.x.parameters()):
                avg_delta = (delta_cpu / total_count).to(self.device)
                v.data = self.beta * v.data + avg_delta
                param.data = param.data + self.lr * v.data

    def step(self):
        client_ids = self.sample_clients()
        self.communicate(client_ids)
        self.update_clients(client_ids)
        logging.info(f"Process {self.rank}: client_update completed")
        self.server_update(client_ids)
        logging.info(f"Process {self.rank}: server_update completed")

    def train(self):
        self.results = {"loss": [], "accuracy": []}
        for rnd in range(self.num_rounds):
            logging.info(f"\nProcess {self.rank}: Communication Round {rnd+1}")
            self.step()

            # Evaluate (use correct signature: dataloader, model, loss_fn, device)
            test_loss, test_acc = evaluate_fn(self.data, self.x, self.criterion, self.device)

            # Gather metrics on root
            losses = self.comm.gather(test_loss, root=0)
            accs   = self.comm.gather(test_acc, root=0)

            if self.rank == 0:
                avg_loss = float(sum(losses) / len(losses))
                avg_acc  = float(sum(accs) / len(accs))
                self.results['loss'].append(avg_loss)
                self.results['accuracy'].append(avg_acc)
                logging.info(f"\tLoss: {avg_loss:.4f}   Accuracy: {avg_acc:.2f}%")
