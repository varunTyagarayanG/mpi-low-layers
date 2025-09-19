import numpy as np
import logging
import torch
from copy import deepcopy
from mpi4py import MPI

from .client import Client
from src.models import *
from src.load_data_for_clients import dist_data_per_client
from src.util_functions import set_seed, evaluate_fn

class Server:
    """
    MPI-compatible FedAvg/MimeLite-style server
    - One client per MPI rank
    - Clients compute delta_y; server applies momentum update to global model
    """

    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={}, comm=None, rank=None, size=None):
        set_seed(global_config["seed"])

        # MPI
        self.comm  = comm if comm else MPI.COMM_WORLD
        self.rank  = rank if rank is not None else self.comm.Get_rank()
        self.size  = size if size is not None else self.comm.Get_Size()

        # Configs
        self.device = torch.device(global_config["device"])
        self.data_path    = data_config.get("data_path", data_config.get("dataset_path"))
        if self.data_path is None:
            raise KeyError("data_config must contain 'data_path' (or 'dataset_path').")
        self.dataset_name = data_config["dataset_name"]
        self.non_iid_per  = data_config["non_iid_per"]

        self.fraction     = fed_config.get("fraction_clients", 1.0)  # ignored in 1-client-per-rank
        self.num_clients  = fed_config["num_clients"]
        self.num_rounds   = fed_config["num_rounds"]
        self.num_epochs   = fed_config["num_epochs"]
        self.batch_size   = fed_config["batch_size"]
        self.criterion    = eval(fed_config["criterion"])()
        self.lr           = fed_config["global_stepsize"]   # server LR on momentum velocity
        self.lr_l         = fed_config["local_stepsize"]    # client LR
        self.beta         = fed_config.get("momentum", 0.9) # momentum factor

        # Global model + momentum state
        self.x = eval(model_config["name"])().to(self.device)
        self.state = [torch.zeros_like(p.data, device=self.device) for p in self.x.parameters()]

        self.clients = None
        self.data    = None

        logging.info(f"Server init on rank {self.rank}/{self.size-1} | device={self.device}")

    def create_clients(self, local_datasets):
        # In 1-client-per-rank mode, local_datasets has length 1
        clients = []
        for dataset in local_datasets:
            cid = self.rank  # stable id = rank (clean logs)
            c = Client(
                client_id=cid,
                local_data=dataset,
                device=self.device,
                num_epochs=self.num_epochs,
                criterion=self.criterion,
                lr=self.lr_l
            )
            clients.append(c)
        return clients

    def setup(self):
        """
        Root builds exactly num_clients loaders, then we broadcast test set and scatter ONE dataset per rank.
        Enforce num_clients == MPI world size to keep 1 client per rank.
        """
        if self.rank == 0:
            local_datasets, test_dataset = dist_data_per_client(
                self.data_path, self.dataset_name, self.num_clients,
                self.batch_size, self.non_iid_per, self.device
            )
            if len(local_datasets) != self.num_clients:
                raise ValueError(f"dist_data_per_client returned {len(local_datasets)} clients, expected {self.num_clients}.")
            if self.num_clients != self.size:
                raise ValueError(f"num_clients ({self.num_clients}) must equal MPI world size ({self.size}) for 1 client per process.")
        else:
            local_datasets = None
            test_dataset   = None

        # share test loader
        test_dataset = self.comm.bcast(test_dataset, root=0)
        self.data = test_dataset

        # scatter exactly one dataset to this rank
        local_dataset = self.comm.scatter(local_datasets, root=0)
        self.clients = self.create_clients([local_dataset])
        logging.info(f"Process {self.rank}: Initialized a single client with its own dataset.")

    def sample_clients(self):
        # 1 client per rank → always the single local client
        return [0]

    def communicate(self, client_ids):
        # Refresh local clients with the latest global weights
        for idx in client_ids:
            self.clients[idx].x = deepcopy(self.x)

    def update_clients(self, client_ids):
        for idx in client_ids:
            self.clients[idx].client_update()

    def server_update(self, client_ids):
        """
        Aggregate delta_y across all ranks and apply momentum update:
            v = (1 - beta) * avg_delta + beta * v
            x = x + lr * v
        """
        # Sum local deltas (CPU tensors)
        summed_deltas_cpu = [torch.zeros_like(p.data, device='cpu') for p in self.x.parameters()]
        for idx in client_ids:
            if self.clients[idx].delta_y is None:
                raise RuntimeError(f"Rank {self.rank}: delta_y is None for client {self.clients[idx].id}")
            for acc, d in zip(summed_deltas_cpu, self.clients[idx].delta_y):
                acc.add_(d)  # still CPU

        # Allreduce: sum deltas across ranks (numpy buffers)
        for i in range(len(summed_deltas_cpu)):
            buf = summed_deltas_cpu[i].numpy()           # contiguous float32
            self.comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)

        # number of participating clients globally this round
        total_count = self.comm.allreduce(len(client_ids), op=MPI.SUM)
        if total_count <= 0:
            return

        # Momentum + param update on device
        with torch.no_grad():
            for v, d_cpu, p in zip(self.state, summed_deltas_cpu, self.x.parameters()):
                avg_delta = (d_cpu / total_count).to(self.device)
                # momentum in the same form you were using: s = (1-β)*grad + β*s
                v.data = (1.0 - self.beta) * avg_delta + self.beta * v.data
                p.data = p.data + self.lr * v.data

    def step(self):
        ids = self.sample_clients()
        self.communicate(ids)
        self.update_clients(ids)
        logging.info(f"Rank {self.rank}: client_update completed")
        self.server_update(ids)
        logging.info(f"Rank {self.rank}: server_update completed")

    def train(self):
        self.results = {"loss": [], "accuracy": []}
        for round_idx in range(self.num_rounds):
            logging.info(f"Rank {self.rank}: Communication Round {round_idx + 1}")
            self.step()

            # Evaluate with correct signature (dataloader, model, loss_fn, device)
            test_loss, test_acc = evaluate_fn(self.data, self.x, self.criterion, self.device)

            # gather to root for logging
            losses = self.comm.gather(test_loss, root=0)
            accs   = self.comm.gather(test_acc,  root=0)
            if self.rank == 0:
                avg_loss = float(sum(losses) / len(losses))
                avg_acc  = float(sum(accs)   / len(accs))
                self.results["loss"].append(avg_loss)
                self.results["accuracy"].append(avg_acc)
                logging.info(f"Round {round_idx + 1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.2f}%")
