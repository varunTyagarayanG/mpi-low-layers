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
    SCAFFOLD-style server with control variates (server_c / client_c).
    Changes vs your original:
      - 1 MPI rank = 1 client (scatter exactly one dataset per rank)
      - Robust MPI Allreduce over CPU numpy buffers for delta_y and delta_c
      - Accept both data_path and dataset_path from config
      - evaluate_fn called with (dataloader, model, loss_fn, device)
    Algorithmic updates (how x and server_c change) are unchanged in spirit.
    """

    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={},
                 comm=None, rank=0, size=1):
        set_seed(global_config["seed"])

        # MPI
        self.comm = comm if comm else MPI.COMM_WORLD
        self.rank = rank
        self.size = size

        # Configs
        self.device = torch.device(global_config["device"])
        self.data_path = data_config.get("data_path", data_config.get("dataset_path"))  # keep compat
        if self.data_path is None:
            raise KeyError("data_config must contain 'data_path' (or 'dataset_path').")
        self.dataset_name = data_config["dataset_name"]
        self.non_iid_per  = data_config["non_iid_per"]

        self.fraction    = fed_config.get("fraction_clients", 1.0)  # not used in 1-client-per-rank sampling
        self.num_clients = fed_config["num_clients"]
        self.num_rounds  = fed_config["num_rounds"]
        self.num_epochs  = fed_config["num_epochs"]
        self.batch_size  = fed_config["batch_size"]
        self.criterion   = eval(fed_config["criterion"])()
        self.lr          = fed_config["global_stepsize"]   # server LR for model update
        self.lr_l        = fed_config["local_stepsize"]    # client LR
        self.beta        = fed_config.get("momentum", None)  # not used in pure SCAFFOLD; kept for compat

        # Global model + server control variate
        self.x = eval(model_config["name"])().to(self.device)
        self.server_c = [torch.zeros_like(p.data, device=self.device) for p in self.x.parameters()]

        self.clients = None
        self.data = None

        logging.info(f"Server init on rank {self.rank}/{self.size-1} | device={self.device}")

    def create_clients(self, local_datasets):
        # 1 dataset here -> 1 client per rank; id = rank for clearer logs
        clients = []
        for dataset in local_datasets:
            c = Client(
                client_id=self.rank,
                local_data=dataset,
                device=self.device,
                num_epochs=self.num_epochs,
                criterion=self.criterion,
                lr=self.lr_l,
                client_c=deepcopy(self.server_c),
            )
            clients.append(c)
        return clients

    def setup(self, **init_kwargs):
        """
        Root creates exactly num_clients loaders; scatter ONE to each rank.
        Enforce num_clients == MPI world size (1 client per process).
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
            test_dataset = None

        # Broadcast the test dataset to all ranks
        test_dataset = self.comm.bcast(test_dataset, root=0)
        self.data = test_dataset

        # Scatter exactly one local dataset to each rank
        local_dataset = self.comm.scatter(local_datasets, root=0)
        self.clients = self.create_clients([local_dataset])
        logging.info(f"Process {self.rank}: initialized a single client with its own dataset")

    def sample_clients(self):
        # 1 client per rank → always the local client [0]
        return [0]

    def communicate(self, client_ids):
        # Send current global model and server control variate to selected client(s)
        for idx in client_ids:
            self.clients[idx].x = deepcopy(self.x)
            self.clients[idx].server_c = deepcopy(self.server_c)

    def update_clients(self, client_ids):
        for idx in client_ids:
            self.clients[idx].client_update()

    def server_update(self, client_ids):
        """
        Aggregate client deltas with MPI, then update:
            x ← x + lr * avg(delta_y)
            server_c ← server_c + avg(delta_c)   (same semantics as your original)
        """
        # Local accumulation (CPU tensors for MPI)
        sum_dy_cpu = [torch.zeros_like(p.data, device="cpu") for p in self.x.parameters()]
        sum_dc_cpu = [torch.zeros_like(c.data, device="cpu") for c in self.server_c]

        for idx in client_ids:
            if self.clients[idx].delta_y is None or self.clients[idx].delta_c is None:
                raise RuntimeError(f"Rank {self.rank}: missing deltas for client {self.clients[idx].id}")
            for acc, d in zip(sum_dy_cpu, self.clients[idx].delta_y):
                acc.add_(d.detach().cpu())
            for acc_c, d_c in zip(sum_dc_cpu, self.clients[idx].delta_c):
                acc_c.add_(d_c.detach().cpu())

        # Allreduce across ranks (operate on contiguous numpy buffers)
        for t in sum_dy_cpu:
            self.comm.Allreduce(MPI.IN_PLACE, t.numpy(), op=MPI.SUM)
        for t in sum_dc_cpu:
            self.comm.Allreduce(MPI.IN_PLACE, t.numpy(), op=MPI.SUM)

        # Global #participants this round (always 1 per rank here)
        total_clients = self.comm.allreduce(len(client_ids), op=MPI.SUM)
        if total_clients <= 0:
            return

        # Apply averaged updates (keep your update rules)
        with torch.no_grad():
            for p, dy in zip(self.x.parameters(), sum_dy_cpu):
                p.data.add_((dy / total_clients).to(self.device) * self.lr)

            # Preserve your semantics for server_c: add the averaged delta_c
            for c_g, dc in zip(self.server_c, sum_dc_cpu):
                c_g.data.add_((dc / total_clients).to(self.device))

    def step(self):
        ids = self.sample_clients()
        self.communicate(ids)
        self.update_clients(ids)
        logging.info(f"Process {self.rank}: client_update completed")
        self.server_update(ids)
        logging.info(f"Process {self.rank}: server_update completed")

    def train(self):
        self.results = {"loss": [], "accuracy": []}
        for rnd in range(self.num_rounds):
            logging.info(f"\nProcess {self.rank}: Communication Round {rnd+1}")
            self.step()

            # evaluate_fn signature: (dataloader, model, loss_fn, device)
            test_loss, test_acc = evaluate_fn(self.data, self.x, self.criterion, self.device)

            # Gather metrics on root
            losses = self.comm.gather(test_loss, root=0)
            accs   = self.comm.gather(test_acc,  root=0)

            if self.rank == 0:
                avg_loss = float(sum(losses) / len(losses))
                avg_acc  = float(sum(accs)   / len(accs))
                self.results["loss"].append(avg_loss)
                self.results["accuracy"].append(avg_acc)
                logging.info(f"\tLoss: {avg_loss:.4f}   Accuracy: {avg_acc:.2f}%")
