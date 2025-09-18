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

        self.data_path = data_config["dataset_path"]
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
        self.beta = fed_config.get("momentum", 0.9)  # momentum factor

        self.x = eval(model_config["name"])()
        self.clients = None
        self.velocity = [torch.zeros_like(param.data, device=self.device) for param in self.x.parameters()]

        # MPI attributes
        self.comm = comm if comm else MPI.COMM_WORLD
        self.rank = rank
        self.size = size

    def create_clients(self, local_datasets):
        clients = []
        for id_num, dataset in enumerate(local_datasets):
            client = Client(client_id=id_num, local_data=dataset, device=self.device,
                            num_epochs=self.num_epochs, criterion=self.criterion, lr=self.lr_l)
            clients.append(client)
        return clients

    def setup(self, **init_kwargs):
        """Initializes all the Clients and splits the train dataset among them"""
        if self.rank == 0:
            local_datasets, test_dataset = dist_data_per_client(
                self.data_path, self.dataset_name, self.num_clients,
                self.batch_size, self.non_iid_per, self.device
            )
        else:
            local_datasets = None
            test_dataset = None

        test_dataset = self.comm.bcast(test_dataset, root=0)
        self.data = test_dataset

        if self.rank == 0:
            datasets_per_rank = np.array_split(local_datasets, self.size)
        else:
            datasets_per_rank = None

        local_dataset = self.comm.scatter(datasets_per_rank, root=0)
        self.clients = self.create_clients(local_dataset)
        logging.info(f"Process {self.rank}: Clients are successfully initialized")

    def sample_clients(self):
        num_sampled_clients = max(int(self.fraction * len(self.clients)), 1)
        sampled_client_ids = sorted(np.random.choice(a=[i for i in range(len(self.clients))],
                                                     size=num_sampled_clients, replace=False).tolist())
        return sampled_client_ids

    def communicate(self, client_ids):
        for idx in client_ids:
            self.clients[idx].x = deepcopy(self.x)

    def update_clients(self, client_ids):
        for idx in client_ids:
            self.clients[idx].client_update()

    def server_update(self, client_ids):
        avg_params = [torch.zeros_like(param.data, device='cpu') for param in self.x.parameters()]

        for idx in client_ids:
            for avg, param in zip(avg_params, self.clients[idx].delta_y):
                avg.add_(param.detach().cpu())

        for i in range(len(avg_params)):
            self.comm.Allreduce(MPI.IN_PLACE, avg_params[i].numpy(), op=MPI.SUM)

        total_clients = len(client_ids) * self.size
        for v, avg, param in zip(self.velocity, avg_params, self.x.parameters()):
            avg /= total_clients
            v.data = self.beta * v.data + avg.to(self.device)
            param.data = param.data + self.lr * v.data

    def step(self):
        sampled_client_ids = self.sample_clients()
        self.communicate(sampled_client_ids)
        self.update_clients(sampled_client_ids)
        logging.info(f"Process {self.rank}: client_update has completed")
        self.server_update(sampled_client_ids)
        logging.info(f"Process {self.rank}: server_update has completed")

    def train(self):
        self.results = {"loss": [], "accuracy": []}
        for rnd in range(self.num_rounds):
            logging.info(f"\nProcess {self.rank}: Communication Round {rnd+1}")
            self.step()
            test_loss, test_acc = evaluate_fn(self.data, self.x, self.criterion, self.device)

            losses = self.comm.gather(test_loss, root=0)
            accs = self.comm.gather(test_acc, root=0)

            if self.rank == 0:
                avg_loss = sum(losses) / len(losses)
                avg_acc = sum(accs) / len(accs)
                self.results['loss'].append(avg_loss)
                self.results['accuracy'].append(avg_acc)
                logging.info(f"\tLoss: {avg_loss:.4f}   Accuracy: {avg_acc:.2f}%")
