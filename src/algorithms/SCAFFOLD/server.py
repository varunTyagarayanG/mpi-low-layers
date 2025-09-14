# import numpy as np
# import logging
# import torch
# from copy import deepcopy

# from .client import Client
# from src.models import *
# from src.load_data_for_clients import dist_data_per_client
# from src.util_functions import set_seed, evaluate_fn

# class Server:
#     def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={}):
#         set_seed(global_config["seed"])
#         self.device = global_config["device"]

#         self.data_path = data_config["dataset_path"]
#         self.dataset_name = data_config["dataset_name"]
#         self.non_iid_per = data_config["non_iid_per"]

#         self.fraction = fed_config["fraction_clients"]
#         self.num_clients = fed_config["num_clients"]
#         self.num_rounds = fed_config["num_rounds"]
#         self.num_epochs = fed_config["num_epochs"]
#         self.batch_size = fed_config["batch_size"]
#         self.criterion = eval(fed_config["criterion"])()
#         self.lr = fed_config["global_stepsize"]
#         self.lr_l = fed_config["local_stepsize"]

#         self.x = eval(model_config["name"])()
#         self.server_c = [torch.zeros_like(param, device=self.device) for param in self.x.parameters()]
#         self.clients = None

#     def create_clients(self, local_datasets):
#         clients = []
#         for id_num, dataset in enumerate(local_datasets):
#             client = Client(client_id=id_num, local_data=dataset, device=self.device, num_epochs=self.num_epochs,
#                             criterion=self.criterion, lr=self.lr_l, client_c=deepcopy(self.server_c))
#             clients.append(client)
#         return clients

#     def setup(self, **init_kwargs):
#         """Initializes all clients and splits the dataset"""
#         local_datasets, test_dataset = dist_data_per_client(
#             self.data_path, self.dataset_name, self.num_clients,
#             self.batch_size, self.non_iid_per, self.device
#         )
#         self.data = test_dataset
#         self.clients = self.create_clients(local_datasets)
#         logging.info("\nClients are successfully initialized")

#     def sample_clients(self):
#         """Selects a fraction of clients from all the available clients"""
#         num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
#         sampled_client_ids = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients,
#                                                      replace=False).tolist())
#         return sampled_client_ids

#     def communicate(self, client_ids):
#         """Communicates global model and server control variate to the selected clients"""
#         for idx in client_ids:
#             self.clients[idx].x = deepcopy(self.x)
#             self.clients[idx].server_c = deepcopy(self.server_c)

#     def update_clients(self, client_ids):
#         """Tells all the clients to perform client_update"""
#         for idx in client_ids:
#             self.clients[idx].client_update()

#     def server_update(self, client_ids):
#         """Updates the global model and server control variate by averaging updates"""
#         # Accumulate updates
#         for idx in client_ids:
#             with torch.no_grad():
#                 # Update global model
#                 for param, diff in zip(self.x.parameters(), self.clients[idx].delta_y):
#                     param.data.add_(diff.data * self.lr / len(client_ids))

#                 # Update server control variate
#                 for c_g, c_d in zip(self.server_c, self.clients[idx].delta_c):
#                     c_g.data.add_(c_d.data * self.fraction)

#     def step(self):
#         """Performs single round of training"""
#         sampled_client_ids = self.sample_clients()
#         self.communicate(sampled_client_ids)
#         self.update_clients(sampled_client_ids)
#         logging.info("\tclient_update has completed")
#         self.server_update(sampled_client_ids)
#         logging.info("\tserver_update has completed")

#     def train(self):
#         """Performs multiple rounds of training"""
#         self.results = {"loss": [], "accuracy": []}
#         for rnd in range(self.num_rounds):
#             logging.info(f"\nCommunication Round {rnd + 1}")
#             self.step()
#             test_loss, test_acc = evaluate_fn(self.data, self.x, self.criterion, self.device)
#             self.results['loss'].append(test_loss)
#             self.results['accuracy'].append(test_acc)
#             logging.info(f"\tLoss: {test_loss:.4f}   Accuracy: {test_acc:.2f}%")


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

        self.x = eval(model_config["name"])()
        self.server_c = [torch.zeros_like(param, device=self.device) for param in self.x.parameters()]

        self.clients = None

        # MPI attributes
        self.comm = comm if comm else MPI.COMM_WORLD
        self.rank = rank
        self.size = size

    def create_clients(self, local_datasets):
        clients = []
        for id_num, dataset in enumerate(local_datasets):
            client = Client(client_id=id_num, local_data=dataset, device=self.device, num_epochs=self.num_epochs,
                            criterion=self.criterion, lr=self.lr_l, client_c=deepcopy(self.server_c))
            clients.append(client)
        return clients

    def setup(self, **init_kwargs):
        """Initializes all clients and splits the dataset"""
        if self.rank == 0:
            local_datasets, test_dataset = dist_data_per_client(
                self.data_path, self.dataset_name, self.num_clients,
                self.batch_size, self.non_iid_per, self.device
            )
        else:
            local_datasets = None
            test_dataset = None

        # Broadcast the test dataset to all processes
        test_dataset = self.comm.bcast(test_dataset, root=0)
        self.data = test_dataset

        # Split local datasets among all ranks
        if self.rank == 0:
            datasets_per_rank = np.array_split(local_datasets, self.size)
        else:
            datasets_per_rank = None

        # Scatter the datasets to all ranks
        local_dataset = self.comm.scatter(datasets_per_rank, root=0)

        self.clients = self.create_clients(local_dataset)
        logging.info(f"Process {self.rank}: Clients are successfully initialized")

    def sample_clients(self):
        """Selects a fraction of clients from all the available clients"""
        num_sampled_clients = max(int(self.fraction * len(self.clients)), 1)
        sampled_client_ids = sorted(np.random.choice(a=[i for i in range(len(self.clients))], size=num_sampled_clients,
                                                     replace=False).tolist())
        return sampled_client_ids

    def communicate(self, client_ids):
        """Communicates global model and server control variate to the selected clients"""
        for idx in client_ids:
            self.clients[idx].x = deepcopy(self.x)
            self.clients[idx].server_c = deepcopy(self.server_c)

    def update_clients(self, client_ids):
        """Tells all the clients to perform client_update"""
        for idx in client_ids:
            self.clients[idx].client_update()

    def server_update(self, client_ids):
        """Updates the global model and server's control variate using MPI Allreduce"""
        # Prepare accumulators for model and control variate updates
        avg_params = [torch.zeros_like(param.data, device='cpu') for param in self.x.parameters()]
        avg_server_c = [torch.zeros_like(c.data, device='cpu') for c in self.server_c]

        # Sum updates from local clients
        for idx in client_ids:
            with torch.no_grad():
                for avg, param, diff in zip(avg_params, self.x.parameters(), self.clients[idx].delta_y):
                    avg.add_(diff.detach().cpu())
                for avg_c, c in zip(avg_server_c, self.clients[idx].delta_c):
                    avg_c.add_(c.detach().cpu())

        # Reduce across all processes
        for i in range(len(avg_params)):
            self.comm.Allreduce(MPI.IN_PLACE, avg_params[i].numpy(), op=MPI.SUM)
        for i in range(len(avg_server_c)):
            self.comm.Allreduce(MPI.IN_PLACE, avg_server_c[i].numpy(), op=MPI.SUM)

        # Apply averaged updates
        total_clients = len(client_ids) * self.size
        for param, avg in zip(self.x.parameters(), avg_params):
            param.data.add_(avg / total_clients * self.lr)
        for c_g, avg_c in zip(self.server_c, avg_server_c):
            c_g.data.add_(avg_c / total_clients * self.size * self.fraction)

    def step(self):
        """Performs a single round of training"""
        sampled_client_ids = self.sample_clients()
        self.communicate(sampled_client_ids)
        self.update_clients(sampled_client_ids)
        logging.info(f"Process {self.rank}: client_update has completed")
        self.server_update(sampled_client_ids)
        logging.info(f"Process {self.rank}: server_update has completed")

    def train(self):
        """Performs multiple rounds of training"""
        self.results = {"loss": [], "accuracy": []}
        for rnd in range(self.num_rounds):
            logging.info(f"\nProcess {self.rank}: Communication Round {rnd + 1}")
            self.step()
            test_loss, test_acc = evaluate_fn(self.data, self.x, self.criterion, self.device)

            # Gather test results at the root process
            all_losses = self.comm.gather(test_loss, root=0)
            all_accs = self.comm.gather(test_acc, root=0)

            if self.rank == 0:
                avg_loss = sum(all_losses) / len(all_losses)
                avg_acc = sum(all_accs) / len(all_accs)
                self.results['loss'].append(avg_loss)
                self.results['accuracy'].append(avg_acc)
                logging.info(f"\tLoss: {avg_loss:.4f}   Accuracy: {avg_acc:.2f}%")
