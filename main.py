import os
import json
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI
from datetime import datetime
from src.util_functions import set_logger, save_plt
import importlib


def run_fl(Server, global_config, data_config, fed_config, model_config, comm, rank, size, run_id):
    # Create log directories only on root
    if rank == 0:
        log_dir = f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/{run_id}/"
        os.makedirs(log_dir, exist_ok=True)

    comm.Barrier()

    # Set logger per rank with unique run ID
    log_filename = f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/{run_id}/log_rank{rank}.txt"
    set_logger(log_filename)
    logging.info(f"Process {rank} is initializing the server")

    # Initialize server
    server = Server(model_config, global_config, data_config, fed_config, comm=comm, rank=rank, size=size)
    logging.info(f"Process {rank}: Server is successfully initialized")

    # Setup clients and start training
    server.setup()
    server.train()

    # Save plots on root only
    if rank == 0:
        save_plt(list(range(1, server.num_rounds + 1)), server.results['accuracy'],
                 "Communication Round", "Test Accuracy",
                 f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/{run_id}/accgraph.png")
        save_plt(list(range(1, server.num_rounds + 1)), server.results['loss'],
                 "Communication Round", "Test Loss",
                 f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/{run_id}/lossgraph.png")
        logging.info("Plots saved successfully")

    logging.info(f"Process {rank}: Execution has completed")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load config on root
    if rank == 0:
        with open('config.json', 'r') as f:
            config = json.load(f)
    else:
        config = None

    # Broadcast config to all ranks
    config = comm.bcast(config, root=0)

    global_config = config["global_config"]
    data_config = config["data_config"]
    fed_config = config["fed_config"]
    model_config = config["model_config"]

    # Generate unique run ID using timestamp
    if rank == 0:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = None
    run_id = comm.bcast(run_id, root=0)

    # Dynamically import Server
    module_name = f"src.algorithms.{fed_config['algorithm']}.server"
    server_module = importlib.import_module(module_name)
    Server = server_module.Server

    # Run federated learning with the unique run ID
    run_fl(Server, global_config, data_config, fed_config, model_config, comm, rank, size, run_id)