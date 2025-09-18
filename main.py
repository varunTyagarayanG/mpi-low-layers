import os
import json
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI
from datetime import datetime
import argparse
from src.util_functions import set_logger, save_plt
import importlib
import time  # 🔹 added for timing

def run_fl(Server, global_config, data_config, fed_config, model_config, comm, rank, size, run_id):
    # Create log directories only on root
    if rank == 0:
        log_dir = f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/{run_id}/"
        os.makedirs(log_dir, exist_ok=True)

    # Barrier to ensure directory exists before other ranks start logging
    comm.Barrier()

    # Set logger per rank with unique run ID
    log_filename = f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/{run_id}/log_rank{rank}.txt"
    set_logger(log_filename)
    logging.info(f"Process {rank} is initializing the server")

    # Initialize server
    server = Server(model_config=model_config, global_config=global_config,
                    data_config=data_config, fed_config=fed_config,
                    comm=comm, rank=rank, size=size)
    logging.info(f"Process {rank}: Server is successfully initialized")

    # 🔹 Measure total training time
    start_time = time.time()
    server.setup()
    server.train()
    total_time = time.time() - start_time

    # Gather timing on root for visibility
    all_times = comm.gather(total_time, root=0)
    if rank == 0:
        logging.info(f"Total wall time per rank (s): {all_times}")

    # Evaluate on root only if your evaluate uses shared model; otherwise broadcast as needed
    if rank == 0:
        logging.info("Training complete on all ranks.")

if __name__ == "__main__":
    # Init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse CLI only on root; broadcast to others
    if rank == 0:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="config.json",
                            help="Path to config JSON")
        parser.add_argument("--num-clients", type=int, required=True,
                            help="Total number of FL clients (must equal MPI world size when using 1 client per process).")
        args, _ = parser.parse_known_args()
    else:
        args = None
    args = comm.bcast(args, root=0)

    # Load config on root
    if rank == 0:
        with open(args.config, "r") as f:
            config = json.load(f)
        global_config = config["global_config"]
        data_config   = config["data_config"]
        fed_config    = config["fed_config"]
        model_config  = config["model_config"]

        # Enforce 1 client per process: num_clients == world size
        if args.num_clients != size:
            raise ValueError(f"--num-clients ({args.num_clients}) must equal MPI world size (-n == {size}) when running 1 client per process.")
        fed_config['num_clients'] = args.num_clients

        # Module/algorithm loader (expects e.g., src.algorithms.FedAvg.server:Server)
        algo_name = fed_config["algorithm"]
        server_module = importlib.import_module(f"src.algorithms.{algo_name}.server")
        Server = getattr(server_module, "Server")

        # Generate unique run ID using timestamp
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        global_config = None
        data_config = None
        fed_config = None
        model_config = None
        Server = None
        run_id = None

    # Broadcast loaded objects
    global_config = comm.bcast(global_config, root=0)
    data_config   = comm.bcast(data_config, root=0)
    fed_config    = comm.bcast(fed_config, root=0)
    model_config  = comm.bcast(model_config, root=0)
    Server        = comm.bcast(Server, root=0)
    run_id        = comm.bcast(run_id, root=0)

    # Run
    run_fl(Server, global_config, data_config, fed_config, model_config, comm, rank, size, run_id)