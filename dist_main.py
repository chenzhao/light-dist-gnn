import os
import argparse
import torch

import dist_utils
import dist_gcn_train
import torch.distributed as dist


def process_wrapper(rank, nprocs, backend, func):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_DEBUG']='INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

    env = dist_utils.DistEnv(rank, nprocs, backend)
    func(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--backend", type=str, default='nccl' if torch.cuda.device_count()>1 else 'gloo')
    args = parser.parse_args()
    process_args = (args.nprocs, args.backend, dist_gcn_train.main)
    torch.multiprocessing.spawn(process_wrapper, process_args, args.nprocs)
