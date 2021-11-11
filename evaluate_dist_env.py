# Copyright 2021, Zhao CHEN
# All rights reserved.

import os
import datetime as dt
import math
import torch
import torch.distributed as dist
from dist_utils import DistEnv
from torch.multiprocessing import Process


def batch_bcast(env, sz_tag, size, repeat):
    dtype = torch.int8
    # dtype = torch.float
    env.logger.log(f'Size: {sz_tag: <5} Repeat: {repeat}',  rank=0)
    data = torch.ones(size, dtype=dtype, device=env.device)
    recv = torch.zeros(size, dtype=dtype, device=env.device)
    tag = f'{env.backend}_{env.world_size}_broadcast'
    for i in range(repeat):
        torch.cuda.synchronize()
        env.timer.start(tag)
        for src in range(env.world_size):
            buf = data if env.rank == src else recv
            env.broadcast(tensor=buf, src=src)
        torch.cuda.synchronize()
        env.logger.log(f'{sz_tag} {i+1}/{repeat}', oneline=True, rank=0)
        env.timer.stop(tag, sz_tag)


def eval_broadcast(env):
    # sizes = [ (160,'L1',[29121, 602]),
            # (160,'L2',[29121, 16]), ]
    sizes = [(16000, '4K', (4,1024)), 
            (8000, '16K', (16, 1024)), 
            (8000, '64K', (64, 1024)),  (2000, '256K', (256, 1024)), 
            (1000, '1M', (1, 1024, 1024)), (256, '4M', (4, 1024, 1024)), (64, '16M', (16, 1024, 1024)),
            (32,'64M', (64, 1024, 1024)), (8, '256M', (256, 1024, 1024)), (2, '1G', (1024, 1024, 1024)), (1, '2G', (1, 1024, 1024, 1024))]
    for repeat, tag, size in sizes:
        batch_bcast(env, tag, size, repeat)


def evaluate(rank, nprocs, backend):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # os.environ['NCCL_DEBUG']='INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    env = DistEnv(rank, nprocs, backend)
    env.timer.start('total')
    eval_broadcast(env)
    env.timer.stop('total')
    env.logger.log(env.timer.summary_all(), rank=0)


if __name__ == "__main__":
    num_GPUs = torch.cuda.device_count()
    nprocs = num_GPUs if num_GPUs>1 else 4
    backend = 'nccl' if num_GPUs>1 else 'gloo'
    torch.multiprocessing.spawn(evaluate, (nprocs, backend), nprocs)
