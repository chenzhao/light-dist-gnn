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
    data = torch.rand(size, dtype=torch.float32, device=env.device)
    recv = torch.zeros(size, dtype=torch.float32, device=env.device)
    tag = f'{env.backend}_{env.world_size}_broadcast'
    for i in range(repeat):
        env.timer.start(tag)
        for src in range(env.world_size):
            buf = data if env.rank == src else recv
            env.broadcast(tensor=buf, src=src)
        torch.cuda.synchronize()
        env.logger.log(f'{sz_tag} {i+1}/{repeat}', rank=0)
        env.timer.stop(tag, sz_tag)


def eval_broadcast(env):
    small_size = (10, 1024, 1024)
    middle_size = (100, 1024, 1024)
    large_size = (1000, 1024, 1024)
    repeat = 10
    batch_bcast(env, 'small broadcast', small_size, repeat)
    batch_bcast(env, 'middle broadcast', middle_size, repeat//2)
    batch_bcast(env, 'large broadcast', large_size, repeat//4)


def evaluate(rank, nprocs, backend):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # os.environ['NCCL_DEBUG']='INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    env = DistEnv(rank, nprocs, backend)
    env.timer.start('total')
    eval_broadcast(env)
    env.timer.stop('total')
    env.logger.log(env.timer.summary(), rank=0)


if __name__ == "__main__":
    nprocs = 8
    backend = 'gloo'
    backend = 'nccl'
    torch.multiprocessing.spawn(evaluate, (nprocs, backend), nprocs)
