import os
import torch
import torch.distributed as dist
import tempfile

from .timer import DistTimer
from .logger import DistLogger


class DistEnv:
    def __init__(self, rank, world_size, backend='nccl'):
        assert(rank>=0)
        assert(world_size>0)
        self.rank, self.world_size = rank, world_size
        self.backend = backend
        self.init_device()
        self.init_dist_groups()
        self.logger = DistLogger(self)
        self.timer = DistTimer(self)
        self.store = dist.FileStore(os.path.join(tempfile.gettempdir(), 'torch-dist'), self.world_size)
        DistEnv.env = self  # no global...

    def __repr__(self):
        return '<DistEnv %d/%d %s>'%(self.rank, self.world_size, self.backend)

    def init_device(self):
        if torch.cuda.device_count()>1:
            self.device = torch.device('cuda', self.rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

    def all_reduce_sum(self, tensor):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.world_group)

    def broadcast(self, tensor, src):
        dist.broadcast(tensor, src=src, group=self.world_group)

    def all_gather_then_cat(self, src_t):
        recv_list = [torch.zeros_like(src_t) for _ in range(self.world_size)]
        dist.all_gather(recv_list, src_t, group=self.world_group)
        return torch.cat(recv_list, dim=0)

    def barrier_all(self):
        dist.barrier(self.world_group)

    def init_dist_groups(self):
        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size, init_method='env://')
        self.world_group = dist.new_group(list(range(self.world_size)))
        self.p2p_group_dict = {}
        for src in range(self.world_size):
            for dst in range(src+1, self.world_size):
                self.p2p_group_dict[(src, dst)] = dist.new_group([src, dst])
                self.p2p_group_dict[(dst, src)] = self.p2p_group_dict[(src, dst)]


class DistUtil:
    def __init__(self, env):
        self.env = env


if __name__ == '__main__':
    pass

