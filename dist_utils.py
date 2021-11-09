import os
import datetime as dt
import torch
import torch.distributed as dist
import math
import time
import pickle
import statistics
from collections import defaultdict
import tempfile


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

    def all_gather(self, recv_list, src_t):
        dist.all_gather(recv_list, src_t, group=self.world_group)

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
        # print('dist groups inited')


class DistUtil:
    def __init__(self, env):
        self.env = env


class DistLogger(DistUtil):
    def log(self, *args, oneline=False, rank=-1):
        if rank!=-1 and self.env.rank!=rank:
            return
        head = '%s [%1d] '%(dt.datetime.now().time(), self.env.rank)
        tail = '\r' if oneline else '\n'
        the_whole_line = head+' '.join(map(str, args))+tail
        print(the_whole_line, end='', flush=True)  # to prevent line breaking
        with open('all_log_%d.txt'%self.env.rank, 'a+') as f:
            print(the_whole_line, end='', file=f, flush=True)  # to prevent line breaking


class DistTimer(DistUtil):
    def __init__(self, env):
        super().__init__(env)
        self.start_time_dict = {}
        self.duration_dict = defaultdict(float)
        self.count_dict = defaultdict(int)

    def summary(self):
        s = '\ntimer summary:\n' +  "\n".join("%6.2fs %5d %s" % (self.duration_dict[key], self.count_dict[key], key) for key in self.duration_dict)
        return s

    def sync_duration_dicts(self):
        self.env.store.set('duration_dict_%d'%self.env.rank, pickle.dumps(self.duration_dict))
        self.env.barrier_all()
        self.all_durations = [pickle.loads(self.env.store.get('duration_dict_%d'%rank)) for rank in range(self.world_size)]

    def summary_all(self):
        self.sync_duration_dicts()
        avg_dict = {}
        std_dict = {}
        for key in self.duration_dict:
            data = [d[key] for d in self.all_durations]
            avg_dict[key], std_dict[key] = statistics.mean(data), statistics.stdev(data)
        s = '\ntimer summary:\n' +  "\n".join("%6.2fs %6.2fs %5d %s" % (avg_dict[key], std_dict[key], self.count_dict[key], key) for key in self.duration_dict)
        return s

    def barrier_all(self):
        return
        self.start('barrier')
        self.env.barrier_all()
        self.stop('barrier')

    def start(self, key):
        self.start_time_dict[key] = time.time()
        return self.start_time_dict[key]

    def stop(self, key, *other_keys):
        def log(k, d=time.time() - self.start_time_dict[key]):
            self.duration_dict[k]+=d
            self.count_dict[k]+=1
        log(key)
        for subkey in other_keys:
            log(key+'-'+subkey)
        return


if __name__ == '__main__':
    pass

