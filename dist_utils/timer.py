import datetime as dt
import torch
import math
import time
import pickle
import statistics
from collections import defaultdict


class TimerCtx:
    def __init__(self, timer, key, cuda):
        self.cuda = cuda
        self.timer = timer
        self.key = key

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.timer.start_time_dict[self.key] = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if self.cuda:
            torch.cuda.synchronize()
        d = time.time() - self.timer.start_time_dict[self.key]
        self.timer.duration_dict[self.key] += d
        self.timer.count_dict[self.key] += 1


class DistTimer:
    def __init__(self, env):
        self.env = env
        self.start_time_dict = {}
        self.duration_dict = defaultdict(float)
        self.count_dict = defaultdict(int)

    def summary(self):
        s = '\ntimer summary:\n' +  "\n".join("%6.2fs %5d %s" % (self.duration_dict[key], self.count_dict[key], key) for key in self.duration_dict)
        return s

    def sync_duration_dicts(self):
        self.env.store.set('duration_dict_%d'%self.env.rank, pickle.dumps(self.duration_dict))
        self.env.barrier_all()
        self.all_durations = [pickle.loads(self.env.store.get('duration_dict_%d'%rank)) for rank in range(self.env.world_size)]

    def summary_all(self):
        self.sync_duration_dicts()
        avg_dict = {}
        std_dict = {}
        for key in self.duration_dict:
            data = [d[key] for d in self.all_durations]
            avg_dict[key], std_dict[key] = statistics.mean(data), statistics.stdev(data)
        s = '\ntimer summary:\n' +  "\n".join("%6.2fs %6.2fs %5d %s" % (avg_dict[key], std_dict[key], self.count_dict[key], key) for key in self.duration_dict)
        return s

    def detail_all(self):
        self.sync_duration_dicts()
        avg_dict = {}
        std_dict = {}
        detail_dict = {}
        for key in self.duration_dict:
            data = [d[key] for d in self.all_durations]
            avg_dict[key], std_dict[key] = statistics.mean(data), statistics.stdev(data)
            detail_dict[key] = ' '.join("%6.2f"%x for x in data)
        s = '\ntimer summary:\n' +  "\n".join("%6.2fs %6.2fs %5d %s \ndetail: %s \n--------------" % (avg_dict[key], std_dict[key], self.count_dict[key], key, detail_dict[key]) for key in self.duration_dict)
        return s

    def timing(self, key):
        return TimerCtx(self, key, cuda=False)

    def timing_cuda(self, key):
        return TimerCtx(self, key, cuda=True)

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

