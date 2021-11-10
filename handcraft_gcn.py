import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from torch.nn import init

from dist_utils import DistEnv
import torch.distributed as dist


def broadcast_testing(local_adj_parts, local_feature):
    env = DistEnv.env
    device = 'cuda:0'
    z_loc = torch.zeros((local_adj_parts[0].size(0), local_feature.size(1)), device=device)
    # z_loc = torch.zeros((local_feature.size(1), local_adj_parts[0].size(0), ), device=env.device)

    for i in range(env.world_size):
        if i == env.rank:
            # feature_recv = local_feature.clone()
            feature_recv = local_feature.to(device)
        else:
            feature_recv = torch.zeros((local_adj_parts[i].size(1), local_feature.size(1)), device=device)
        env.broadcast(feature_recv, src=i)
        device_feature = feature_recv
        device_adj = local_adj_parts[i].to(device)
        z_loc = torch.addmm(z_loc, device_adj, device_feature)
    return z_loc.to('cpu')



def broad_func(node_count, am_partitions, inputs, btype=None):
    print('new broad')
    g_env = DistEnv.env
    g_timer = g_env.timer
    g_logger = g_env.logger
    device = g_env.device
    n_per_proc = math.ceil(float(node_count) / g_env.world_size)
    z_loc = torch.zeros((am_partitions[0].size(0), inputs.size(1)), device=device)
    inputs_recv = torch.zeros((n_per_proc, inputs.size(1)), device=device)

    for i in range(g_env.world_size):
        if i == g_env.rank:
            inputs_recv = inputs.clone()
        elif i == g_env.world_size - 1:
            inputs_recv = torch.zeros((am_partitions[i].size(1), inputs.size(1)), device=device)

        g_timer.barrier_all()
        torch.cuda.synchronize()

        g_timer.start('broadcast')
        g_logger.log(f'{i} {inputs_recv.size()}, {inputs_recv.dtype}, {inputs_recv.device}, {inputs_recv.sum()}', rank=0)
        dist.broadcast(inputs_recv, src=i, group=g_env.world_group)
        torch.cuda.synchronize()  # comm or comp?
        # g_logger.log(f'{g_env.rank}->{i} {inputs_recv.size()}, {inputs_recv.dtype}, {inputs_recv.device} done')
        g_timer.stop('broadcast', btype)#,'comm')

        g_timer.barrier_all()
        torch.cuda.synchronize()

        g_timer.start('spmm')

        z_loc = torch.addmm(z_loc, am_partitions[i], inputs_recv)
        #spmm_gpu(am_partitions[i].indices()[0].int(), am_partitions[i].indices()[1].int(), 
                        #am_partitions[i].values(), am_partitions[i].size(0), 
                        #am_partitions[i].size(1), inputs_recv, z_loc)

        torch.cuda.synchronize()
        g_timer.stop('spmm')#, 'comp')
        g_timer.barrier_all()
    return z_loc


def broadcast(local_adj_parts, local_feature, tag):
    # return broad_func(g_node_num, local_adj_parts, local_feature, tag)
    if torch.cuda.device_count()<=1:
        return broadcast_testing(local_adj_parts, local_feature)
    env = DistEnv.env
    z_loc = torch.zeros((local_adj_parts[0].size(0), local_feature.size(1)), device=env.device)
    feature_recv = torch.zeros((local_adj_parts[0].size(1), local_feature.size(1)), device=env.device)
    
    for i in range(env.world_size):
        if i == env.rank:
            feature_recv = local_feature.clone()
        elif i==env.world_size-1:
            feature_recv = torch.zeros((local_adj_parts[i].size(1), local_feature.size(1)), device=env.device)
            
        env.barrier_all()
        torch.cuda.synchronize()
        env.timer.start('broadcast')
        env.logger.log(f'{feature_recv.size()}, {feature_recv.dtype}, {feature_recv.device}, {feature_recv.sum()}', rank=0)
        # env.broadcast(feature_recv, src=i)
        dist.broadcast(feature_recv, src=i)
        torch.cuda.synchronize()
        env.barrier_all()
        env.timer.stop('broadcast', tag)


        env.timer.start('spmm')
        z_loc = torch.addmm(z_loc, local_adj_parts[i], feature_recv)
        torch.cuda.synchronize()
        env.timer.stop('spmm')
    return z_loc


class DistGCNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_feature, weight, local_adj_parts, layer):
        ctx.save_for_backward(local_feature, weight)
        ctx.local_adj_parts = local_adj_parts
        ctx.layer = layer
        tag = 'fwd'+layer
        z_local = broadcast(local_adj_parts, local_feature, tag)
        z_local = torch.mm(z_local, weight)
        return z_local

    @staticmethod
    def backward(ctx, grad_output):
        local_feature,  weight = ctx.saved_tensors
        layer = ctx.layer
        tag = 'bkd'+layer
        ag = broadcast(ctx.local_adj_parts, grad_output, tag)

        grad_feature = torch.mm(ag, weight.t())
        grad_weight = torch.mm(local_feature.t(), ag)
        DistEnv.env.all_reduce_sum(grad_weight)

        return grad_feature, grad_weight, None, None


g_node_num =0


class GCN(nn.Module):
    def __init__(self, g, env):
        global g_node_num
        super().__init__()
        self.g, self.env = g, env
        in_dim = g.local_features.size(1)
        g_node_num = g.num_nodes
        hidden_dim = 16
        out_dim = g.num_classes
        torch.manual_seed(0)
        self.weight1 = nn.Parameter(torch.rand(in_dim, hidden_dim).to(env.device))
        self.weight2 = nn.Parameter(torch.rand(hidden_dim, out_dim).to(env.device))
        # init.xavier_uniform_(self.weight1)
        # init.xavier_uniform_(self.weight2)

    def forward(self):
        hidden_features1 = F.relu(DistGCNLayer.apply(self.g.local_features, self.weight1, self.g.local_adj_parts, 'L1'))
        outputs = DistGCNLayer.apply(hidden_features1, self.weight2, self.g.local_adj_parts,  'L2')
        return F.log_softmax(outputs, 1)
