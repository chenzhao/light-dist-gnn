import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from torch.nn import init

from dist_utils import DistEnv


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


def broadcast(local_adj_parts, local_feature):
    env = DistEnv.env
    z_loc = torch.zeros((local_adj_parts[0].size(0), local_feature.size(1)), device=env.device)
    for i in range(env.world_size):
        if i == env.rank:
            feature_recv = local_feature
        else:
            feature_recv = torch.zeros((local_adj_parts[i].size(1), local_feature.size(1)), device=env.device)
        env.broadcast(feature_recv, src=i)
        z_loc = torch.addmm(z_loc, local_adj_parts[i], feature_recv)
    return z_loc


class DistGCNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_feature, weight, local_adj_parts, layer):
        ctx.save_for_backward(local_feature, weight)
        ctx.local_adj_parts = local_adj_parts
        ctx.layer = layer
        if torch.cuda.device_count()<=1:
            z_local = broadcast_testing(local_adj_parts, local_feature)
        else:
            z_local = broadcast(local_adj_parts, local_feature)
        z_local = torch.mm(z_local, weight)
        return z_local

    @staticmethod
    def backward(ctx, grad_output):
        local_feature,  weight = ctx.saved_tensors
        ag = broadcast(ctx.local_adj_parts, grad_output)

        grad_feature = torch.mm(ag, weight.t())
        grad_weight = torch.mm(local_feature.t(), ag)
        DistEnv.env.all_reduce_sum(grad_weight)

        return grad_feature, grad_weight, None, None


class GCN(nn.Module):
    def __init__(self, g, env):
        super().__init__()
        self.g, self.env = g, env
        in_dim = g.local_features.size(1)
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
