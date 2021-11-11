import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_utils import DistEnv
import torch.distributed as dist

try:
    import spmm_cpp
    spmm = lambda A,B,C: spmm_cpp.spmm_cusparse(A.indices()[0].int(), A.indices()[1].int(), A.values(), A.size(0), \
                                                                    A.size(1), B, C, 1, 1)
except ImportError:
    spmm = lambda A,B,C: C.addmm_(A,B)



def broadcast(local_adj_parts, local_feature, tag):
    env = DistEnv.env
    z_loc = torch.zeros((local_adj_parts[0].size(0), local_feature.size(1)), device=env.device)
    feature_recv = torch.zeros((local_adj_parts[0].size(1), local_feature.size(1)), device=env.device)
    
    for i in range(env.world_size):
        p = local_adj_parts[i].coalesce()
        if i == env.rank:
            feature_recv = local_feature.clone()
        elif i==env.world_size-1:
            feature_recv = torch.zeros((local_adj_parts[i].size(1), local_feature.size(1)), device=env.device)
            
        torch.cuda.synchronize()
        # env.barrier_all()

        env.timer.start('broadcast')
        dist.broadcast(feature_recv, src=i)
        torch.cuda.synchronize()
        env.timer.stop('broadcast', tag)

        #env.barrier_all()
        env.timer.start('spmm')
        # spmm_cpp.spmm_cusparse(p.indices()[0].int(), p.indices()[1].int(), p.values(), p.size(0), p.size(1),feature_recv,z_loc, 1,1)
        # z_loc.addmm_(p, feature_recv)
        spmm(p, feature_recv, z_loc)
        torch.cuda.synchronize()
        env.timer.stop('spmm')
    return z_loc


class DistGCNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_feature, weight, local_adj_parts, tag):
        ctx.save_for_backward(local_feature, weight)
        ctx.local_adj_parts = local_adj_parts
        ctx.tag = tag
        z_local = broadcast(local_adj_parts, local_feature, 'Forward'+tag)
        z_local = torch.mm(z_local, weight)
        return z_local

    @staticmethod
    def backward(ctx, grad_output):
        local_feature,  weight = ctx.saved_tensors
        ag = broadcast(ctx.local_adj_parts, grad_output, 'Backward'+ctx.tag)
        grad_feature = torch.mm(ag, weight.t())
        grad_weight = torch.mm(local_feature.t(), ag)
        DistEnv.env.all_reduce_sum(grad_weight)
        return grad_feature, grad_weight, None, None


class GCN(nn.Module):
    def __init__(self, g, env, hidden_dim=16):
        super().__init__()
        self.g, self.env = g, env
        in_dim, out_dim = g.local_features.size(1), g.num_classes
        torch.manual_seed(0)
        self.weight1 = nn.Parameter(torch.rand(in_dim, hidden_dim).to(env.device))
        self.weight2 = nn.Parameter(torch.rand(hidden_dim, out_dim).to(env.device))

    def forward(self, features):
        hidden_features1 = F.relu(DistGCNLayer.apply(features, self.weight1, self.g.local_adj_parts, 'L1'))
        outputs = DistGCNLayer.apply(hidden_features1, self.weight2, self.g.local_adj_parts,  'L2')
        return F.log_softmax(outputs, 1)

