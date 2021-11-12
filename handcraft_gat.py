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
    z_loc = torch.zeros_like(local_feature)
    feature_bcast = torch.zeros_like(local_feature)
    
    for src in range(env.world_size):
        if src==env.rank:
            feature_bcast = local_feature.clone()
        # env.barrier_all()
        with env.timer.timing_cuda('broadcast'):
            dist.broadcast(feature_bcast, src=src)

        with env.timer.timing_cuda('spmm'):
            spmm(local_adj_parts[src], feature_bcast, z_loc)
    return z_loc


class DistGCNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_feature, weight, local_adj_parts, tag):
        ctx.save_for_backward(local_feature, weight)
        ctx.local_adj_parts = local_adj_parts
        ctx.tag = tag
        z_local = broadcast(local_adj_parts, local_feature, 'Forward'+tag)
        with DistEnv.env.timer.timing_cuda('mm'):
            z_local = torch.mm(z_local, weight)
        return z_local

    @staticmethod
    def backward(ctx, grad_output):
        local_feature,  weight = ctx.saved_tensors
        ag = broadcast(ctx.local_adj_parts, grad_output, 'Backward'+ctx.tag)
        with DistEnv.env.timer.timing_cuda('mm'):
            grad_feature = torch.mm(ag, weight.t())
            grad_weight = torch.mm(local_feature.t(), ag)
        with DistEnv.env.timer.timing_cuda('all_reduce'):
            DistEnv.env.all_reduce_sum(grad_weight)
        return grad_feature, grad_weight, None, None


class DistMMLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_feature, weight, tag):
        ctx.save_for_backward(local_feature, weight)
        ctx.tag = tag
        Hw = torch.mm(local_feature, weight)
        all_Hw = DistEnv.env.all_gather_then_cat(Hw)
        return all_Hw

    @staticmethod
    def backward(ctx, grad_output):
        local_feature,  weight = ctx.saved_tensors
        split_sz = local_feature.size(0)
        rank = DistEnv.env.rank
        grad_output = grad_output[split_sz*rank:split_sz*(rank+1),:]
        grad_feature = torch.mm(grad_output, weight.t())
        grad_weight = torch.mm(local_feature.t(), grad_output)
        DistEnv.env.all_reduce_sum(grad_weight)
        return grad_feature, grad_weight, None


class GAT(nn.Module):
    def __init__(self, g, env, hidden_dim=16):
        super().__init__()
        self.g, self.env = g, env
        in_dim, out_dim = g.local_features.size(1), g.num_classes
        torch.manual_seed(0)

        self.weight1 = nn.Parameter(torch.rand(in_dim, hidden_dim)).to(env.device)
        self.weight2 = nn.Parameter(torch.rand(hidden_dim, out_dim)).to(env.device)

        self.attention_weight1 = nn.Parameter(torch.rand(2*hidden_dim, 1)).to(env.device)
        self.attention_weight2 = nn.Parameter(torch.rand(out_dim*2, 1)).to(env.device)

    def forward(self, local_features):
        local_edge_index = self.g.local_adj._indices()
        self.env.logger.log('L1', self.weight1.sum(), self.attention_weight1.sum())

        # Hw1 = torch.mm(local_features, self.weight1)
        # all_Hw1 = self.env.all_gather_then_cat(Hw1)
        all_Hw1 = DistMMLayer.apply(local_features, self.weight1, 'L1')

        # Hw_bcast = torch.zeros_like(Hw1)
        # for src in range(self.env.world_size):
        #     if src == self.env.rank:
        #         Hw_bcast = Hw1.clone()
        #     dist.broadcast(Hw_bcast, src=src)

        edge_features = torch.cat((all_Hw1[local_edge_index[0, :], :], all_Hw1[local_edge_index[1, :], :]), dim=1)

        att_input = F.leaky_relu(torch.mm(edge_features, self.attention_weight1).squeeze())
        att_input = torch.sparse_coo_tensor(local_edge_index, att_input, self.g.local_adj.size())
        attention = torch.sparse.softmax(att_input, dim=1)
        # print(attention.size(), Hw1.size())

        hidden_features = torch.sparse.mm(attention, all_Hw1)
        hidden_features = F.elu(hidden_features)


        # self.env.logger.log('L2', self.weight2.sum(), self.attention_weight2.sum())
        # Hw2 = torch.mm(hidden_features, self.weight2)
        # all_Hw2 = self.env.all_gather_then_cat(Hw2)
        all_Hw2 = DistMMLayer.apply(hidden_features, self.weight2, 'L2')
        edge_features = torch.cat((all_Hw2[local_edge_index[0, :], :], all_Hw2[local_edge_index[1, :], :]), dim=1)

        att_input = F.leaky_relu(torch.mm(edge_features, self.attention_weight2).squeeze())
        att_input = torch.sparse_coo_tensor(local_edge_index, att_input, self.g.local_adj.size())
        attention = torch.sparse.softmax(att_input, dim=1)

        outputs = torch.sparse.mm(attention, all_Hw2)
        return F.log_softmax(outputs, 1)

