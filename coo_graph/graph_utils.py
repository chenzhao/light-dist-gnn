import os
import os.path
import torch
import math
import datetime

import random
import numpy as np
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()


def add_self_loops(edge_index, num_nodes):  # from pyg
    mask = edge_index[0] != edge_index[1]
    all_nodes = torch.arange(0, num_nodes, dtype=edge_index[0].dtype, device=edge_index[0].device)
    return torch.cat([edge_index[:, mask], torch.stack((all_nodes, all_nodes))], dim=1)


def sym_normalization(edge_index, num_nodes, faster_device='cuda:0'):
    original_device = edge_index.device
    # begin = datetime.datetime.now()
    edge_index = add_self_loops(edge_index, num_nodes)
    # A = torch.sparse_coo_tensor(edge_index, torch.ones(len(edge_index[0])), (num_nodes, num_nodes)).coalesce()
    # return A
    A = torch.sparse_coo_tensor(edge_index, torch.ones(len(edge_index[0])), (num_nodes, num_nodes), device=faster_device).coalesce()
    degree_vec = torch.sparse.sum(A, 0).pow(-0.5).to_dense()
    I_edge_index = torch.stack((torch.arange(num_nodes), torch.arange(num_nodes)))
    D_rsqrt = torch.sparse_coo_tensor(I_edge_index, degree_vec, (num_nodes, num_nodes), device=faster_device)
    DA = torch.sparse.mm(D_rsqrt, A)
    del A  # to save GPU mem
    # print(DA)
    DAD = torch.sparse.mm(DA, D_rsqrt)
    del DA
    # end = datetime.datetime.now()
    # print('sym norm done',  end - begin)
    return DAD.coalesce().to(original_device)




def save_cache_dict(d, path):
    if os.path.exists(path):
        print(f'warning: cache file {path} is overwritten.')
    torch.save(d, path)


def load_cache_dict(path):
    if not os.path.exists(path):
        raise Exception('no such file: '+path)
    d = torch.load(path)
    updated_d = {}
    for k,v in d.items():
        if type(v) == torch.Tensor and v.is_sparse:
            updated_d[k] = v.coalesce()
        if type(v) == list and type(v[0]) == torch.Tensor and v[0].is_sparse:
            updated_d[k] = [i.coalesce() for i in v]
    d.update(updated_d)
    return d

def split_2D_coo_by_size(split_idx, other_idx, values, split_size):  # 2D tensors only
    coo_parts = []
    while len(split_idx) > 0:
        mask: torch.Tensor = split_idx < split_size
        coo_parts.append( (split_idx[mask], other_idx[mask], values[mask], split_size) )  # padding? TODO
        split_idx = split_idx[mask.logical_not()] - split_size
    return coo_parts


def split_2D_coo(split_idx, other_idx, values, seps):  # 2D tensors only
    coo_parts = []
    for lower, upper in zip(seps[:-1], seps[1:]):
        mask: torch.Tensor = (split_idx < upper) & (split_idx >= lower)
        coo_parts.append( (split_idx[mask]-lower, other_idx[mask], values[mask], upper-lower) )
    return coo_parts


def make_2D_coo(idx0, idx1, val, sz0, sz1):  # to mat: row is 0, col is 1
    return torch.sparse_coo_tensor(torch.stack([idx0, idx1]), val, (sz0, sz1)).coalesce()


def CAGNET_split(coo_adj, split_size):  # A.T==A only
    seps = list(range(0, coo_adj.size(0), split_size))+[coo_adj.size(0)]
    row_parts = split_2D_coo(coo_adj.indices()[0], coo_adj.indices()[1], coo_adj.values(), seps)  # Ai is rows part i
    row_part_coo_list, row_col_part_coos_list = [], []
    for part_row_idx, full_col_idx, val, row_sz in row_parts:
        print(f'coo split: {val.size(0)}, {row_sz}')
        row_part_coo_list.append(make_2D_coo(part_row_idx, full_col_idx, val, row_sz, coo_adj.size(0)))
        # row_col_part_coos_list.append( [make_2D_coo(p_row, p_col, p_val, row_sz, col_sz) \
        #         for p_col, p_row, p_val, col_sz in split_2D_coo(full_col_idx, part_row_idx, val, seps)])
        row_col_part_coos = []
        for p_col, p_row, p_val, col_sz in split_2D_coo(full_col_idx, part_row_idx, val, seps):
            print(f'\tcoo split: {p_val.size(0)}, {row_sz}, {col_sz}')
            row_col_part_coos.append(make_2D_coo(p_row, p_col, p_val, row_sz, col_sz))
        row_col_part_coos_list.append(row_col_part_coos)
    return row_part_coo_list, row_col_part_coos_list
