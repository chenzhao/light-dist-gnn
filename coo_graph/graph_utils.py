import torch
import datetime


def preprocess(name, attr_dict, preprocess_for):  # normalize feature and make adj matrix from edge index
    begin = datetime.datetime.now()
    print(name, preprocess_for, 'preprocess begin', begin)
    attr_dict["features"] = attr_dict["features"] / attr_dict["features"].sum(1, keepdim=True).clamp(min=1)
    if preprocess_for == 'GCN':  # make the coo format sym lap matrix
        attr_dict['adj'] = sym_normalization(attr_dict['edge_index'], attr_dict['num_nodes'])
    elif preprocess_for == 'GAT':
        attr_dict['adj'] = attr_dict['edge_index']
    attr_dict.pop('edge_index')
    print(preprocess_for, 'preprocess done', datetime.datetime.now() - begin)
    return attr_dict


def add_self_loops(edge_index, num_nodes):  # from pyg
    mask = edge_index[0] != edge_index[1]
    all_nodes = torch.arange(0, num_nodes, dtype=edge_index[0].dtype, device=edge_index[0].device)
    return torch.cat([edge_index[:, mask], torch.stack((all_nodes, all_nodes))], dim=1)


def sym_normalization(edge_index, num_nodes, faster_device='cuda:0'):
    if num_nodes>1000000:  # adjust with GPU
        faster_device = 'cpu'
    original_device = edge_index.device
    begin = datetime.datetime.now()
    edge_index = add_self_loops(edge_index, num_nodes)
    A = torch.sparse_coo_tensor(edge_index, torch.ones(len(edge_index[0])), (num_nodes, num_nodes), device=faster_device).coalesce()
    degree_vec = torch.sparse.sum(A, 0).pow(-0.5).to_dense()
    I_edge_index = torch.stack((torch.arange(num_nodes), torch.arange(num_nodes)))
    D_rsqrt = torch.sparse_coo_tensor(I_edge_index, degree_vec, (num_nodes, num_nodes), device=faster_device)
    DA = torch.sparse.mm(D_rsqrt, A)
    del A  # to save GPU mem
    DAD = torch.sparse.mm(DA, D_rsqrt)
    del DA
    end = datetime.datetime.now()
    print('sym norm done',  end - begin)
    return DAD.coalesce().to(original_device)


def sparse_2d_split(st, split_size, split_dim=0):
    seps = list(range(0, st.size(split_dim), split_size)) + [st.size(split_dim)]
    parts = []
    split_idx = st.indices()[split_dim]
    other_idx = st.indices()[1 - split_dim]
    def make_2d_st(idx0, idx1, val, sz0, sz1):
        return  torch.sparse_coo_tensor(torch.stack([idx0, idx1]), val, (sz0, sz1)).coalesce()
    for lower, upper in zip(seps[:-1], seps[1:]):
        mask: torch.Tensor = (split_idx < upper) & (split_idx >= lower)
        if split_dim == 0:
            parts.append(make_2d_st(split_idx[mask]-lower, other_idx[mask], st.values()[mask], upper-lower, st.size(1)))
        else:
            parts.append(make_2d_st(other_idx[mask], split_idx[mask]-lower, st.values()[mask], st.size(0), upper-lower))
    return parts
