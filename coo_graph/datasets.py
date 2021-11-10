# Copyright 2021, Zhao CHEN
# All rights reserved.

import os
import numpy
import scipy
import scipy.sparse
import torch
import json

data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
dgl_root = os.path.join(data_root, 'dgl_datasets')
pyg_root = os.path.join(data_root, 'pyg_datasets')

def save_dataset(edge_index, features, labels, train_mask, val_mask, test_mask, num_nodes, num_edges, num_classes, name):
    if name.startswith('a_quarter'):
        max_node = num_nodes//4
        smaller_mask = (edge_index[0]<max_node) & (edge_index[1]<max_node)

        edge_index = edge_index[:, smaller_mask].clone()
        features = features[:max_node].clone()
        labels = labels[:max_node].clone()
        train_mask = train_mask[:max_node].clone()
        val_mask = val_mask[:max_node].clone()
        test_mask = test_mask[:max_node].clone()
        num_nodes = max_node
        num_edges = edge_index.size(1)
    path = os.path.join(data_root, name+'.torch')
    torch.save({"edge_index": edge_index, "features": features, "labels": labels,
                "train_mask": train_mask, 'val_mask': val_mask, 'test_mask': test_mask,
                "num_nodes": num_nodes, 'num_edges': num_edges, 'num_classes': num_classes}, path)


def load_dataset(name):
    path = os.path.join(data_root, name+'.torch')
    if not os.path.exists(path):
        prepare_dataset(name)
    return torch.load(path)


def prepare_dgl_dataset(source, name):
    dgl_dataset: dgl.data.DGLDataset = source(raw_dir=dgl_root)
    g = dgl_dataset[0]
    edge_index = torch.stack(g.adj_sparse('coo'))
    save_dataset(edge_index, g.ndata['feat'], g.ndata['label'],
                 g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'],
                 g.num_nodes(), g.num_edges(), dgl_dataset.num_classes, name)


def prepare_pyg_dataset(source, name):
    pyg_dataset: torch_geometric.data.Dataset = source(root=os.path.join(pyg_root, name))
    data: torch_geometric.data.Data = pyg_dataset[0]
    save_dataset(data.edge_index, data.x, data.y,
                 data.val_mask, data.val_mask, data.test_mask,
                 data.num_nodes, data.num_edges, pyg_dataset.num_classes, name)


def prepare_dataset(name):
    import dgl
    import torch_geometric
    dataset_source_mapping = {'cora': dgl.data.CoraGraphDataset,
                              'reddit_reorder': dgl.data.RedditDataset,
                              'reddit': torch_geometric.datasets.Reddit,
                              'a_quarter_reddit': torch_geometric.datasets.Reddit,
                              'flickr': torch_geometric.datasets.Flickr,
                              'yelp': torch_geometric.datasets.Yelp}

    for path in [data_root, dgl_root, pyg_root]:
        os.makedirs(path, exist_ok=True)
    try:
        source_class = dataset_source_mapping[name]
    except KeyError:
        raise Exception('no source for such dataset', name)
    if source_class.__module__.startswith('dgl.'):
        prepare_dgl_dataset(source_class, name)
    elif source_class.__module__.startswith('torch_geometric.'):
        prepare_pyg_dataset(source_class, name)
    else:  # other libs TODO
        pass



def check_edges(edge_index, num_nodes):
    print(f'edges {edge_index[0].size(0)} nodes:{num_nodes}')
    num_parts = 4
    split_size = num_nodes//num_parts
    first_limit = split_size
    last_limit = num_nodes - split_size

    fist_size = (edge_index[0] < first_limit).sum()
    last_size = (edge_index[0] > last_limit).sum()
    print(f'first block {fist_size} last block {last_size}')

    mask_first = (edge_index[0] < first_limit) & (edge_index[1] < first_limit)
    mask_last = (edge_index[0] > last_limit) & (edge_index[1] > last_limit)

    fist_size = edge_index[0][mask_first].size(0)
    last_size = edge_index[0][mask_last].size(0)
    print(f'first p {fist_size} last p {last_size}')


def main():
    r = load_dataset('reddit')
    check_edges(r['edge_index'], r['num_nodes'])

    data = numpy.load(os.path.join(dgl_root, 'reddit', 'reddit_data.npz'))
    x = torch.from_numpy(data['feature']).to(torch.float)
    y = torch.from_numpy(data['label']).to(torch.long)
    # split = torch.from_numpy(data['node_types'])

    adj = scipy.sparse.load_npz(os.path.join(dgl_root, 'reddit', 'reddit_graph.npz'))
    row = torch.from_numpy(adj.row).to(torch.long)
    col = torch.from_numpy(adj.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    check_edges(edge_index, x.size(0))



    return
    for dataset_name in ['cora', 'reddit', 'flickr', 'yelp']:
        prepare_dataset(dataset_name)
    pass


if __name__ == '__main__':
    main()
