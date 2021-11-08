import datetime
import os.path
import torch
import torch.sparse
from . import graph_utils
from . import datasets


class Parted_COO_Graph():
    def full_graph_path(self, root=datasets.data_root):
        return os.path.join(root, f'{self.name}_{self.preprocess_for}_full.coo_graph')

    def parted_graph_path(self, rank, total_parted, root=datasets.data_root):
        dirpath = os.path.join(root, f'{self.name}_{self.preprocess_for}_{total_parted}_parts')
        os.makedirs(dirpath, exist_ok=True)
        return os.path.join(dirpath, f'part_{rank}_of_{total_parted}.coo_graph')

    @property
    def split_size(self):
        assert self.num_parts > 1
        return (self.num_nodes+self.num_parts-1)//self.num_parts

    def __init__(self, name, preprocess_for='GCN', rank=-1, num_parts=-1, full_graph_cache_enabled=True):
        self.name, self.preprocess_for, self.rank, self.num_parts = name, preprocess_for, rank, num_parts
        if rank != -1:
            if not os.path.exists(self.parted_graph_path(rank, num_parts)):
                raise Exception('Not parted yet. Run COO_Graph.partition() first.', self.parted_graph_path(rank, num_parts))
            cached_attr_dict = graph_utils.load_cache_dict(self.parted_graph_path(rank, num_parts))
        else:
            if full_graph_cache_enabled and os.path.exists(self.full_graph_path()):
                cached_attr_dict = graph_utils.load_cache_dict(self.full_graph_path())
            else:
                cached_attr_dict = self.preprocess(datasets.load_dataset(name))
                graph_utils.save_cache_dict(cached_attr_dict, self.full_graph_path())
        self.attr_dict = cached_attr_dict
        for attr in cached_attr_dict:
            setattr(self, attr, cached_attr_dict[attr])

    def __repr__(self):
        masks = ','.join(str(torch.count_nonzero(mask).item()) for mask in [self.train_mask, self.val_mask, self.test_mask])
        return f'<COO Graph: {self.name}, |V|: {self.num_nodes}, |E|: {self.num_edges}, masks: {masks}>'

    def partition(self, num_parts):
        begin = datetime.datetime.now()
        print(f'{num_parts} partition begin', self.full_graph_path(), begin)
        self.num_parts = num_parts
        full_dict = {k:v for k,v in self.attr_dict.items() if k not in ['adj', 'edge_index', 'features']}
        local_dict = {'local_'+x: torch.split(getattr(self,x), self.split_size) for x in ['train_mask', 'labels', 'features']}
        local_dict.update(dict(zip(['local_adj', 'local_adj_parts'], graph_utils.CAGNET_split(self.adj, self.split_size))))
        for i in range(num_parts):
            full_dict.update({k: v[i] for k, v in local_dict.items()})
            graph_utils.save_cache_dict(full_dict, self.parted_graph_path(i, num_parts))
        print(f'{num_parts} partition done ', datetime.datetime.now()-begin)

    def preprocess(self, attr_dict):
        begin = datetime.datetime.now()
        print('preprocess begin', self.full_graph_path(), begin)
        attr_dict["features"] = attr_dict["features"] / attr_dict["features"].sum(1, keepdim=True).clamp(min=1)
        if self.preprocess_for=='GCN':  # make the coo format sym lap matrix
            attr_dict['adj'] = graph_utils.sym_normalization(attr_dict['edge_index'], attr_dict['num_nodes'])
        elif self.preprocess_for=='GAT':
            pass
        attr_dict.pop('edge_index')
        print('preprocess done', self.full_graph_path(), datetime.datetime.now()-begin)
        return attr_dict


