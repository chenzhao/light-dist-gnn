import datetime
import os.path
import torch
import torch.sparse
from . import graph_utils
from . import datasets


class BasicGraph:
    def __init__(self, d, name, device):
        self.name, self.device, self.attr_dict = name, device, d
        self.adj = d['adj'].to(device)
        self.features = d['features'].to(device)
        self.labels = d['labels'].to(device).to(torch.float if d['labels'].dim()==2 else torch.long)
        self.train_mask, self.val_mask, self.test_mask = (d[t].bool().to(device) for t in ("train_mask", 'val_mask', 'test_mask'))
        self.num_nodes, self.num_edges, self.num_classes = d["num_nodes"], d['num_edges'], d['num_classes']

    def __repr__(self):
        masks = ','.join(str(torch.sum(mask).item()) for mask in [self.train_mask, self.val_mask, self.test_mask])
        return f'<COO Graph: {self.name}, |V|: {self.num_nodes}, |E|: {self.num_edges}, masks: {masks}>'


class GraphCache:
    @staticmethod
    def full_graph_path(name, preprocess_for, root=datasets.data_root):
        return os.path.join(root, f'{name}_{preprocess_for}_full.coo_graph')
    @staticmethod
    def parted_graph_path(name, preprocess_for, rank, num_parts, root=datasets.data_root):
        dirpath = os.path.join(root, f'{name}_{preprocess_for}_{num_parts}_parts')
        os.makedirs(dirpath, exist_ok=True)
        return os.path.join(dirpath, f'part_{rank}_of_{num_parts}.coo_graph')
    @staticmethod
    def save_dict(d, path):
        if os.path.exists(path):
            print(f'warning: cache file {path} is overwritten.')
        d_to_save = {}
        for k, v in d.items():
            d_to_save[k] = v.clone() if type(v)==torch.Tensor else v
        torch.save(d_to_save, path)
    @staticmethod
    def load_dict(path):
        d = torch.load(path)
        updated_d = {}
        for k, v in d.items():
            if type(v) == torch.Tensor and v.is_sparse:
                updated_d[k] = v.coalesce()
        d.update(updated_d)
        return d


class COO_Graph(BasicGraph):
    def __init__(self, name, preprocess_for='GCN', full_graph_cache_enabled=True, device='cpu'):
        self.preprocess_for = preprocess_for
        self.cache_path = GraphCache.full_graph_path(name, preprocess_for)
        if full_graph_cache_enabled and os.path.exists(self.cache_path):
            cached_attr_dict = GraphCache.load_dict(self.cache_path)
        else:
            src_data = datasets.load_dataset(name)
            cached_attr_dict = graph_utils.preprocess(name, src_data, preprocess_for)  # norm feat, remove edge_index, add adj
            GraphCache.save_dict(cached_attr_dict, self.cache_path)
        super().__init__(cached_attr_dict, name, device)

    def partition(self, num_parts, padding=True):
        begin = datetime.datetime.now()
        print(self.name, num_parts, 'partition begin', begin)
        attr_dict = self.attr_dict.copy()
        split_size = (self.num_nodes+num_parts-1)//num_parts
        pad_size = split_size*num_parts-self.num_nodes

        adj_list = graph_utils.sparse_2d_split(self.adj, split_size)
        features_list = list(torch.split(self.features, split_size))

        if padding and pad_size>0:
            padding_feat = torch.zeros((pad_size, self.features.size(1)), dtype=self.features.dtype, device=self.device)
            features_list[-1] = torch.cat((features_list[-1], padding_feat))

            padding_labels_size = torch.Size([pad_size])+self.labels.size()[1:]
            padding_labels = torch.zeros(padding_labels_size, dtype=self.labels.dtype, device=self.device)
            attr_dict['labels'] = torch.cat((self.labels, padding_labels))

            padding_mask = torch.zeros(pad_size, dtype=self.train_mask.dtype, device=self.device)
            for key in ['train_mask', 'val_mask', 'test_mask']:
                attr_dict[key] = torch.cat((attr_dict[key], padding_mask))

            adj_list = [torch.sparse_coo_tensor(adj._indices(), adj._values(), (split_size, split_size*num_parts))
                        for adj in adj_list]

        for i in range(num_parts):
            cache_path = GraphCache.parted_graph_path(self.name, self.preprocess_for, i, num_parts)
            attr_dict.update({'adj': adj_list[i], 'features': features_list[i]})
            GraphCache.save_dict(attr_dict, cache_path)
            print(Parted_COO_Graph(self.name, i, num_parts, self.preprocess_for))
        print(self.name, num_parts, 'partition done', datetime.datetime.now()-begin)


class Parted_COO_Graph(BasicGraph):
    def __init__(self, name, rank, num_parts, preprocess_for='GCN', device='cpu'):
        # self.full_g = COO_Graph(name, preprocess_for, True, 'cpu')
        self.rank, self.num_parts = rank, num_parts
        cache_path = GraphCache.parted_graph_path(name, preprocess_for, rank, num_parts)
        if not os.path.exists(cache_path):
            raise Exception('Not parted yet. Run COO_Graph.partition() first.', cache_path)
        cached_attr_dict = GraphCache.load_dict(cache_path)
        super().__init__(cached_attr_dict, name, device)

        self.local_num_nodes = self.adj.size(0)
        self.local_num_edges = self.adj.values().size(0)
        self.local_labels = self.labels[self.local_num_nodes*rank:self.local_num_nodes*(rank+1)]
        self.local_train_mask = self.train_mask[self.local_num_nodes*rank:self.local_num_nodes*(rank+1)].bool()

        self.adj_parts = graph_utils.sparse_2d_split(self.adj, self.local_num_nodes, split_dim=1)

    def __repr__(self):
        local_g = f'<Local: {self.rank}, |V|: {self.local_num_nodes}, |E|: {self.local_num_edges}>'
        return super().__repr__() + local_g

