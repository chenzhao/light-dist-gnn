from coo_graph import Parted_COO_Graph
from models import GCN, GAT, CachedGCN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from sklearn.metrics import f1_score
from dist_utils import DistEnv


def f1(y_true, y_pred, multilabel=True):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    if multilabel:
        y_pred[y_pred > 0.5] = 1.0
        y_pred[y_pred <= 0.5] = 0.0
        for node in [10,100,1000]:
            DistEnv.env.logger.log('pred', y_pred[node] , rank=0)
            DistEnv.env.logger.log('true', y_true[node] , rank=0)
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
           f1_score(y_true, y_pred, average="macro")

def train(g, env, total_epoch):
    model = GCN(g, env, hidden_dim=256)
    model = CachedGCN(g, env, hidden_dim=16)
    # model = GAT(g, env)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if g.labels.dim()==1:
        loss_func = nn.CrossEntropyLoss()
    elif g.labels.dim()==2:
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    for epoch in range(total_epoch):
        with env.timer.timing('epoch'):
            with autocast(env.half_enabled):
                outputs = model(g.features)
                optimizer.zero_grad()
                if g.local_labels[g.local_train_mask].size(0) > 0:
                    loss = loss_func(outputs[g.local_train_mask], g.local_labels[g.local_train_mask])
                else:
                    env.logger.log('Warning: no training nodes in this partition! Backward fake loss.')
                    loss = (outputs * 0).sum()
            loss.backward()
            optimizer.step()
            env.logger.log("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), rank=0)

        if epoch%10==0 or epoch==total_epoch-1:
            all_outputs = env.all_gather_then_cat(outputs)
            if g.labels.dim()>1:
                mask = g.train_mask
                env.logger.log(f'Epoch: {epoch:03d}', f1(g.labels[mask], torch.sigmoid(all_outputs[mask])), rank=0)
            else:
                acc = lambda mask: all_outputs[mask].max(1)[1].eq(g.labels[mask]).sum().item()/mask.sum().item()
                env.logger.log(f'Epoch: {epoch:03d}, Train: {acc(g.train_mask):.4f}, Val: {acc(g.val_mask):.4f}, Test: {acc(g.test_mask):.4f}', rank=0)


def main(env, args):
    env.csr_enabled = False
    env.csr_enabled = True

    env.half_enabled = True
    env.half_enabled = False
    env.logger.log('proc begin:', env)
    with env.timer.timing('total'):
        g = Parted_COO_Graph(args.dataset, env.rank, env.world_size, env.device, env.half_enabled, env.csr_enabled)
        env.logger.log('graph loaded', g)
        env.logger.log('graph loaded', torch.cuda.memory_summary())
        train(g, env, total_epoch=args.epoch)
    env.logger.log(env.timer.summary_all(), rank=0)

