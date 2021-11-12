from coo_graph import Parted_COO_Graph
from models import GCN, GAT

import torch
import torch.nn.functional as F


def train(g, env, total_epoch):
    model = GCN(g, env)
    # model = GAT(g, env)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(total_epoch):
        with env.timer.timing('epoch'):
            outputs = model(g.features)
            optimizer.zero_grad()
            if g.local_labels[g.local_train_mask].size(0) > 0:
                loss = F.nll_loss(outputs[g.local_train_mask], g.local_labels[g.local_train_mask])
            else:
                env.logger.log('Warning: no training nodes in this partition! Backward fake loss.')
                loss = (outputs * 0).sum()
            loss.backward()
            optimizer.step()
            env.logger.log("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), rank=0)

        if epoch%10==0 or epoch==total_epoch-1:
            all_outputs = env.all_gather_then_cat(outputs)
            acc = lambda mask: all_outputs[mask].max(1)[1].eq(g.labels[mask]).sum().item()/mask.sum().item()
            env.logger.log(f'Epoch: {epoch:03d}, Train: {acc(g.train_mask):.4f}, Val: {acc(g.val_mask):.4f}, Test: {acc(g.test_mask):.4f}', rank=0)


def main(env, args):
    env.logger.log('proc begin:', env)
    with env.timer.timing('total'):
        g = Parted_COO_Graph(args.dataset, rank=env.rank, num_parts=env.world_size, device=env.device)
        env.logger.log('graph loaded', g)
        train(g, env, total_epoch=args.epoch)
    env.logger.log(env.timer.summary_all(), rank=0)

