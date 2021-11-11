import datetime
from coo_graph import Parted_COO_Graph
from handcraft_gcn import GCN

import torch
import torch.nn.functional as F


def train(g, env, total_epoch):
    model = GCN(g, env)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(total_epoch):
        env.timer.start('epoch')
        outputs = model(g.local_features)
        optimizer.zero_grad()
        if g.local_labels[g.local_train_mask].size(0) > 0:
            loss = F.nll_loss(outputs[g.local_train_mask], g.local_labels[g.local_train_mask])
        else:
            env.logger.log('Warning: no training nodes in this partition! Backward fake loss.')
            loss = (outputs * 0).sum()
        loss.backward()
        optimizer.step()
        env.logger.log("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), rank=0)
        env.timer.stop('epoch')

        if (epoch+1)%5==0:
            output_parts = [torch.zeros(g.split_size, g.num_classes, device=env.device) for _ in range(env.world_size)]
            if outputs.size(0) != g.split_size:
                pad_row = g.split_size - outputs.size(0)
                outputs = torch.cat((outputs, torch.zeros(pad_row, g.num_classes, device=env.device)), dim=0)
            env.all_gather(output_parts, outputs)  # output_parts[g_env.rank] = outputs

            last_part_size = g.num_nodes - g.split_size * (env.world_size - 1)
            output_parts[env.world_size - 1] = output_parts[env.world_size - 1][:last_part_size, :]
            outputs = torch.cat(output_parts, dim=0)

            acc = lambda mask: (outputs[mask].max(1)[1].eq(g.labels[mask]).sum()/mask.sum()).item()
            env.logger.log(f'Epoch: {epoch:03d}, Train: {acc(g.train_mask):.4f}, Val: {acc(g.val_mask):.4f}, Test: {acc(g.test_mask):.4f}', rank=0)

def main(env, args):
    env.logger.log('proc begin:', env)
    env.timer.start('total')
    g = Parted_COO_Graph(args.dataset, rank=env.rank, num_parts=env.world_size, device=env.device)
    env.logger.log('graph loaded', g)
    train(g, env, total_epoch=args.epoch)
    env.timer.stop('total')
    env.logger.log(env.timer.summary_all(), rank=0)

