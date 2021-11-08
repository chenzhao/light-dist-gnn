import datetime
from coo_graph import Parted_COO_Graph
from handcraft_gcn import GCN

import torch
import torch.nn.functional as F


def train(g, env, total_epoch):
    model = GCN(g, env)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    torch.cuda.synchronize()
    env.timer.start('training')
    for epoch in range(total_epoch):
        begin = datetime.datetime.now()
        outputs = model()
        optimizer.zero_grad()
        if list(g.local_labels[g.local_train_mask].size())[0] > 0:
            loss = F.nll_loss(outputs[g.local_train_mask], g.local_labels[g.local_train_mask])
            loss.backward()
        else:
            env.logger.log('Warning: no training nodes in this partition!')
            fake_loss = (outputs * 0).sum()
            fake_loss.backward()
        optimizer.step()
        env.logger.log("Epoch {:05d} | Loss {:.4f} | Time: {}".format(epoch, loss.item(), datetime.datetime.now()-begin), rank=0)

        if (epoch+1)%5==0:
            output_parts = [torch.zeros(g.split_size, g.num_classes, device=env.device) for _ in range(env.world_size)]
            if outputs.size(0) != g.split_size:
                pad_row = g.split_size - outputs.size(0)
                outputs = torch.cat((outputs, torch.zeros(pad_row, g.num_classes, device=env.device)), dim=0)
            env.all_gather(output_parts, outputs)  # output_parts[g_env.rank] = outputs

            last_part_size = g.num_nodes - g.split_size * (env.world_size - 1)
            output_parts[env.world_size - 1] = output_parts[env.world_size - 1][:last_part_size, :]
            outputs = torch.cat(output_parts, dim=0)

            if env.rank == 0:
                accs =  []
                for mask in [g.train_mask, g.val_mask, g.test_mask]:
                    pred = outputs[mask].max(1)[1]
                    acc = pred.eq(g.labels[mask]).sum().item() / mask.sum().item()
                    accs.append(acc)
                env.logger.log('Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(epoch, accs[0], accs[1], accs[2]), rank=0)
    env.timer.stop('training')


def copy_data_to_device(g, device):
    g.local_features = g.local_features.to(device)
    for i in range(len(g.local_adj_parts)):
        g.local_adj_parts[i] = g.local_adj_parts[i].to(device)
    g.local_labels = g.local_labels.to(device)


def main(env):
    env.logger.log('train begin at proc:', env)
    env.timer.start('total')
    # g = Parted_COO_Graph('reddit', rank=env.rank, num_parts=env.world_size)
    # g = Parted_COO_Graph('flickr', rank=env.rank, num_parts=env.world_size)
    g = Parted_COO_Graph('a_quarter_reddit', rank=env.rank, num_parts=env.world_size)
    env.logger.log('dataset loaded', g)
    copy_data_to_device(g, env.device)
    train(g, env, total_epoch=10)
    env.timer.stop('total')
    env.logger.log(env.timer.summary(), rank=0)
    pass
