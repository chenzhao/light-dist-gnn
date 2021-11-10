# Copyright 2021, Zhao CHEN
# All rights reserved.
import coo_graph

def main():
    # r = COO_Graph('cora')
    # r = Parted_COO_Graph('flickr')
    # r = coo_graph.Parted_COO_Graph('cora', full_graph_cache_enabled=True)
    # r = coo_graph.Parted_COO_Graph('flickr', full_graph_cache_enabled=False)
    r = coo_graph.Parted_COO_Graph('reddit', full_graph_cache_enabled=False)
    # r = coo_graph.Parted_COO_Graph('a_quarter_reddit', full_graph_cache_enabled=False)
    print(r.adj.size())
    r.partition(4)

    # r = COO_Graph('reddit', cached=True)
    # r = COO_Graph('AmazonProducts')
    # r = COO_Graph('Yelp')
    print(r)
    return

    # pr = COO_Graph('PartedReddit')
    # r = COO_Graph('SmallerReddit')
    # pr = COO_Graph('PartedSmallerReddit')
    # r = COO_Graph('OneQuarterReddit')
    # pr = COO_Graph('PartedOneQuarterReddit')
    pass





# split 22312 tensor(23030.6914, device='cuda:0')
# sub split 22312 tensor(9685.2031, device='cuda:0')
# sub split 22312 tensor(4778.6084, device='cuda:0')
# sub split 22312 tensor(4412.4980, device='cuda:0')
# sub split 22312 tensor(4153.9517, device='cuda:0')
# sub split 2 tensor(0.3528, device='cuda:0')
# split 22312 tensor(17396.8789, device='cuda:0')
# sub split 22312 tensor(4778.5796, device='cuda:0')
# sub split 22312 tensor(7103.8589, device='cuda:0')
# sub split 22312 tensor(2929.7168, device='cuda:0')
# sub split 22312 tensor(2584.0396, device='cuda:0')
# sub split 2 tensor(0.2919, device='cuda:0')
# split 22312 tensor(15435.5361, device='cuda:0')
# sub split 22312 tensor(4412.4155, device='cuda:0')
# sub split 22312 tensor(2929.7158, device='cuda:0')
# sub split 22312 tensor(6047.8208, device='cuda:0')
# sub split 22312 tensor(2044.3082, device='cuda:0')
# sub split 2 tensor(0.2834, device='cuda:0')
# split 22312 tensor(13964.4004, device='cuda:0')
# sub split 22312 tensor(4153.8833, device='cuda:0')
# sub split 22312 tensor(2584.0420, device='cuda:0')
# sub split 22312 tensor(2044.3065, device='cuda:0')
# sub split 22312 tensor(5180.3916, device='cuda:0')
# sub split 2 tensor(0.1291, device='cuda:0')
# split 2 tensor(1.3905, device='cuda:0')
# sub split 22312 tensor(0.3528, device='cuda:0')
# sub split 22312 tensor(0.2919, device='cuda:0')
# sub split 22312 tensor(0.2834, device='cuda:0')
# sub split 22312 tensor(0.1291, device='cuda:0')
# sub split 2 tensor(0.3333, device='cuda:0')











if __name__ == '__main__':
    main()
