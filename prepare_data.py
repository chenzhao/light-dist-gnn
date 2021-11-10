# Copyright 2021, Zhao CHEN
# All rights reserved.
import coo_graph
import argparse


def main():
    cached = True
    # r = COO_Graph('cora')
    # r = Parted_COO_Graph('flickr')
    # r = coo_graph.Parted_COO_Graph('cora', full_graph_cache_enabled=cached)
    # r = coo_graph.Parted_COO_Graph('flickr', full_graph_cache_enabled=cached)
    r = coo_graph.Parted_COO_Graph('reddit', full_graph_cache_enabled=cached)
    # r = coo_graph.Parted_COO_Graph('a_quarter_reddit', full_graph_cache_enabled=True)
    print(r.adj.size())
    r.partition(8)
    r.partition(4)

    # r = COO_Graph('reddit', cached=True)
    # r = COO_Graph('AmazonProducts')
    # r = COO_Graph('Yelp')
    print(r)
    return


if __name__ == '__main__':
    main()
