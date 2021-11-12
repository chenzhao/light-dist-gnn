# Copyright 2021, Zhao CHEN
# All rights reserved.
import coo_graph
import argparse


def main():
    cached = False
    # r = COO_Graph('cora')
    # r = coo_graph.COO_Graph('cora', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('flickr', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('reddit', full_graph_cache_enabled=cached)
    r = coo_graph.COO_Graph('a_quarter_reddit', full_graph_cache_enabled=cached)
    r.partition(8)
    r.partition(4)
    return
    for name in ['reddit', 'flickr', 'cora']:
        r = coo_graph.COO_Graph(name, full_graph_cache_enabled=cached)
        r.partition(8)
        r.partition(4)
        print(r)
    return


if __name__ == '__main__':
    main()
