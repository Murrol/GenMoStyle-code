import numpy as np
from data.graph import Graph
import torch

node_map1 = {'l_leg1': [1, 2],
             'l_leg2': [3, 4],
             'r_leg1': [5, 6],
             'r_leg2': [7, 8],
             'abdom': [0, 9],
             'head': [10, 11, 12],
             'l_arm1': [13, 14],
             'l_arm2': [15, 16],
             'r_arm1': [17, 18],
             'r_arm2': [19, 20]}

node_map2 = {'l_leg': [0, 1],
             'r_leg': [2, 3],
             'spine': [4, 5],
             'l_arm': [6, 7],
             'r_arm': [8, 9]}

node_map3 = {'top': [2, 3, 4],
             'bottom': [0, 1]}


###############################################################################
# Skeleton Graph
###############################################################################
class SkeletonGraph(object):
    def __init__(self):
        self.G0 = Graph('styletransfer')
        self.G1 = Graph('styletransfer_down1')
        self.G2 = Graph('styletransfer_down2')
        self.G3 = Graph('styletransfer_down3')

        self.node_map1 = node_map1
        self.node_map2 = node_map2
        self.node_map3 = node_map3

    def get_graph(self, blk_type=None, order=None):
        if blk_type == 'encode':
            self.graph = getattr(self, 'G' + str(order - 1))
            self.graph_new = getattr(self, 'G' + str(order))
            self.node_map = getattr(self, 'node_map' + str(order))
            self.M = aggregate_node(self.graph.num_node, self.graph_new.num_node, self.node_map)
        elif blk_type == 'decode':
            self.graph = getattr(self, 'G' + str(order))
            self.graph_new = getattr(self, 'G' + str(order - 1))
            self.node_map = getattr(self, 'node_map' + str(order))
            self.M = interpolate_node(self.graph.num_node, self.graph_new.num_node, self.node_map)
        else:
            raise NotImplementedError('Block type should be specified.')

    def get_adjacency(self, blk_type=None, order=None):
        self.get_graph(blk_type, order)
        A10 = torch.tensor(self.graph.A1, dtype=torch.float)
        A11 = torch.tensor(self.graph_new.A1, dtype=torch.float)
        A30 = torch.tensor(self.graph.A3, dtype=torch.float)
        A31 = torch.tensor(self.graph_new.A3, dtype=torch.float)
        M = torch.tensor(self.M, dtype=torch.float)

        return A10, A11, A30, A31, M


def aggregate_node(A, B, node_map):
    assert len(node_map) == B
    M = np.zeros((A, B))
    for i, (key, val) in enumerate(node_map.items()):
        for j in range(len(val)):
            M[val[j], i] = 1 / len(val)
    return M


def interpolate_node(B, A, node_map):
    assert len(node_map) == B
    M = np.zeros((B, A))
    for i, (key, val) in enumerate(node_map.items()):
        for j in range(len(val)):
            M[i, val[j]] = 1
    return M