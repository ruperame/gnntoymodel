import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def get_positive_negative_edges(data, num_negatives=1):
    pos_edge_index = data.edge_index.T
    num_nodes = data.num_nodes
    neg_edges = set()
    pos_edges_set = set(map(tuple, pos_edge_index.tolist()))

    while len(neg_edges) < len(pos_edge_index) * num_negatives:
        i, j = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
        if i != j and (i, j) not in pos_edges_set and (i, j) not in neg_edges:
            neg_edges.add((i, j))

    neg_edge_index = torch.tensor(list(neg_edges), dtype=torch.long).T
    return pos_edge_index.T, neg_edge_index
