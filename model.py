import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap


class GNN(torch.nn.Module):
    def __init__(self, input_dim, embedding_size, output_dim, task='node'):
        super(GNN, self).__init__()

        # GNN Layers
        self.conv1 = GATConv(input_dim, embedding_size, heads=3)
        self.transform1 = nn.Linear(embedding_size*3, embedding_size)
        self.pooling1 = TopKPooling(embedding_size, ratio=0.8)
        self.conv2 = GATConv(embedding_size, embedding_size, heads=3)
        self.transform2 = nn.Linear(embedding_size*3, embedding_size)
        self.pooling2 = TopKPooling(embedding_size, ratio=0.5)
        self.conv3 = GATConv(embedding_size, embedding_size, heads=3)
        self.transform3 = nn.Linear(embedding_size*3, embedding_size)
        self.pooling3 = TopKPooling(embedding_size, ratio=0.2)

        # Linear layers:
        self.linear1 = nn.Linear(embedding_size * 2, 1024)
        self.linear2 = nn.Linear(1024, output_dim)

    def forward(self, x, edge_index, edge_attr, batch_index):
        x = self.conv1(x, edge_index)
        x = self.transform1(x)

        x, edge_index, edge_attr, batch_index, _, _ = self.pooling1(x,
                                                                 edge_index,
                                                                 None, 
                                                                 batch_index)
        x1 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        x = self.conv2(x, edge_index)
        x = self.transform2(x)

        x, edge_index, edge_attr, batch_index, _, _ = self.pooling2(x,
                                                                 edge_index,
                                                                 None, 
                                                                 batch_index)
        x2 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        x = self.conv3(x, edge_index)
        x = self.transform3(x)

        x, edge_index, edge_attr, batch_index, _, _ = self.pooling3(x,
                                                                 edge_index,
                                                                 None, 
                                                                 batch_index)
        x3 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        x = x1 + x2 + x3

        x = self.linear1(x).relu()
        x = f.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)

        return x
    