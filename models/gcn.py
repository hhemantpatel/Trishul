import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class SpatialGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        A simple Graph Convolutional Network for the spatial interactions 
        in a single frame.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            batch: Batch vector [N] indicating which graph a node belongs to
        Returns:
            x: Updated node features [N, out_channels]
            graph_embed: Graph-level embedding [batch_size, out_channels]
        """
        # First GCN Layer
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        
        # Second GCN Layer
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        
        # If batch is provided, we can pool the node features into a frame-level embedding
        if batch is not None:
            graph_embed = global_mean_pool(x, batch)
        else:
            # Assume all nodes belong to a single graph (one frame)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_embed = global_mean_pool(x, batch)
            
        return x, graph_embed
