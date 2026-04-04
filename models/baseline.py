import torch
import torch.nn as nn
from models.gcn import SpatialGCN
from models.temporal import TemporalLSTM

class BaselineSpatioTemporalModel(nn.Module):
    def __init__(self, node_in_features, gcn_hidden=64, gcn_out=128, lstm_hidden=128, num_classes=1):
        """
        Combines the Spatial GCN and Temporal LSTM into an end-to-end model.
        Args:
            node_in_features: Number of features per node (from graph/nodes.py)
            gcn_hidden: Hidden dimension of GCN layer
            gcn_out: Output dimension of spatial graphs (and input to LSTM)
            lstm_hidden: Hidden dimension of the LSTM
            num_classes: Output dimension, 1 for binary risk score (using BCE/Sigmoid)
        """
        super().__init__()
        
        # Spatial Modality
        self.gcn = SpatialGCN(in_channels=node_in_features, 
                              hidden_channels=gcn_hidden, 
                              out_channels=gcn_out)
                              
        # Temporal Modality
        self.lstm = TemporalLSTM(input_dim=gcn_out, 
                                 hidden_dim=lstm_hidden)
                                 
        # Output Head
        self.prediction_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            # Usually a Sigmoid is added at the end for single-class risk score probability [0, 1]
            # but it is better to use BCEWithLogitsLoss during training which expects raw logits.
            # So we omit Sigmoid here. 
        )
        
    def forward(self, graph_sequence_data):
        """
        Args:
            graph_sequence_data: A list or batch of torch_geometric `Data` objects
                                 representing a temporal sequence of frames.
        """
        # Because the number of nodes per frame varies, we process the GCN per frame (or use PyG batches)
        # We'll iteratively process each frame for simplicity in this baseline.
        
        frame_embeddings = []
        for frame_data in graph_sequence_data:
            x, edge_index = frame_data.x, frame_data.edge_index
            
            # Forward GCN
            _, graph_embed = self.gcn(x, edge_index) # graph_embed is [1, gcn_out]
            frame_embeddings.append(graph_embed)
            
        if not frame_embeddings:
           return None
            
        # Stack temporal representations into sequence: shape [1, seq_len, gcn_out]
        # (Assuming batch size of 1 sequence for now)
        temporal_seq = torch.stack(frame_embeddings, dim=1)
        
        # Forward LSTM
        lstm_out, _ = self.lstm(temporal_seq) # [1, seq_len, lstm_hidden]
        
        # We can either make a prediction at every time step (early detection scoring)
        # or just at the end of the video segment. We want to do early detection,
        # so we run the prediction head across all time steps.
        
        # Output shape: [1, seq_len, num_classes]
        logits = self.prediction_head(lstm_out)
        
        return logits
