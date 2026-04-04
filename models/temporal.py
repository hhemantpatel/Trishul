import torch
import torch.nn as nn

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        """
        A simple LSTM to model the temporal evolution of the scene graphs.
        Args:
            input_dim: The dimension of the graph-level embeddings (from GCN).
            hidden_dim: The hidden dimension of the LSTM.
        """
        super().__init__()
        # batch_first=True means inputs are [batch_size, seq_len, features]
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers,
                            batch_first=True)
                            
    def forward(self, x):
        """
        Args:
            x: Sequence of graph embeddings. Shape: [batch_size, seq_len, input_dim]
               (If batch_size=1, just [1, seq_len, input_dim] for a single video segment)
        Returns:
            outputs: LSTM outputs at each time step. Shape: [batch_size, seq_len, hidden_dim]
            hidden: Tuple (h_n, c_n) of the final hidden states.
        """
        # lstm returns: output sequence, and (final hidden state, final cell state)
        outputs, hidden = self.lstm(x)
        return outputs, hidden
