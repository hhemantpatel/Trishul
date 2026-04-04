import torch
import torch.nn as nn
from torch_geometric.data import Data
from models.baseline import BaselineSpatioTemporalModel

def run_gradient_check():
    """
    Constructs a series of dummy graph structures mimicking the output of our tracker,
    verifies the models can ingest them and outputs sequences, and tests if gradients flow 
    backward from a loss function.
    """
    print("Initializing Baseline Model...")
    # Assume 16 pose embeddings + 4 spatial + 2 velocity = 22 node features
    model = BaselineSpatioTemporalModel(node_in_features=22, gcn_hidden=32, gcn_out=64, lstm_hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # 1. Create a dummy sequence of 5 graphs (e.g. 5 consecutive video frames)
    print("Generating dummy graph sequence...")
    sequence = []
    seq_length = 5
    for t in range(seq_length):
        # Frame t has 3 people (nodes)
        num_nodes = 3
        # Dummy features
        x = torch.rand((num_nodes, 22), dtype=torch.float32)
        # Dummy fully connected edges (ignoring self-loops)
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
                                   [1, 2, 0, 2, 0, 1]], dtype=torch.long)
        
        sequence.append(Data(x=x, edge_index=edge_index))
        
    # 2. Forward Pass
    print("Running forward pass...")
    logits = model(sequence) # [1, seq_length, 1]
    
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (1, seq_length, 1), "Output shape mismatch!"
    
    # 3. Dummy Targets
    # Let's say it's a suspicious event that escalates: first 2 frames safe (0), last 3 unsafe (1)
    targets = torch.tensor([[[0.0], [0.0], [1.0], [1.0], [1.0]]])
    
    # 4. Compute Loss
    loss = criterion(logits, targets)
    print(f"Loss computed: {loss.item():.4f}")
    
    # 5. Backward Pass
    loss.backward()
    
    # 6. Verify Gradients Flowed (check a random parameter from early in the network)
    # Check spatial GCN layer gradients
    gcn_grad = model.gcn.conv1.lin.weight.grad
    if gcn_grad is not None and gcn_grad.abs().sum() > 0:
        print("SUCCESS! Gradients successfully propagated through LSTM and GCN back to spatial layers.")
    else:
        print("ERROR: Gradients are broken.")
        
if __name__ == "__main__":
    run_gradient_check()
