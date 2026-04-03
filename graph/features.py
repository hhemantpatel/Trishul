import torch

def compute_edge_features(edge_index, bboxes, velocities):
    """
    Computes initial edge features for the spatial graph.
    
    Args:
        edge_index: Tensor of shape (2, E) where edge_index[0] is source, edge_index[1] is target
        bboxes: Tensor of shape (N, 4) in xyxy format
        velocities: Tensor of shape (N, 2) in (vx, vy) format
    Returns:
        edge_attr: Tensor of shape (E, feature_dim)
    """
    if edge_index.shape[1] == 0:
        # Return empty feature tensor with correct dimension (e.g., 4: dx, dy, dvx, dvy)
        return torch.zeros((0, 4), dtype=torch.float32, device=bboxes.device)
        
    src, dst = edge_index[0], edge_index[1]
    
    # Calculate geometric centers
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    
    # Relative distance features
    # How far is dst from src
    dx = cx[dst] - cx[src]
    dy = cy[dst] - cy[src]
    
    # Relative velocity features
    dvx = velocities[dst, 0] - velocities[src, 0]
    dvy = velocities[dst, 1] - velocities[src, 1]
    
    # We can also add euclidean distance / euclidean relative speed
    # dist = torch.sqrt(dx**2 + dy**2 + 1e-6)
    # rel_speed = torch.sqrt(dvx**2 + dvy**2 + 1e-6)
    
    # Combine into edge feature vector. 
    # Shape: [E, 4]
    edge_attr = torch.stack([dx, dy, dvx, dvy], dim=1)
    
    return edge_attr
