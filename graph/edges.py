import torch
from torch_geometric.nn import knn_graph, radius_graph

def create_spatial_edges(bboxes, max_k=5, radius=None, img_width=1920, img_height=1080):
    """
    Creates edge index based on spatial proximity.
    To prevent graph explosion (O(N^2)), we use k-NN or radius graphs.
    
    Args:
        bboxes: Tensor of shape (N, 4) in xyxy format
        max_k: Int, maximum number of neighbors to connect. 
        radius: Float, if provided, use radius threshold instead of KNN. 
                Radius should be normalized relative to image size [0, 1].
    Returns:
        edge_index: Tensor of shape (2, E)
    """
    N = bboxes.shape[0]
    if N <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=bboxes.device)
        
    # Calculate centers
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    
    # Normalize
    cx = cx / img_width
    cy = cy / img_height
    
    centers = torch.stack([cx, cy], dim=1)
    
    # Actually, K should be min(N - 1, max_k)
    k = min(N - 1, max_k)
    
    if radius is not None:
        # Create edges where points are within `radius` distance
        edge_index = radius_graph(centers, r=radius, loop=False)
    else:
        # Create k-nearest neighbor edges
        # loop=False prevents self-loops, but standard graphs might benefit from them.
        # We'll add self-loops explicitly later if needed.
        edge_index = knn_graph(centers, k=k, loop=False)
        
        # PyTorch Geometric knn_graph creates directed edges. 
        # For our interaction graph, if A interacts with B, B interacts with A.
        # So we should make the graph undirected.
        row, col = edge_index
        undirected_edge_index = torch.cat([edge_index, torch.stack([col, row], dim=0)], dim=1)
        
        # Remove duplicates
        undirected_edge_index = torch.unique(undirected_edge_index, dim=1)
        return undirected_edge_index
        
    return edge_index

def create_temporal_edges(current_ids, prev_ids):
    """
    Creates temporal edges linking node IDs in the current frame to 
    matching node IDs in the previous frame.
    
    Args:
        current_ids: Tensor of tracked IDs for the current frame [N]
        prev_ids: Tensor of tracked IDs for the previous frame [M]
    Returns:
        edge_index: Tensor of shape (2, E) connecting current to prev
    """
    edges = []
    
    # Basic O(N*M) matching, which is fine since N and M are small (number of people)
    for i, cid in enumerate(current_ids):
        if cid != -1: # -1 means lost track
            for j, pid in enumerate(prev_ids):
                if cid == pid:
                    # We connect the index in the current frame to the index in the previous
                    edges.append((i, j))
                    
    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=current_ids.device)
        
    edge_index = torch.tensor(edges, dtype=torch.long, device=current_ids.device).t()
    return edge_index
