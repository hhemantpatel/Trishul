import torch
from torch_geometric.data import Data
from graph.nodes import NodeFeatureExtractor
from graph.edges import create_spatial_edges
from graph.features import compute_edge_features

class GraphBuilder:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_extractor = NodeFeatureExtractor().to(self.device)
        
        # State to track temporal velocities
        self.prev_centers = {} # track_id -> (cx, cy)
        
    def build_frame_graph(self, tracking_data_entry, img_width=1920, img_height=1080):
        """
        Converts a single frame's structured outputs into a PyTorch Geometric Data object.
        
        Args:
            tracking_data_entry: Dictionary containing 'bboxes', 'keypoints', 'track_ids'
            
        Returns:
            torch_geometric.data.Data object representing the spatial graph of the frame.
        """
        bboxes = torch.tensor(tracking_data_entry['bboxes'], dtype=torch.float32, device=self.device)
        keypoints = torch.tensor(tracking_data_entry['keypoints'], dtype=torch.float32, device=self.device)
        track_ids = tracking_data_entry['track_ids']
        
        N = bboxes.shape[0]
        
        if N == 0:
            # Empty graph
            return Data(
                x=torch.zeros((0, self.node_extractor.feature_proj[0].out_features), device=self.device),
                edge_index=torch.zeros((2, 0), dtype=torch.long, device=self.device),
                edge_attr=torch.zeros((0, 4), device=self.device),
                track_ids=track_ids
            )
            
        # 1. Compute Velocities based on previous frame centers
        velocities = torch.zeros((N, 2), dtype=torch.float32, device=self.device)
        current_centers = {}
        
        for i, tid in enumerate(track_ids):
            if tid == -1: 
                continue
                
            w = bboxes[i, 2] - bboxes[i, 0]
            h = bboxes[i, 3] - bboxes[i, 1]
            cx = (bboxes[i, 0] + w / 2).item()
            cy = (bboxes[i, 1] + h / 2).item()
            
            if tid in self.prev_centers:
                # Velocity = (current - prev) / 1 (frame)
                vx = cx - self.prev_centers[tid][0]
                vy = cy - self.prev_centers[tid][1]
                velocities[i, 0] = vx / img_width
                velocities[i, 1] = vy / img_height
                
            current_centers[tid] = (cx, cy)
            
        # Update state
        self.prev_centers = current_centers
        
        # 2. Extract Node Features
        x = self.node_extractor(bboxes, keypoints, velocities, img_width, img_height)
        
        # 3. Create Spatial Edge Index (k-NN)
        edge_index = create_spatial_edges(bboxes, max_k=3, img_width=img_width, img_height=img_height)
        
        # 4. Compute Edge Features
        edge_attr = compute_edge_features(edge_index, bboxes, velocities)
        
        # 5. Package into PyTorch Geometric Data
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, track_ids=track_ids)
        
        return graph_data
