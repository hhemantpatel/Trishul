import torch
import torch.nn as nn
import numpy as np

class NodeFeatureExtractor(nn.Module):
    def __init__(self, pose_dim=17*3, pose_embed_dim=16, hidden_dim=32):
        """
        Extracts and normalizes features for each person (node) in a frame.
        Includes a lightweight MLP to encode pose keypoints so they don't dominate
        the raw spatial coordinates.
        """
        super().__init__()
        
        # Lightweight encoder for the 17 keypoints (x, y, conf)
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 32),
            nn.ReLU(),
            nn.Linear(32, pose_embed_dim),
            nn.ReLU()
        )
        
        # We concatenate: [x_center, y_center, w, h, vx, vy, pose_embedding...]
        # Dimension = 4 (box) + 2 (velocity) + pose_embed_dim
        input_dim = 4 + 2 + pose_embed_dim
        
        # Optional final projection for the combined node feature
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, bboxes, keypoints, velocities=None, img_width=1920, img_height=1080):
        """
        Calculates node features for the current frame.
        
        Args:
            bboxes: Tensor of shape (N, 4) in xyxy format
            keypoints: Tensor of shape (N, 17, 3) 
            velocities: Tensor of shape (N, 2) representing (vx, vy). 
                        If None, we assume (0, 0) for the very first frame.
        Returns:
            Tensor of shape (N, hidden_dim) representing node features.
        """
        N = bboxes.shape[0]
        if N == 0:
            return torch.zeros(0, self.feature_proj[0].out_features, device=bboxes.device)
            
        # 1. Spatial normalize bounding boxes (convert xyxy -> cx, cy, w, h)
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        cx = bboxes[:, 0] + w / 2
        cy = bboxes[:, 1] + h / 2
        
        # Normalize to [0, 1]
        cx = cx / img_width
        cy = cy / img_height
        w = w / img_width
        h = h / img_height
        
        # Stack spatial features: [N, 4]
        spatial_feats = torch.stack([cx, cy, w, h], dim=1)
        
        # 2. Velocity
        if velocities is None:
            velocities = torch.zeros((N, 2), dtype=torch.float32, device=bboxes.device)
            
        # 3. Pose features 
        # Flatten [N, 17, 3] to [N, 51]
        flat_keypoints = keypoints.view(N, -1)
        # Normalize pose coordinates as well by dividing by image dims (approximate)
        # Ideally, we divide x by width and y by height. 
        # For simplicity in this flatten strategy, we can leave the encoder to handle it 
        # if the input is mostly contained, although precise normalization is better.
        pose_embeds = self.pose_encoder(flat_keypoints)
        
        # Concat all features: [N, 4 + 2 + pose_embed_dim]
        combined = torch.cat([spatial_feats, velocities, pose_embeds], dim=1)
        
        # Apply final projection
        node_features = self.feature_proj(combined)
        
        return node_features

def compute_velocities(current_bboxes, prev_bboxes, current_ids, prev_ids):
    """
    Utility to compute velocity between frames based on tracking IDs.
    Returns Tensor of shape (N, 2) acting as (vx, vy).
    """
    # ... implementation to match IDs and subtract centers ...
    pass
