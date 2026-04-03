import torch
from ultralytics import YOLO
from typing import Dict, Any, List

class HumanTracker:
    def __init__(self, model_size='yolov8n-pose.pt', device=None):
        """
        Initializes the ByteTrack tracking pipeline via YOLOv8.
        We encapsulate the YOLO model here again because Ultralytics .track() 
        operates directly on the model instance and integrates tracking natively.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Tracking model {model_size} on {self.device}...")
        self.model = YOLO(model_size)
        self.model.to(self.device)

    def track_frame(self, img, persist=True) -> Dict[str, Any]:
        """
        Runs tracking on a single frame. 
        Args:
            img: An image represented as a numpy array.
            persist: bool, whether to persist tracker state between frames.
        Returns extracted tracking data.
        """
        # Run ByteTrack tracking natively. 
        # classes=[0] filters to humans only.
        results = self.model.track(img, classes=[0], persist=persist, tracker="bytetrack.yaml", verbose=False)
        result = results[0]
        
        bboxes = []
        keypoints = []
        track_ids = []
        
        if result.boxes is not None:
            bboxes = result.boxes.xyxy.cpu().numpy()
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                # Fallback if no IDs are assigned (e.g. tracking was lost momentarily)
                track_ids = [-1] * len(bboxes)
                
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            if result.keypoints.data is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                
        return {
            "track_ids": track_ids,
            "bboxes": bboxes,
            "keypoints": keypoints
        }
