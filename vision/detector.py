import torch
from ultralytics import YOLO
from typing import Dict, Any

class HumanDetector:
    def __init__(self, model_size='yolov8n-pose.pt', device=None):
        """
        Initializes the YOLOv8 pose estimation model.
        Args:
            model_size: String dictating model size (n, s, m, l, x). Default is nano for speed.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading YOLO model {model_size} on {self.device}...")
        self.model = YOLO(model_size)
        self.model.to(self.device)

    def detect(self, img) -> Any:
        """
        Detects humans and keypoints in a single image.
        Returns the raw results object from YOLO.
        """
        # class 0 is person in COCO dataset
        results = self.model(img, classes=[0], verbose=False)
        return results[0]

    def extract_features(self, result) -> Dict[str, Any]:
        """
        Extracts formatted bounding boxes and keypoints from a result object.
        """
        # result.boxes.xyxy: [N, 4] formatted bounding boxes
        # result.keypoints.data: [N, 17, 3] formatted as (x, y, confidence)
        
        bboxes = []
        keypoints = []
        confidences = []
        
        if result.boxes is not None:
            bboxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            if result.keypoints.data is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                
        return {
            "bboxes": bboxes,
            "keypoints": keypoints,
            "confidences": confidences
        }
