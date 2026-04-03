import json
import os
import time
from typing import List, Dict, Any
import numpy as np

# Try importing decord, fallback to cv2 if not available yet
try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    import cv2
    HAS_DECORD = False

from vision.tracker import HumanTracker

class VisionPipeline:
    def __init__(self, model_size='yolov8n-pose.pt', device=None):
        """
        Initializes the vision pipeline.
        """
        self.tracker = HumanTracker(model_size=model_size, device=device)

    def process_video(self, video_path: str, output_json_path: str, max_frames=None, frames_to_skip=1):
        """
        Processes a video and extracts ID-consistent bounding boxes and keypoints.
        
        Args:
            video_path: Path to the input video.
            output_json_path: Path to save the extracted track sequence.
            max_frames: Stop processing after this many frames (useful for testing).
            frames_to_skip: Only process every N-th frame to simulate real-time limitations or speed up processing.
        """
        print(f"Processing {video_path}...")
        start_time = time.time()
        
        frame_tracking_data = [] # Store all data here
        
        if HAS_DECORD:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # Using random seek loop for Decord testing/flexibility (though sequential here)
            frame_indices = list(range(0, total_frames, frames_to_skip))
            if max_frames:
                frame_indices = frame_indices[:max_frames]
                
            for idx in frame_indices:
                # Get the frame
                frame = vr[idx].asnumpy()
                # Frame from Decord is RGB, YOLO expects BGR like OpenCV, but Ultralytics automatically handles RGB/BGR inference
                # However, to be safe and match standard OpenCV inputs, we convert
                # frame = frame[:, :, ::-1] # RGB to BGR
                
                track_data = self.tracker.track_frame(frame, persist=True)
                
                # Format for JSON serialization (convert numpy arrays to lists)
                frame_entry = {
                    "frame_id": int(idx),
                    "track_ids": track_data['track_ids'].tolist() if isinstance(track_data['track_ids'], np.ndarray) else track_data['track_ids'],
                    "bboxes": track_data['bboxes'].tolist() if isinstance(track_data['bboxes'], np.ndarray) else track_data['bboxes'],
                    "keypoints": track_data['keypoints'].tolist() if isinstance(track_data['keypoints'], np.ndarray) else track_data['keypoints']
                }
                frame_tracking_data.append(frame_entry)
                
                if idx % 100 == 0:
                    print(f"Processed frame {idx}/{total_frames}")

        else:
            print("Decord not found. Falling back to OpenCV sequential read. (Please install decord for faster testing)")
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if idx % frames_to_skip == 0:
                    track_data = self.tracker.track_frame(frame, persist=True)
                    frame_entry = {
                        "frame_id": int(idx),
                        "track_ids": track_data['track_ids'].tolist() if isinstance(track_data['track_ids'], np.ndarray) else track_data['track_ids'],
                        "bboxes": track_data['bboxes'].tolist() if isinstance(track_data['bboxes'], np.ndarray) else track_data['bboxes'],
                        "keypoints": track_data['keypoints'].tolist() if isinstance(track_data['keypoints'], np.ndarray) else track_data['keypoints']
                    }
                    frame_tracking_data.append(frame_entry)
                    
                idx += 1
                if max_frames and idx >= (max_frames * frames_to_skip):
                    break
                    
                if idx % 100 == 0:
                    print(f"Processed frame {idx}/{total_frames}")
                    
            cap.release()
            
        # Save to JSON
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(frame_tracking_data, f)
            
        fps = len(frame_tracking_data) / (time.time() - start_time)
        print(f"Processing complete! Saved to {output_json_path}. Speed: {fps:.2f} trackings/sec")

if __name__ == "__main__":
    # Example usage for testing
    # pipeline = VisionPipeline()
    # pipeline.process_video("data/sample.mp4", "data/sample_tracking.json", max_frames=50)
    pass
