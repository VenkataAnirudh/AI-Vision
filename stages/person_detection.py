import cv2
import supervision as sv
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_manager, config):
        self.config = config['models']['person']
        self.device_config = config['hardware']['device']
        
        # Load YOLO into VRAM
        self.model = model_manager.load_torch_model(
            'yolo_person', 
            lambda: YOLO(self.config['model'])
        )
        self.tracker = sv.ByteTrack()
        self.unique_track_ids = set()

    def process_frames(self, frames):
        """Processes a list of frames and assigns persistent track IDs."""
        results = []
        
        for frame_idx, frame_data in enumerate(frames):
            # frame_data can be a frame numpy array or a dict from adaptive sampling
            frame = frame_data['frame'] if isinstance(frame_data, dict) else frame_data
            
            # YOLO Inference
            yolo_result = self.model(frame, classes=self.config['classes'], conf=self.config['confidence'], verbose=False)[0]
            
            # Convert to supervision format for tracking
            detections = sv.Detections.from_ultralytics(yolo_result)
            tracked_detections = self.tracker.update_with_detections(detections)
            
            frame_persons = []
            if tracked_detections.tracker_id is not None:
                for idx in range(len(tracked_detections)):
                    bbox = tracked_detections.xyxy[idx]
                    conf = tracked_detections.confidence[idx]
                    tracker_id = tracked_detections.tracker_id[idx]
                    
                    self.unique_track_ids.add(int(tracker_id))
                    frame_persons.append({
                        'track_id': int(tracker_id),
                        'bbox': [int(x) for x in bbox],
                        'confidence': float(conf)
                    })
            
            results.append({
                'frame_idx': frame_idx,
                'timestamp': frame_data['timestamp'] if isinstance(frame_data, dict) else float(frame_idx),
                'persons': frame_persons
            })
            
        return {
            'unique_count': len(self.unique_track_ids),
            'per_frame': results
        }