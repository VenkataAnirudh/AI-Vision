import torch
import json
from ultralytics import YOLO
import torchvision.models.video as models_video
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

class EventDetector:
    def __init__(self, model_manager, config):
        self.config = config['models']
        self.manager = model_manager
        
        # Pre-load Kinetics labels
        with open(self.config['violence']['kinetics_labels'], 'r') as f:
            kinetics_labels = json.load(f)
            
        self.violence_ids = [
            kinetics_labels[c] for c in self.config['violence']['violence_classes'] 
            if c in kinetics_labels
        ]

    def _get_fire_model(self):
        return self.manager.load_torch_model(
            'fire_yolo', 
            lambda: YOLO(self.config['fire']['model'])
        )

    def _get_violence_model(self):
        def load_r3d():
            model = models_video.r3d_18(weights=None)
            weights_path = 'models/weights/r3d_18-b3b3357e.pth'
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            return model
        return self.manager.load_torch_model('violence_r3d', load_r3d)

    def detect_fire(self, frame):
        model = self._get_fire_model()
        results = model(frame, conf=self.config['fire']['confidence'], verbose=False)[0]
        
        events = []
        for box in results.boxes:
            events.append({
                'type': 'fire/smoke',
                'confidence': float(box.conf[0]),
                'bbox': [int(x) for x in box.xyxy[0]]
            })
        return events

    def detect_violence(self, clip_tensor):
        """Expects a tensor of shape [1, 3, 16, 112, 112]"""
        model = self._get_violence_model()
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = model(clip_tensor.to(self.manager.device))
                probs = torch.nn.functional.softmax(output, dim=1)[0]
        
        violence_prob = sum([probs[idx].item() for idx in self.violence_ids])
        
        if violence_prob >= self.config['violence']['confidence_threshold']:
            return {'type': 'violence', 'confidence': violence_prob, 'severity': 'HIGH'}
        return None