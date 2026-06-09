import torch
import cv2
import numpy as np
from torchvision.models.video import r3d_18
import torch.nn as nn

class IndoorActionDetector:
    def __init__(self, model_manager, config):
        self.config = config['models']['indoor_action']
        self.manager = model_manager
        self.classes = self.config['classes']
        self.alert_indices = self.config['alert_class_indices']
        
        # Load the custom fine-tuned weights
        self.model = self.manager.load_torch_model(
            'indoor_action_r3d', 
            self._load_model
        )

    def _load_model(self):
        # Must match the architecture from train_indoor_action.py
        model = r3d_18(weights=None)
        model.fc = nn.Linear(512, len(self.classes))
        
        try:
            checkpoint = torch.load(self.config['model_path'], map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"[Warning] Failed to load fine-tuned action model. Did training finish? Error: {e}")
            
        return model

    def process_clip(self, frames):
        """Expects a list of exact 16 frames as a motion-triggered clip"""
        if len(frames) != self.config['clip_frames']:
            return None
            
        # Preprocess to match training logic
        processed = []
        for frame in frames:
            resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                               (self.config['frame_size'], self.config['frame_size']))
            processed.append(resized)
            
        clip = np.stack(processed).astype(np.float32) / 255.0
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0) # [1, C, T, H, W]
        
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
        clip = (clip - mean) / std

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = self.model(clip.to(self.manager.device))
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        
        if confidence >= self.config['confidence_threshold']:
            is_alert = pred_idx in self.alert_indices
            return {
                'action': self.classes[pred_idx],
                'confidence': confidence,
                'is_alert': is_alert
            }
        return None