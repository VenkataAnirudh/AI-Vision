"""
VisionAI — Behavioral Context Engine (Kinetics-400 R3D-18)
───────────────────────────────────────────────────────────────
Acts as a secondary Behavioral Context Layer to catch micro-actions
and serve as a false-positive shield for the primary spatial YOLO models.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np

class BehavioralContextEngine:
    def __init__(self, model_manager, config, shared_model=None):
        self.config = config['models']['indoor_action']
        self.manager = model_manager
        self.model = shared_model
        
        self.kinetics_classes = {
            
            80: 'crying',
            180: 'laughing',
            
            151: 'headbutting',
            315: 'slapping',
            260: 'punching person',
            393: 'wrestling',
            
            158: 'hugging',
            289: 'shaking hands',
            357: 'tickling',
            
            92: 'dining',
            265: 'reading book',
            371: 'using computer'
        }
        
        self.custom_head_classes = {
            0: 'falling_down',
            1: 'lying_on_floor',
            2: 'no_action',
            3: 'sitting_down',
            4: 'standing_up',
            5: 'walking',
            6: 'watching_tv'
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _preprocess_clip(self, clip_frames):
        """Prepares a list of 16 numpy frames for R3D-18."""
        frames = []
        for frame in clip_frames:
            img = cv2.resize(frame, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            
        
        video_tensor = np.array(frames, dtype=np.float32)
        video_tensor = video_tensor.transpose((3, 0, 1, 2))
        
        video_tensor /= 255.0
        mean = np.array([0.43216, 0.394666, 0.37645]).reshape(3, 1, 1, 1)
        std = np.array([0.22803, 0.22145, 0.216989]).reshape(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        return torch.tensor(video_tensor).unsqueeze(0).to(self.device).half()

    def analyze_clip(self, clip_frames):
        """
        Runs R3D-18 on a 16-frame clip and returns filtered contexts.
        Returns a dict of active states.
        """
        if self.model is None or len(clip_frames) < 16:
            return {}
            
        tensor = self._preprocess_clip(clip_frames)
        
        with torch.no_grad():
            kinetics_logits, custom_logits = self.model(tensor)
            
            k_probs = F.softmax(kinetics_logits, dim=1)[0]
            c_probs = F.softmax(custom_logits, dim=1)[0]
            
        active_states = {
            'fp_shields': [],
            'micro_violence': [],
            'essential': [],
            'enrichment': [],
            'indoor_states': []
        }
        
        CONF_THRESH = 0.60 
        
        
        for class_id, label in self.kinetics_classes.items():
            conf = k_probs[class_id].item()
            if conf > CONF_THRESH:
                event_data = {'type': label, 'confidence': conf}
                if class_id in [158, 289, 357]:
                    active_states['fp_shields'].append(event_data)
                elif class_id in [151, 315, 260, 393]:
                    active_states['micro_violence'].append(event_data)
                elif class_id in [80, 180]:
                    active_states['essential'].append(event_data)
                elif class_id in [92, 265, 371]:
                    active_states['enrichment'].append(event_data)
                    
        
        for class_id, label in self.custom_head_classes.items():
            if class_id >= len(c_probs): continue
            conf = c_probs[class_id].item()
            if conf > CONF_THRESH and label != 'no_action':
                active_states['indoor_states'].append({'type': label, 'confidence': conf})
                
        return active_states
