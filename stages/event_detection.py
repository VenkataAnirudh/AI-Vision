"""
VisionAI — Event Detection (Fire/Smoke)
───────────────────────────────────────────────────
Fire/Smoke:  YOLOv8s / firedetect-11s.pt (with Volumetric & Co-occurrence filters)

Violence is handled separately by the temporal VideoMAE classifier in
stages/violence_detection.py.
"""

import math
from ultralytics import YOLO


class EventDetector:
    def __init__(self, model_manager, config):
        """
        Args:
            model_manager: ModelManager instance
            config: Full config dict
        """
        self.config = config['models']
        self.manager = model_manager

        self.fire_box_history = []

    def _get_fire_model(self):
        return self.manager.load_torch_model(
            'fire_yolo',
            lambda: YOLO(self.config['fire']['model'])
        )

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def _get_optimal_imgsz(self, frame):
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        imgsz = max(320, min(640, int(round(max_dim / 32) * 32)))
        return imgsz

    def detect_fire(self, frame):
        """Detect fire/smoke with mathematical filters."""
        model = self._get_fire_model()
        imgsz = self._get_optimal_imgsz(frame)
        results = model.predict(frame, imgsz=imgsz, conf=self.config['fire']['confidence'], verbose=False, half=True)[0]

        events = []
        fire_boxes = []
        smoke_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            xyxy = [int(x) for x in box.xyxy[0]]
            conf = float(box.conf[0])
            name = results.names[cls_id].lower()

            if 'fire' in name:
                fire_boxes.append({'bbox': xyxy, 'conf': conf})
            elif 'smoke' in name:
                smoke_boxes.append({'bbox': xyxy, 'conf': conf})
            
            events.append({
                'type': name,
                'confidence': conf,
                'bbox': xyxy
            })

        
        
        
        if len(fire_boxes) > 0:
            largest_fire = max(fire_boxes, key=lambda b: (b['bbox'][2]-b['bbox'][0])*(b['bbox'][3]-b['bbox'][1]))
            w = largest_fire['bbox'][2] - largest_fire['bbox'][0]
            h = largest_fire['bbox'][3] - largest_fire['bbox'][1]
            current_area = w * h
            
            self.fire_box_history.append(current_area)
            if len(self.fire_box_history) > 10:
                self.fire_box_history.pop(0)
                
            if len(self.fire_box_history) == 10 and current_area > self.fire_box_history[0] * 2:
                events.append({
                    'type': 'explosion_flash_fire',
                    'confidence': 1.0,
                    'severity': 'CRITICAL',
                    'bbox': largest_fire['bbox'],
                    'details': 'Rapid volumetric expansion detected'
                })

        
        
        
        for fb in fire_boxes:
            for sb in smoke_boxes:
                
                iou = self._compute_iou(fb['bbox'], sb['bbox'])
                
                
                cx_f = (fb['bbox'][0] + fb['bbox'][2]) / 2
                cy_f = (fb['bbox'][1] + fb['bbox'][3]) / 2
                cx_s = (sb['bbox'][0] + sb['bbox'][2]) / 2
                cy_s = (sb['bbox'][1] + sb['bbox'][3]) / 2
                dist = math.hypot(cx_f - cx_s, cy_f - cy_s)
                
                if iou > 0.05 or dist < 150:
                    events.append({
                        'type': 'verified_fire_hazard',
                        'confidence': max(fb['conf'], sb['conf']),
                        'severity': 'HIGH',
                        'bbox': fb['bbox'],
                        'details': 'Smoke and Flame co-occurrence validated'
                    })
                    break

        return events