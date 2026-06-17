"""
VisionAI — Structural Pose Engine (Indoor Actions)
───────────────────────────────────────────────────────────────
Uses YOLO11x-Pose to detect falls, bent over states, unresponsive 
individuals, and defensive guard stances.
"""

import cv2
import math
import numpy as np
import torch
from ultralytics import YOLO

from stages.stgcn_model import STGCN, skeleton_clip_from_frames, skeleton_clip_from_cache


class IndoorActionSTGCN:
    """Fine-tuned skeleton ST-GCN home-action classifier (10-class).

    Runs YOLO pose over a clip → COCO-17 skeleton sequence → ST-GCN → softmax.
    Returns only the alert classes (falling_down / lying_on_floor) so it slots
    into the pipeline alongside the geometric ``IndoorActionDetector`` pass.
    The checkpoint vocab uses ``lying_on_the_floor``; this is remapped to the
    app vocab ``lying_on_floor`` so existing consumers (escalation, chapters)
    keep working unchanged.
    """

    _NAME_MAP = {'lying_on_the_floor': 'lying_on_floor'}

    def __init__(self, model_manager, config):
        self.config = config['models']['indoor_action']
        self.manager = model_manager

        self.class_names = None
        self.alert_class_indices = []
        self.clip_len = 32
        self.max_persons = 1
        self.pose_conf = 0.30
        self.img_norm = True
        self.conf_thresh = float(self.config.get('confidence_threshold', 0.65))

    def _get_model(self):
        def loader():
            ckpt = torch.load(self.config['stgcn_model_path'], map_location='cpu', weights_only=False)
            cfg = ckpt.get('config', {})
            self.class_names = ckpt['class_names']
            self.alert_class_indices = ckpt.get('alert_class_indices', [])
            self.clip_len = int(cfg.get('clip_len', 32))
            self.max_persons = int(cfg.get('max_persons', 1))
            self.pose_conf = float(cfg.get('pose_conf', 0.30))
            self.img_norm = bool(cfg.get('img_norm', True))
            model = STGCN(in_ch=3, num_classes=len(self.class_names))
            model.load_state_dict(ckpt['model_state_dict'])
            return model
        # ST-GCN coexists with the pose model in VRAM; fp32 (BatchNorm1d is fragile in fp16).
        return self.manager.load_torch_model('home_action_stgcn', loader, keep_previous=True, skip_fp16=True)

    def _get_pose_model(self):
        pose_path = self.config.get('pose_model_path', self.config.get('model_path', 'models/weights/yolo11x-pose.pt'))
        return self.manager.load_torch_model('indoor_action_pose', lambda: YOLO(pose_path), keep_previous=True)

    def classify_clip(self, clip_frames, frame_skeletons=None, frame_keys=None, frame_hw=None):
        """Return ``{'class_name', 'confidence'}`` if the clip's top class is an
        alert action above threshold, else ``None``.

        Fast path: when ``frame_skeletons`` (a shared per-frame keypoint cache) is supplied,
        the clip tensor is built from cached skeletons (no pose inference). Otherwise falls
        back to running YOLO pose over ``clip_frames``.
        """
        model = self._get_model()
        if frame_skeletons is not None:
            if not frame_keys:
                return None
            H, W = frame_hw if frame_hw else (1.0, 1.0)
            x = skeleton_clip_from_cache(frame_skeletons, frame_keys, H, W,
                                         clip_len=self.clip_len, max_persons=self.max_persons,
                                         img_norm=self.img_norm)
        else:
            if not clip_frames:
                return None
            pose_model = self._get_pose_model()
            use_half = self.manager.device.type == 'cuda'
            x = skeleton_clip_from_frames(clip_frames, pose_model, clip_len=self.clip_len,
                                          max_persons=self.max_persons, pose_conf=self.pose_conf,
                                          img_norm=self.img_norm, use_half=use_half)
        if x is None:
            return None
        x = x.to(self.manager.device).float()
        with torch.no_grad():
            probs = torch.softmax(model(x).float(), dim=-1)[0]
        cls_idx = int(probs.argmax().item())
        prob = float(probs[cls_idx].item())
        if cls_idx not in self.alert_class_indices or prob < self.conf_thresh:
            return None
        raw_name = self.class_names[cls_idx]
        return {'class_name': self._NAME_MAP.get(raw_name, raw_name), 'confidence': prob}

    def unload(self):
        self.manager.unload('home_action_stgcn')
        self.manager.unload('indoor_action_pose')


class IndoorActionDetector:
    def __init__(self, model_manager, config):
        """
        Args:
            model_manager: ModelManager instance
            config: Full config dict
        """
        self.config = config['models']['indoor_action']
        self.manager = model_manager
        
        
        self.pose_history = {}
        self.frame_count = 0

    def _get_pose_model(self):
        return self.manager.load_torch_model(
            'pose_yolo',
            lambda: YOLO(self.config['model_path'])
        )

    def process_frame(self, frame, person_bboxes=None):
        """
        Detect indoor actions using pose tracking on a single frame.
        person_bboxes: List of person bounding boxes dicts from YOLOv8 tracking.
        """
        model = self._get_pose_model()
        events = []
        self.frame_count += 1
        
        if person_bboxes is None or len(person_bboxes) == 0:
            return events

        
        crops = []
        offsets = []
        h_frame, w_frame = frame.shape[:2]
        
        for p in person_bboxes:
            
            x1, y1, x2, y2 = p['bbox']
            w = x2 - x1
            h = y2 - y1
            
            px = int(w * 0.1)
            py = int(h * 0.1)
            
            cx1 = max(0, x1 - px)
            cy1 = max(0, y1 - py)
            cx2 = min(w_frame, x2 + px)
            cy2 = min(h_frame, y2 + py)
            
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crops.append(crop)
                offsets.append((cx1, cy1))

        if len(crops) == 0:
            return events

        
        results = model.predict(crops, imgsz=self.config.get('imgsz', 256), classes=[0], conf=self.config.get('confidence_threshold', 0.5), verbose=False, half=True)

        current_persons = []
            
        for i, res in enumerate(results):
            if res.keypoints is None or len(res.keypoints.data) == 0:
                continue
                
            offset_x, offset_y = offsets[i]
            track_id = person_bboxes[i].get('track_id')
                
            for person_kpts in res.keypoints.data:
                if person_kpts.shape[0] < 17:
                    continue
                    
                kpts = person_kpts.cpu().numpy().copy() 
                
                
                kpts[:, 0] += offset_x
                kpts[:, 1] += offset_y
                
                
                nose = kpts[0]
                l_wrist, r_wrist = kpts[9], kpts[10]
                l_shoulder, r_shoulder = kpts[5], kpts[6]
                l_hip, r_hip = kpts[11], kpts[12]
                
                if min(l_shoulder[2], r_shoulder[2], l_hip[2], r_hip[2]) < 0.4:
                    continue
                    
                mid_shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2
                mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
                mid_hip_x = (l_hip[0] + r_hip[0]) / 2
                mid_hip_y = (l_hip[1] + r_hip[1]) / 2
                
                torso_height = abs(mid_hip_y - mid_shoulder_y)
                shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
                
                
                cx = (mid_shoulder_x + mid_hip_x) / 2
                cy = (mid_shoulder_y + mid_hip_y) / 2
                current_persons.append({
                    'cx': cx, 'cy': cy, 'kpts': kpts, 
                    'shoulder_y': mid_shoulder_y, 'l_wrist': l_wrist, 'r_wrist': r_wrist,
                    'shoulder_width': shoulder_width, 'torso_height': torso_height,
                    'track_id': track_id
                })
                
                
                
                
                if shoulder_width > 0:
                    aspect_ratio = torso_height / shoulder_width
                    if aspect_ratio < 0.7:
                        events.append({
                            'type': 'bent_over',
                            'confidence': 0.8,
                            'severity': 'INFO',
                            'details': 'Person is bent over or crouching',
                            'track_id': track_id
                        })
                        
                
                
                
                dx = mid_hip_x - mid_shoulder_x
                dy = mid_hip_y - mid_shoulder_y
                spine_angle_deg = np.degrees(np.arctan2(abs(dx), abs(dy)))
                
                
                if spine_angle_deg > 65 or torso_height < (max(10.0, shoulder_width) * 0.5):
                    events.append({
                        'type': 'falling_down',
                        'confidence': 0.9,
                        'severity': 'CRITICAL',
                        'details': 'Fall anomaly detected based on scale-invariant spine angle collapse',
                        'track_id': track_id
                    })

        
        
        
        new_history = {}
        for p in current_persons:
            matched_id = None
            min_dist = float('inf')
            
            
            max_link_dist = max(50.0, p['shoulder_width'] * 1.5)
            
            for tid, hist in self.pose_history.items():
                last_cx, last_cy = hist['positions'][-1]
                dist = math.hypot(p['cx'] - last_cx, p['cy'] - last_cy)
                if dist < max_link_dist and dist < min_dist:
                    min_dist = dist
                    matched_id = tid
                    
            if matched_id is None:
                matched_id = self.frame_count * 1000 + len(new_history)
                new_history[matched_id] = {'positions': [(p['cx'], p['cy'])], 'kpts_history': [p['kpts']], 'wrists': [(p['l_wrist'], p['r_wrist'])], 'sw': p['shoulder_width'], 'track_id': p['track_id']}
            else:
                hist = self.pose_history[matched_id]
                positions = hist['positions'] + [(p['cx'], p['cy'])]
                kpts_history = hist['kpts_history'] + [p['kpts']]
                wrists = hist.get('wrists', []) + [(p['l_wrist'], p['r_wrist'])]
                sw = hist.get('sw', p['shoulder_width'])
                
                
                if len(positions) > 15:
                    positions.pop(0)
                    kpts_history.pop(0)
                    wrists.pop(0)
                    
                new_history[matched_id] = {'positions': positions, 'kpts_history': kpts_history, 'wrists': wrists, 'sw': sw, 'track_id': p['track_id']}
                
                if len(kpts_history) == 15:
                    
                    all_kpts = np.array(kpts_history) 
                    xy_coords = all_kpts[:, :, :2]
                    var_x = np.var(xy_coords[:, :, 0], axis=0).mean()
                    var_y = np.var(xy_coords[:, :, 1], axis=0).mean()
                    
                    
                    sw_norm = max(10.0, sw)
                    norm_var_x = var_x / sw_norm
                    norm_var_y = var_y / sw_norm
                    
                    if norm_var_x < 0.05 and norm_var_y < 0.05:
                        events.append({
                            'type': 'lying_on_floor',
                            'confidence': 0.95,
                            'severity': 'CRITICAL',
                            'details': 'Medical Emergency / Unresponsive Individual',
                            'track_id': p['track_id']
                        })

                
                
                
                if len(wrists) >= 3:
                    l_w, r_w = p['l_wrist'], p['r_wrist']
                    if l_w[2] > 0.5 and r_w[2] > 0.5:
                        if l_w[1] < p['shoulder_y'] and r_w[1] < p['shoulder_y']:
                            recent_wrists = np.array(wrists[-3:])
                            l_w_var = np.var(recent_wrists[:, 0, :2], axis=0).sum()
                            r_w_var = np.var(recent_wrists[:, 1, :2], axis=0).sum()
                            
                            sw_norm = max(10.0, p['shoulder_width'])
                            if (l_w_var / sw_norm) > 1.0 or (r_w_var / sw_norm) > 1.0:
                                events.append({
                                    'type': 'aggressive_guard',
                                    'confidence': 0.85,
                                    'severity': 'HIGH',
                                    'details': 'Defensive guard or striking stance detected',
                                    'track_id': p['track_id']
                                })

        self.pose_history = new_history
        return events