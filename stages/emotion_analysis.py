import cv2
import numpy as np
from deepface import DeepFace

class EmotionAnalyzer:
    def __init__(self, config):
        self.config = config['models']['emotion']

    def _eye_aspect_ratio_106(self, landmark, eye_indices):
        """
        Calculates Eye Aspect Ratio (EAR) using 106 facial landmarks.
        eye_indices: [left_corner, top1, top2, right_corner, bot1, bot2]
        """
        p_left = landmark[eye_indices[0]]
        p_top1 = landmark[eye_indices[1]]
        p_top2 = landmark[eye_indices[2]]
        p_right = landmark[eye_indices[3]]
        p_bot1 = landmark[eye_indices[4]]
        p_bot2 = landmark[eye_indices[5]]
        
        A = np.linalg.norm(p_top1 - p_bot1)
        B = np.linalg.norm(p_top2 - p_bot2)
        C = np.linalg.norm(p_left - p_right)
        
        if C < 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)

    def analyze_face(self, frame, face_obj):
        """
        Analyzes a face object from InsightFace for dominant emotion (DeepFace)
        and visual cry/stress heuristics using 106 landmarks.
        """
        bbox = face_obj.bbox
        x1, y1, x2, y2 = [int(x) for x in bbox]
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 1. DeepFace Emotion (CPU fallback)
        face_crop = frame[y1:y2, x1:x2]
        emotion_result = {"dominant_emotion": "neutral", "emotion": {}}
        
        # Check if crop size is sufficient
        if face_crop.size > 0 and (x2 - x1) >= self.config['min_face_size_px']:
            try:
                res = DeepFace.analyze(
                    img_path=face_crop,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                emotion_result = res[0]
            except Exception as e:
                # Fallback on inference error
                pass

        # 2. Extract Landmarks (InsightFace provides 106 points in face_obj.landmark)
        # face_obj.landmark is expected to be a (106, 2) numpy array
        landmark = getattr(face_obj, 'landmark', None)
        
        cry_prob = 0.0
        stress_score = 0.0

        if landmark is not None and landmark.shape[0] == 106:
            # Indices for eye points
            left_eye_indices = [35, 37, 38, 39, 41, 40]
            right_eye_indices = [89, 91, 92, 93, 95, 96]
            
            # Compute EAR
            ear_left = self._eye_aspect_ratio_106(landmark, left_eye_indices)
            ear_right = self._eye_aspect_ratio_106(landmark, right_eye_indices)
            ear = (ear_left + ear_right) / 2.0
            
            # Cry Heuristic (low EAR = eyes closed or squeezed)
            cry_prob = 1.0 if ear < 0.22 else 0.0
            
            # Stress Heuristic (brow narrowing relative to face width)
            # Left inner brow index: 48, Right inner brow index: 102
            inner_brows = np.linalg.norm(landmark[48] - landmark[102])
            face_width = float(x2 - x1)
            if face_width > 0:
                brow_ratio = inner_brows / face_width
                # Squeezed brows (smaller ratio) maps to higher stress score
                stress_score = max(0.0, min(1.0, 1.0 - (brow_ratio * 4.5)))
        
        return {
            'dominant_emotion': emotion_result.get('dominant_emotion', 'neutral'),
            'emotion_scores': emotion_result.get('emotion', {}),
            'visual_cry_prob': cry_prob,
            'visual_stress_score': stress_score
        }