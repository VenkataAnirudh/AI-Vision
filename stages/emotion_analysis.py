import cv2
import numpy as np
import torch

class EmotionAnalyzer:
    def __init__(self, config, model_manager=None):
        self.config = config['models']['emotion']
        self.manager = model_manager
        self.brow_baselines = {}
        self.emotion_ema = {}                                  # per-track smoothed prob vectors
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # FER CNN metadata, populated lazily on first model load.
        self.class_names = None
        self.img_size = 160
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _get_model(self):
        """Lazily build + load the fine-tuned timm FER classifier via ModelManager."""
        def loader():
            import timm
            ckpt = torch.load(self.config['model_path'], map_location='cpu', weights_only=False)
            cfg = ckpt.get('config', {})
            self.class_names = ckpt['class_names']
            self.img_size = int(cfg.get('img_size', 160))
            self.mean = cfg.get('mean', self.mean)
            self.std = cfg.get('std', self.std)
            model = timm.create_model(cfg.get('model_name', 'efficientnet_b0'),
                                      pretrained=False, num_classes=len(self.class_names))
            model.load_state_dict(ckpt['model_state_dict'])
            return model
        # keep_previous so we coexist with the face model; fp32 (tiny per-face model).
        return self.manager.load_torch_model('emotion_fer', loader,
                                             keep_previous=True, skip_fp16=True)

    def _emotion_probs(self, face_crop_bgr):
        """Run the FER CNN on a BGR face crop with low-resolution hardening.

        Returns ``(probs, quality)`` where ``probs`` is the softmax over ``class_names`` and
        ``quality`` ∈ [0, 1] blends a blur measure (Laplacian variance) with crop size. Low
        quality flattens the distribution toward uniform so unreliable CCTV faces cannot
        produce a high-confidence emotion that falsely drives escalation.
        """
        model = self._get_model()
        n_cls = len(self.class_names)

        # Quality estimate on the raw crop (before any enhancement).
        gray0 = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        h0, w0 = gray0.shape[:2]
        blur_var = float(cv2.Laplacian(gray0, cv2.CV_64F).var())
        blur_q = min(1.0, blur_var / 120.0)                    # ~120+ variance ≈ sharp
        size_q = min(1.0, min(h0, w0) / float(self.img_size))
        quality = float(np.clip(0.5 * blur_q + 0.5 * size_q, 0.05, 1.0))

        # Low-res enhancement: CLAHE contrast + light denoise; resolution-aware interpolation.
        gray = self._clahe.apply(gray0)
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
        rgb3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)          # grayscale replicated to 3ch
        interp = cv2.INTER_AREA if (h0 > self.img_size or w0 > self.img_size) else cv2.INTER_CUBIC
        resized = cv2.resize(rgb3, (self.img_size, self.img_size), interpolation=interp)

        def _to_tensor(img):
            a = img.astype(np.float32) / 255.0
            a = (a - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
            return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

        # Horizontal-flip test-time augmentation (faces ~symmetric for emotion) → mean softmax.
        batch = torch.cat([_to_tensor(resized), _to_tensor(cv2.flip(resized, 1))], dim=0)
        batch = batch.to(self.manager.device).float()
        with torch.no_grad():
            probs = torch.softmax(model(batch).float(), dim=-1).mean(dim=0).cpu().numpy()

        # Temperature-soften toward uniform by (1 - quality).
        uniform = np.full(n_cls, 1.0 / n_cls, dtype=np.float32)
        probs = quality * probs + (1.0 - quality) * uniform
        probs = probs / probs.sum()
        return probs.astype(np.float32), quality

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

    def analyze_face(self, frame, face_obj, track_id=None):
        """
        Analyzes a face object from InsightFace for dominant emotion (DeepFace)
        and visual cry/stress heuristics using 106 landmarks.
        """
        bbox = face_obj.bbox
        x1, y1, x2, y2 = [int(x) for x in bbox]
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        
        face_crop = frame[y1:y2, x1:x2]
        dominant_emotion = "neutral"
        emotion_scores = {}

        if face_crop.size > 0 and (x2 - x1) >= self.config['min_face_size_px'] and self.manager is not None:
            try:
                probs, _quality = self._emotion_probs(face_crop)
                # Per-track temporal EMA over the prob vector → stable label across low-res flicker.
                if track_id is not None:
                    prev = self.emotion_ema.get(track_id)
                    if prev is not None and prev.shape == probs.shape:
                        probs = 0.6 * prev + 0.4 * probs
                    self.emotion_ema[track_id] = probs
                dominant_emotion = self.class_names[int(probs.argmax())]
                emotion_scores = {c: round(float(p) * 100.0, 2) for c, p in zip(self.class_names, probs)}
            except Exception:
                pass

        
        
        landmark = getattr(face_obj, 'landmark', None)
        
        cry_prob = 0.0
        stress_score = 0.0

        if landmark is not None and landmark.shape[0] == 106:
            
            left_eye_indices = [35, 37, 38, 39, 41, 40]
            right_eye_indices = [89, 91, 92, 93, 95, 96]
            
            
            ear_left = self._eye_aspect_ratio_106(landmark, left_eye_indices)
            ear_right = self._eye_aspect_ratio_106(landmark, right_eye_indices)
            ear = (ear_left + ear_right) / 2.0
            
            
            
            cry_prob = max(0.0, min(1.0, (0.28 - ear) / 0.11))
            
            
            
            inner_brows = np.linalg.norm(landmark[48] - landmark[102])
            face_width = float(x2 - x1)
            if face_width > 0:
                brow_ratio = inner_brows / face_width
                if track_id is not None:
                    if track_id not in self.brow_baselines:
                        self.brow_baselines[track_id] = []
                    self.brow_baselines[track_id].append(brow_ratio)
                    
                    if len(self.brow_baselines[track_id]) >= 5:
                        baseline = np.mean(self.brow_baselines[track_id][:10]) 
                        deviation = baseline - brow_ratio 
                        stress_score = max(0.0, min(1.0, deviation * 8.0)) 
                else:
                    
                    stress_score = max(0.0, min(1.0, 1.0 - (brow_ratio * 4.5)))
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_scores': emotion_scores,
            'visual_cry_prob': cry_prob,
            'visual_stress_score': stress_score
        }