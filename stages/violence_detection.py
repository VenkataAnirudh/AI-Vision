"""
VisionAI — Violence Detection (Temporal VideoMAE Classifier)
─────────────────────────────────────────────────────────────
Replaces the single-frame YOLO "Sentinel" detector with a video-clip
classifier. Violence is a motion pattern over time, so a 16-frame clip
model (VideoMAEForVideoClassification) is the correct modality.

Model id is config-driven (config['models']['violence']['model_id']) so a
stronger / fine-tuned checkpoint can be swapped in without code changes.
"""

import cv2
import torch
import numpy as np

from ultralytics import YOLO
from stages.stgcn_model import STGCN, skeleton_clip_from_frames, skeleton_clip_from_cache


class ViolenceSTGCN:
    """Fine-tuned skeleton ST-GCN violence classifier (binary).

    Runs YOLO pose over a clip → COCO-17 skeleton sequence → ST-GCN → softmax
    probability of the 'violence' class. Drop-in for the pipeline's violence pass
    (same ``classify_clip(clip_frames)`` contract as the VideoMAE classifier).
    """

    def __init__(self, model_manager, config):
        self.config = config['models']['violence']
        self.manager = model_manager

        self.class_names = None
        self.violent_idx = 1
        self.clip_len = 32
        self.max_persons = 2
        self.pose_conf = 0.30
        self.img_norm = True

    def _get_model(self):
        def loader():
            ckpt = torch.load(self.config['model_path'], map_location='cpu', weights_only=False)
            cfg = ckpt.get('config', {})
            self.class_names = ckpt['class_names']
            self.violent_idx = self.class_names.index('violence') if 'violence' in self.class_names else 1
            self.clip_len = int(cfg.get('clip_len', 32))
            self.max_persons = int(cfg.get('max_persons', 2))
            self.pose_conf = float(cfg.get('pose_conf', 0.30))
            self.img_norm = bool(cfg.get('img_norm', True))
            model = STGCN(in_ch=3, num_classes=len(self.class_names))
            model.load_state_dict(ckpt['model_state_dict'])
            return model
        # ST-GCN coexists with the pose model in VRAM; fp32 (BatchNorm1d is fragile in fp16).
        return self.manager.load_torch_model('violence_stgcn', loader, keep_previous=True, skip_fp16=True)

    def _get_pose_model(self):
        pose_path = self.config.get('pose_model_path', 'models/weights/yolo11x-pose.pt')
        return self.manager.load_torch_model('violence_pose', lambda: YOLO(pose_path), keep_previous=True)

    def classify_clip(self, clip_frames, frame_skeletons=None, frame_keys=None, frame_hw=None):
        """Return the softmax probability of the 'violence' class (float 0-1).

        Fast path: when ``frame_skeletons`` (a shared per-frame keypoint cache) is supplied,
        the clip tensor is built from cached skeletons (no pose inference). Otherwise falls
        back to running YOLO pose over ``clip_frames``.
        """
        model = self._get_model()
        if frame_skeletons is not None:
            if not frame_keys:
                return 0.0
            H, W = frame_hw if frame_hw else (1.0, 1.0)
            x = skeleton_clip_from_cache(frame_skeletons, frame_keys, H, W,
                                         clip_len=self.clip_len, max_persons=self.max_persons,
                                         img_norm=self.img_norm)
        else:
            if not clip_frames:
                return 0.0
            pose_model = self._get_pose_model()
            use_half = self.manager.device.type == 'cuda'
            x = skeleton_clip_from_frames(clip_frames, pose_model, clip_len=self.clip_len,
                                          max_persons=self.max_persons, pose_conf=self.pose_conf,
                                          img_norm=self.img_norm, use_half=use_half)
        if x is None:
            return 0.0
        x = x.to(self.manager.device).float()
        with torch.no_grad():
            probs = torch.softmax(model(x).float(), dim=-1)[0]
        return float(probs[self.violent_idx].item())

    def unload(self):
        self.manager.unload('violence_stgcn')
        self.manager.unload('violence_pose')


class ViolenceClassifier:
    def __init__(self, model_manager, config):
        """
        Args:
            model_manager: ModelManager instance
            config: Full config dict
        """
        self.config = config['models']['violence']
        self.manager = model_manager

        self.model_id = self.config['model_id']
        self.revision = self.config.get('revision', 'main')
        self.num_frames = int(self.config.get('num_frames', 16))

        self.processor = None
        self._violent_idx = None

    def _ensure_processor(self):
        if self.processor is None:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)

    def _get_model(self):
        def loader():
            from transformers import AutoModelForVideoClassification
            model = AutoModelForVideoClassification.from_pretrained(
                self.model_id, revision=self.revision, dtype=torch.float16
            )
            self._remap_attention_bias(model)
            return model
        return self.manager.load_torch_model('violence_videomae', loader)

    def _remap_attention_bias(self, model):
        """transformers >=5 refactored VideoMAE attention from separate
        ``q_bias``/``v_bias`` parameters to standard ``query/key/value.bias``.
        Pre-v5 checkpoints therefore lose their trained attention biases on load
        (silently random-initialized), which cripples the model. Re-map them from
        the raw checkpoint; ``key.bias`` is zeros to match old VideoMAE (k-bias was
        never trained)."""
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            path = hf_hub_download(self.model_id, 'model.safetensors', revision=self.revision)
            raw = load_file(path)
        except Exception as e:
            print(f"[ViolenceClassifier] Attention-bias remap skipped ({e}); model may be degraded.")
            return

        if not any(k.endswith('.q_bias') for k in raw):
            return  # already new-format checkpoint, nothing to fix

        sd = model.state_dict()
        overlay = {}
        for k, v in raw.items():
            if k.endswith('.attention.attention.q_bias'):
                base = k[:-len('q_bias')]
                overlay[base + 'query.bias'] = v
                overlay[base + 'key.bias'] = torch.zeros_like(v)
            elif k.endswith('.attention.attention.v_bias'):
                base = k[:-len('v_bias')]
                overlay[base + 'value.bias'] = v

        fixed = {name: t.to(sd[name].dtype) for name, t in overlay.items() if name in sd}
        model.load_state_dict(fixed, strict=False)
        print(f"[ViolenceClassifier] Re-mapped {len(fixed)} attention-bias tensors from pretrained checkpoint.")

    def _resolve_violent_index(self, model):
        """Find the output index of the 'violent' class from id2label (robust to swapped checkpoints)."""
        if self._violent_idx is not None:
            return self._violent_idx
        idx = 1  # sensible binary default
        id2label = getattr(model.config, 'id2label', None) or {}
        for k, label in id2label.items():
            if 'viol' in str(label).lower():
                idx = int(k)
                break
        self._violent_idx = idx
        return idx

    def _sample_frames(self, clip_frames):
        """Uniformly sample/pad a list of BGR frames to exactly num_frames."""
        n = len(clip_frames)
        if n == 0:
            return []
        if n == self.num_frames:
            return clip_frames
        idxs = np.linspace(0, n - 1, self.num_frames).astype(int)
        return [clip_frames[i] for i in idxs]

    def classify_clip(self, clip_frames):
        """
        Run VideoMAE on a clip of BGR numpy frames.
        Returns the softmax probability of the 'violent' class (float 0-1).
        """
        if not clip_frames:
            return 0.0

        self._ensure_processor()
        model = self._get_model()
        violent_idx = self._resolve_violent_index(model)

        frames = self._sample_frames(clip_frames)
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

        inputs = self.processor(rgb_frames, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.manager.device)

        param = next(model.parameters())
        pixel_values = pixel_values.to(param.dtype)

        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits
            probs = torch.softmax(logits.float(), dim=-1)[0]

        return float(probs[violent_idx].item())

    def unload(self):
        self.manager.unload('violence_videomae')
