"""
VisionAI — Weapon Detection (Firearm / Knife)
───────────────────────────────────────────────────────────────
A YOLO object detector that localizes weapons. Community weapon models are
trained on small, biased datasets and false-positive on benign indoor scenes,
so this stage is only the *raw* detector — it is meant to be used as a gated
escalator: the pipeline associates each weapon to a person, requires temporal
persistence, and escalates "raised above head". The detection itself never
raises an alert on its own.

Detector ids / thresholds live in config (``config['models']['weapon']``) so a
better checkpoint is a config change, not a code change. Multiple detectors may
be configured (e.g. a firearm model + a knife model); they are loaded and
unloaded sequentially so only one is resident in VRAM at a time.
"""

import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# Raw class name (lowercased, substring match) → normalized weapon category.
_FIREARM_TOKENS = ('gun', 'pistol', 'rifle', 'firearm', 'shotgun', 'handgun', 'weapon')
_KNIFE_TOKENS = ('knife', 'blade', 'dagger', 'machete')


class WeaponDetector:
    def __init__(self, model_manager, config):
        """
        Args:
            model_manager: ModelManager instance
            config: Full config dict
        """
        self.config = config['models']['weapon']
        self.manager = model_manager

        # Normalize to a list of detector specs. Each: {name, model_id, filename, confidence}.
        self.detectors = list(self.config.get('detectors', []))

    def _normalize_class(self, raw_name):
        """Map a raw model class name to 'firearm' | 'knife', or None to drop it."""
        n = str(raw_name).lower()
        if any(tok in n for tok in _KNIFE_TOKENS):
            return 'knife'
        if any(tok in n for tok in _FIREARM_TOKENS):
            return 'firearm'
        return None

    def _get_model(self, spec):
        def loader():
            model_id = spec['model_id']
            if os.path.exists(model_id):
                path = model_id
            else:
                path = hf_hub_download(model_id, spec.get('filename', 'weights/best.pt'))
            return YOLO(path)
        return self.manager.load_torch_model(spec['name'], loader)

    def detect(self, frame):
        """Run all configured weapon detectors on a BGR frame.

        Returns a list of {'weapon_class','bbox':[x1,y1,x2,y2],'confidence'} dicts
        (only weapon classes; non-weapon classes such as 'grenade'/'person' dropped).
        """
        out = []
        for spec in self.detectors:
            model = self._get_model(spec)
            conf = float(spec.get('confidence', 0.45))
            res = model.predict(frame, conf=conf, verbose=False, half=True)[0]
            for box in res.boxes:
                cat = self._normalize_class(model.names[int(box.cls[0])])
                if cat is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                out.append({
                    'weapon_class': cat,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf[0]),
                })
        return out

    def unload(self):
        for spec in self.detectors:
            self.manager.unload(spec['name'])
