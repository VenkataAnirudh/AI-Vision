import torch
import cv2
import numpy as np

CLIP_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
CLIP_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

def preprocess_clip(frames, size=112):
    processed = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (size, size)) for f in frames]
    clip = torch.from_numpy(np.stack(processed).astype(np.float32) / 255.0)
    return ((clip.permute(3, 0, 1, 2).unsqueeze(0) - CLIP_MEAN) / CLIP_STD)
