"""
VisionAI — Dual-Head R3D-18 Architecture
─────────────────────────────────────────
Matches the exact training architecture for the fine-tuned dual-head model:
  kinetics_head: nn.Linear(512, 400)  — Kinetics-400 classes (frozen at inference)
  custom_head:   Sequential(Dropout → Linear → ReLU → Dropout → Linear) — 7 custom indoor actions

State dict keys use: stem, layer1-layer4, kinetics_head, custom_head
This module is shared between IndoorActionDetector (custom_head) and EventDetector (kinetics_head).
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class DualHeadR3D18(nn.Module):
    """
    Dual-head R3D-18 model for combined Kinetics-400 + custom indoor action classification.
    
    Architecture:
        - R3D-18 backbone (stem + layer1-4 + avgpool)
        - kinetics_head: Linear(512, 400) — Kinetics-400 
        - custom_head: Dropout(0.4) → Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 7)
    """
    
    NUM_KINETICS_CLASSES = 400
    NUM_CUSTOM_CLASSES = 7
    
    def __init__(self):
        super().__init__()
        
        
        base = r3d_18(weights=None)
        
        
        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        
        
        self.kinetics_head = nn.Linear(512, self.NUM_KINETICS_CLASSES)
        
        
        self.custom_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.NUM_CUSTOM_CLASSES),
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 3, T, H, W] — batch of video clips
            
        Returns:
            tuple: (kinetics_logits, custom_logits)
                - kinetics_logits: [B, 400] Kinetics-400 class scores
                - custom_logits:   [B, 7]   Custom indoor action scores
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)  
        
        kinetics_logits = self.kinetics_head(x)
        custom_logits = self.custom_head(x)
        
        return kinetics_logits, custom_logits
