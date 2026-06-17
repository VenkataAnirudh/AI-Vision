"""
VisionAI — Shared ST-GCN model + skeleton preprocessing
───────────────────────────────────────────────────────────────
Verbatim port of the spatial-temporal graph conv network trained in the
`01_violence_stgcn` and `02_home_action_stgcn` Colab notebooks (COCO-17 graph).

Both fine-tuned checkpoints share identical state-dict keys (dropout carries no
parameters), so one class loads either one. Read ``num_classes`` / ``max_persons``
/ ``class_names`` from the checkpoint rather than hardcoding.
"""

import numpy as np
import torch
import torch.nn as nn


COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9),
              (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
NUM_JOINTS = 17


def build_adjacency():
    A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for i, j in COCO_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += np.eye(NUM_JOINTS, dtype=np.float32)
    deg = A.sum(1)
    Dinv = np.diag(1.0 / np.maximum(deg, 1e-6)).astype(np.float32)
    return (Dinv @ A).astype(np.float32)


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, residual=True):
        super().__init__()
        self.register_buffer("A", torch.tensor(A))
        self.edge_imp = nn.Parameter(torch.ones_like(self.A))
        self.gcn = nn.Conv2d(cin, cout, 1)
        self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)), nn.BatchNorm2d(cout))
        if not residual:
            self.res = None
        elif cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = 0 if self.res is None else self.res(x)
        x = self.gcn(x)
        x = torch.einsum("nctv,vw->nctw", x, self.A * self.edge_imp)
        x = self.tcn(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        A = build_adjacency()
        self.data_bn = nn.BatchNorm1d(in_ch * NUM_JOINTS)
        self.layers = nn.ModuleList([
            STGCNBlock(in_ch, 64, A, residual=False), STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2), STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A, stride=2), STGCNBlock(256, 256, A)])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 1, 3, 2).contiguous().view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=[2, 3]).view(N, M, -1).mean(dim=1)
        return self.fc(self.drop(x))


def _sample_indices(n_total, n):
    if n_total <= 0:
        return []
    if n_total >= n:
        return [int(v) for v in np.linspace(0, n_total - 1, n)]
    return list(range(n_total)) + [n_total - 1] * (n - n_total)


def skeleton_clip_from_frames(frames, pose_model, clip_len=32, max_persons=1,
                              pose_conf=0.30, img_norm=True, use_half=True):
    """Turn a list of BGR frames into an ST-GCN input tensor.

    Reproduces the notebooks' ``extract_skeleton``: uniformly sample ``clip_len``
    frames, run YOLO pose (person class only), keep the ``max_persons`` highest-
    confidence skeletons per frame, normalize coords to [-1, 1] by frame size.

    Returns a float tensor of shape ``(1, 3, clip_len, 17, max_persons)`` or
    ``None`` if no frames were supplied.
    """
    if not frames:
        return None

    H, W = frames[0].shape[:2]
    W = float(W) or 1.0
    H = float(H) or 1.0

    idx_list = _sample_indices(len(frames), clip_len)
    sampled = [frames[i] for i in idx_list]

    results = pose_model.predict(sampled, conf=pose_conf, classes=[0],
                                 verbose=False, half=use_half)

    clip = np.zeros((clip_len, max_persons, NUM_JOINTS, 3), dtype=np.float32)
    for t, res in enumerate(results):
        if res.keypoints is None or res.keypoints.data is None:
            continue
        kp = res.keypoints.data.cpu().numpy()
        if kp.shape[0] == 0:
            continue
        order = np.argsort(-kp[:, :, 2].mean(axis=1))[:max_persons]
        for m, pidx in enumerate(order):
            person = kp[pidx].copy()
            if img_norm:
                person[:, 0] = (person[:, 0] / W - 0.5) * 2.0
                person[:, 1] = (person[:, 1] / H - 0.5) * 2.0
            clip[t, m] = person

    # (T, M, V, C) -> (C, T, V, M), add batch dim
    x = torch.from_numpy(clip).permute(3, 0, 2, 1).contiguous().unsqueeze(0)
    return x


def skeleton_clip_from_cache(frame_skeletons, frame_keys, H, W, clip_len=32,
                             max_persons=1, img_norm=True):
    """Build an ST-GCN input tensor from precomputed per-frame keypoints (no pose call).

    ``frame_skeletons`` maps a frame key -> ndarray ``(K, 17, 3)`` of COCO-17 keypoints in
    frame coordinates (already sorted best-first). ``frame_keys`` is the ordered list of keys
    for this clip's window. Mirrors ``skeleton_clip_from_frames`` (uniform sample, top-
    ``max_persons``, normalize by frame size) and returns ``(1, 3, clip_len, 17, max_persons)``
    or ``None``.
    """
    if not frame_keys:
        return None

    W = float(W) or 1.0
    H = float(H) or 1.0

    idx_list = _sample_indices(len(frame_keys), clip_len)
    sampled_keys = [frame_keys[i] for i in idx_list]

    clip = np.zeros((clip_len, max_persons, NUM_JOINTS, 3), dtype=np.float32)
    for t, key in enumerate(sampled_keys):
        kp = frame_skeletons.get(key)
        if kp is None or len(kp) == 0:
            continue
        for m in range(min(max_persons, kp.shape[0])):
            person = kp[m].astype(np.float32).copy()
            if img_norm:
                person[:, 0] = (person[:, 0] / W - 0.5) * 2.0
                person[:, 1] = (person[:, 1] / H - 0.5) * 2.0
            clip[t, m] = person

    x = torch.from_numpy(clip).permute(3, 0, 2, 1).contiguous().unsqueeze(0)
    return x
