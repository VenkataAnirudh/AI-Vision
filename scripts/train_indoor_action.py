import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.video import r3d_18


CLASS_MAPPING = {
    "blowing nose or sneezing": 0,
    "cleaning": 1,
    "eating": 2,
    "falling down": 3,
    "lying on the floor": 4,
    "no_action": 5,
    "sitting down": 6,
    "standing up": 7,
    "walking": 8,
    "watching tv": 9
}

TARGET_CLASSES = [
    "blowing_nose_or_sneezing",
    "cleaning",
    "eating",
    "falling_down",
    "lying_on_floor",
    "no_action",
    "sitting_down",
    "standing_up",
    "walking",
    "watching_tv"
]

class VideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, frame_size=112, max_samples_per_class=None, augment=False):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.frame_size = frame_size
        self.augment = augment
        self.samples = []
        self.labels = []
        
        
        self.mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        self.std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

        for folder_name, class_idx in CLASS_MAPPING.items():
            class_path = os.path.join(root_dir, folder_name)
            if not os.path.exists(class_path):
                print(f"[Dataset] Warning: Folder {class_path} not found.")
                continue

            video_files = [f for f in os.listdir(class_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            
            if max_samples_per_class is not None:
                video_files = video_files[:max_samples_per_class]

            for vf in video_files:
                self.samples.append((os.path.join(class_path, vf), class_idx))
                self.labels.append(class_idx)

        print(f"[Dataset] Loaded {len(self.samples)} video samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def _augment_frame(self, frame):
        """Apply random augmentations to a single frame."""
        
        if np.random.rand() > 0.5:
            frame = cv2.flip(frame, 1)
        
        
        factor = 1.0 + np.random.uniform(-0.15, 0.15)
        frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
        
        
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return frame

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        if total_frames >= self.clip_len:
            
            frame_indices = set(np.linspace(0, total_frames - 1, self.clip_len, dtype=int))
            for f_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if f_idx in frame_indices:
                    frames.append(frame)
        else:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            if len(frames) == 0:
                
                frames = [np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8) for _ in range(self.clip_len)]
            else:
                while len(frames) < self.clip_len:
                    frames.append(frames[-1].copy()) 
                    
        cap.release()
        
        
        if self.augment and np.random.rand() > 0.7:
            
            if len(frames) > self.clip_len:
                start = np.random.randint(0, max(1, len(frames) - self.clip_len))
                frames = frames[start:start + self.clip_len]
        
        
        processed_frames = []
        for frame in frames[:self.clip_len]:
            resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.frame_size, self.frame_size))
            
            
            if self.augment:
                resized = self._augment_frame(resized)
            
            normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
            processed_frames.append(normalized)

        
        clip_tensor = torch.from_numpy(np.stack(processed_frames)).permute(3, 0, 1, 2)
        return clip_tensor, label


def compute_class_weights(labels, num_classes):
    """Compute inverse-frequency class weights for balanced training."""
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    
    class_counts = np.maximum(class_counts, 1.0)
    
    weights = 1.0 / class_counts
    
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_model(epochs=10, batch_size=4, lr=1e-4, quick=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Training on device: {device}")

    
    train_dir = "IndoorActionDataset-video/train"
    val_dir = "IndoorActionDataset-video/validation"
    
    max_samples = 3 if quick else None
    train_dataset = VideoDataset(train_dir, max_samples_per_class=max_samples, augment=True)
    val_dataset = VideoDataset(val_dir, max_samples_per_class=max_samples, augment=False)
    
    
    
    sample_weights = compute_class_weights(
        np.array(train_dataset.labels), len(TARGET_CLASSES)
    )
    per_sample_weights = [sample_weights[label].item() for label in train_dataset.labels]
    sampler = WeightedRandomSampler(
        weights=per_sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    
    print("[Train] Initializing R3D-18 model structure...")
    model = r3d_18(weights=None)
    
    
    weights_path = "models/weights/r3d_18-b3b3357e.pth"
    if os.path.exists(weights_path):
        print(f"[Train] Loading pretrained Kinetics backbone from {weights_path}...")
        try:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        except Exception as e:
            print(f"[Train] Could not load local state dict: {e}. Starting fresh.")
    
    
    
    for name, param in model.named_parameters():
        if name.startswith(('stem', 'layer1', 'layer2')):
            param.requires_grad = False
    
    
    model.fc = nn.Linear(model.fc.in_features, len(TARGET_CLASSES))
    model = model.to(device)

    
    class_weights = compute_class_weights(
        np.array(train_dataset.labels), len(TARGET_CLASSES)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 8  
    os.makedirs("models/weights", exist_ok=True)
    save_path = "models/weights/indoor_action_r3d18.pt"

    print(f"[Train] Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"[Train] Frozen layers: stem, layer1, layer2 | Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"[Train] Class weights: {dict(zip(TARGET_CLASSES, class_weights.cpu().numpy().round(2)))}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = corrects.double() / total if total > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} LR: {current_lr:.6f}")

        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        class_correct = np.zeros(len(TARGET_CLASSES))
        class_total = np.zeros(len(TARGET_CLASSES))
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                
                
                for i in range(labels.size(0)):
                    label_idx = labels[i].item()
                    class_total[label_idx] += 1
                    if preds[i].item() == label_idx:
                        class_correct[label_idx] += 1

        v_loss = val_loss / val_total if val_total > 0 else 0
        v_acc = val_corrects.double() / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
        
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print("  Per-class accuracy:")
            for ci, cname in enumerate(TARGET_CLASSES):
                ct = class_total[ci]
                ca = class_correct[ci] / ct if ct > 0 else 0.0
                print(f"    {cname}: {ca:.2%} ({int(class_correct[ci])}/{int(ct)})")

        
        scheduler.step()

        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': float(best_val_acc),
                'class_names': TARGET_CLASSES
            }, save_path)
            print(f"[Train] ★ Saved best checkpoint to {save_path} (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[Train] Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                break

    print(f"[Train] Training complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Indoor Action Classifier (R3D-18)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (lower for 4GB VRAM)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick run with minimal dataset samples")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, quick=args.quick)
