import torch
import gc
from typing import Callable, Any

class ModelManager:
    def __init__(self, device: str = 'cuda', fp16: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.fp16 = fp16
        self.loaded = {}  # name -> model

    def load_torch_model(self, name: str, loader_fn: Callable, keep_previous: bool = False) -> Any:
        """Load a model. Unload all others unless keep_previous=True."""
        if not keep_previous:
            self._unload_all_except(name)
            
        if name not in self.loaded:
            print(f"[ModelManager] Loading {name} into memory/VRAM (Device: {self.device})...")
            model = loader_fn()
            
            # Send PyTorch model to device
            if hasattr(model, 'to'):
                model = model.to(self.device)
                
            # Cast to half precision (FP16) if running on CUDA and model supports it
            if self.fp16 and self.device.type == 'cuda' and hasattr(model, 'half'):
                try:
                    model = model.half()
                    print(f"[ModelManager] Cast {name} to FP16.")
                except Exception as e:
                    print(f"[ModelManager] Warning: Could not cast {name} to FP16: {e}")
                    
            if hasattr(model, 'eval'):
                model.eval()
                
            self.loaded[name] = model
            
        return self.loaded[name]

    def unload(self, name: str = None):
        """Unload a specific model or all models from VRAM."""
        if name:
            if name in self.loaded:
                print(f"[ModelManager] Unloading {name} from VRAM...")
                del self.loaded[name]
        else:
            print("[ModelManager] Unloading all models...")
            self.loaded.clear()
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[ModelManager] CUDA Cache Cleared.")

    def _unload_all_except(self, keep_name: str):
        """Unload all models except the one specified."""
        to_del = [k for k in self.loaded if k != keep_name]
        if to_del:
            print(f"[ModelManager] Auto-unloading models to free VRAM: {to_del}")
            for k in to_del:
                del self.loaded[k]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_free_vram_mb(self) -> float:
        """Returns the free GPU memory in MB."""
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info()
        return free / 1024 / 1024
