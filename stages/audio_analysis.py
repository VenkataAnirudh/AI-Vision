import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from utils.audio_utils import extract_audio


class AudioCNN(nn.Module):
    """Compact 2D CNN over log-mel — verbatim from the 05_audio_events notebook."""
    def __init__(self, n_classes):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.net(x))


class AudioAnalyzer:
    def __init__(self, config, model_manager=None):
        self.config = config['models']['audio']
        self.manager = model_manager
        self.output_dir = Path(config['output']['base_dir']) / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # AudioCNN metadata, populated lazily on first model load.
        self.class_names = None
        self.alert_class_indices = []
        self.cnn_sr = 16000
        self.cnn_duration = 4.0
        self.cnn_n_fft = 1024
        self.cnn_hop = 512
        self.cnn_n_mels = 64

    def _get_model(self):
        """Lazily build + load the fine-tuned AudioCNN via ModelManager."""
        def loader():
            ckpt = torch.load(self.config['model_path'], map_location='cpu', weights_only=False)
            cfg = ckpt.get('config', {})
            self.class_names = ckpt['class_names']
            # Scream is dropped from detection: it was trained without a real scream dataset, so
            # its predictions are unreliable. Only baby_cry is treated as an alert class here.
            self.alert_class_indices = [i for i, c in enumerate(self.class_names) if c == 'baby_cry']
            self.cnn_sr = int(cfg.get('sr', 16000))
            self.cnn_duration = float(cfg.get('duration', 4.0))
            self.cnn_n_fft = int(cfg.get('n_fft', 1024))
            self.cnn_hop = int(cfg.get('hop', 512))
            self.cnn_n_mels = int(cfg.get('n_mels', 64))
            model = AudioCNN(len(self.class_names))
            model.load_state_dict(ckpt['model_state_dict'])
            return model
        return self.manager.load_torch_model('audio_cnn', loader, keep_previous=True, skip_fp16=True)

    def _logmel(self, window):
        """Per-clip log-mel z-score — matches the notebook front-end exactly."""
        n_samples = int(self.cnn_sr * self.cnn_duration)
        if len(window) < n_samples:
            window = np.pad(window, (0, n_samples - len(window)))
        else:
            window = window[:n_samples]
        m = librosa.feature.melspectrogram(y=window, sr=self.cnn_sr, n_fft=self.cnn_n_fft,
                                           hop_length=self.cnn_hop, n_mels=self.cnn_n_mels)
        m = librosa.power_to_db(m, ref=np.max)
        m = (m - m.mean()) / (m.std() + 1e-6)
        return m.astype(np.float32)

    def _run_cnn(self, wav_path):
        """Classify 4s windows with the AudioCNN. Returns a list of baby_cry segments."""
        cry_segs = []
        if self.manager is None or not self.config.get('model_path'):
            return cry_segs
        try:
            model = self._get_model()
            y, _ = librosa.load(wav_path, sr=self.cnn_sr, mono=True)
        except Exception as e:
            print(f"[AudioAnalyzer] CNN pass skipped: {e}")
            return cry_segs

        win = int(self.config.get('cnn_window_seconds', 4.0) * self.cnn_sr)
        hop = int(self.config.get('cnn_hop_seconds', 2.0) * self.cnn_sr)
        thresh = float(self.config.get('cnn_confidence', 0.5))
        if len(y) < win:
            y = np.pad(y, (0, win - len(y)))

        # Stride windows, then always include an end-anchored window so a cry in the
        # final seconds of a short clip isn't silently dropped (the stride alone can skip the tail).
        starts = list(range(0, len(y) - win + 1, hop))
        tail = len(y) - win
        if tail >= 0 and tail not in starts:
            starts.append(tail)

        for i in starts:
            window = y[i:i + win]
            start_ts = i / self.cnn_sr
            end_ts = (i + win) / self.cnn_sr
            feat = self._logmel(window)
            x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(self.manager.device).float()
            with torch.no_grad():
                probs = torch.softmax(model(x).float(), dim=-1)[0].cpu().numpy()
            cls_idx = int(probs.argmax())
            if cls_idx not in self.alert_class_indices or float(probs[cls_idx]) < thresh:
                continue
            cry_segs.append({'start_ts': start_ts, 'end_ts': end_ts, 'timestamp': start_ts,
                             'probability': float(probs[cls_idx])})
        return cry_segs

    def process(self, video_path: str):
        """
        Analyzes the audio of a video to detect cries and loud events.
        """
        temp_wav = str(self.output_dir / f"temp_{Path(video_path).stem}.wav")
        
        
        if not extract_audio(video_path, temp_wav, self.config['sample_rate']):
            print("[AudioAnalyzer] No audio stream extracted or ffmpeg failed.")
            return {
                'has_audio': False,
                'cry_segments': [],
                'raised_voice_segments': [],
                'loud_events': []
            }

        try:
            y, sr = librosa.load(temp_wav, sr=self.config['sample_rate'])
        except Exception as e:
            print(f"[AudioAnalyzer] Error loading WAV file: {e}")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            return {
                'has_audio': False,
                'cry_segments': [],
                'raised_voice_segments': [],
                'loud_events': []
            }

        
        window_samples = int(self.config['window_seconds'] * sr)
        hop_samples = int(self.config['hop_seconds'] * sr)
        
        cries = []
        raised_voices = []
        loud_events = []

        
        rms_history = []

        for i in range(0, len(y) - window_samples, hop_samples):
            window = y[i:i + window_samples]
            start_ts = i / sr
            end_ts = (i + window_samples) / sr
            
            
            rms = float(librosa.feature.rms(y=window)[0].mean())
            rms_history.append(rms)

            
            pitches, mags = librosa.piptrack(y=window, sr=sr)
            if mags.max() > 0:
                freq_idx, time_idx = np.unravel_index(mags.argmax(), mags.shape)
                dominant_pitch = float(pitches[freq_idx, time_idx])
            else:
                dominant_pitch = 0.0
                
            
            mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13).mean(axis=1)
            zcr = float(librosa.feature.zero_crossing_rate(window)[0].mean())
            rolloff = float(librosa.feature.spectral_rolloff(y=window, sr=sr)[0].mean())

            
            if len(rms_history) >= 10:
                baseline = np.mean(rms_history[-30:])
                sigma = np.std(rms_history[-30:]) + 1e-6
                z_rms = (rms - baseline) / sigma
            else:
                z_rms = 0.0 

            
            is_cry = (
                self.config['cry_freq_min_hz'] <= dominant_pitch <= self.config['cry_freq_max_hz']
                and z_rms > 2.5
                and rolloff < 4000  
                and zcr < 0.1       
            )
            if is_cry:
                cries.append({
                    'start_ts': start_ts,
                    'end_ts': end_ts,
                    'timestamp': start_ts,  
                    'probability': 0.85
                })

            
            if z_rms > 3.0:
                raised_voices.append({
                    'start_ts': start_ts,
                    'end_ts': end_ts,
                    'timestamp': start_ts
                })
                    
            
            if rms > self.config['loud_event_rms_threshold']:
                loud_events.append({
                    'timestamp': start_ts,
                    'start_ts': start_ts,
                    'end_ts': end_ts,
                    'severity': 'HIGH'
                })

        # AudioCNN pass (runs alongside the librosa heuristics above; needs the wav at 16 kHz).
        cnn_cry_segments = self._run_cnn(temp_wav)


        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception as e:
                print(f"[AudioAnalyzer] Warning: could not delete temporary audio file: {e}")

        return {
            'has_audio': True,
            'cry_segments': cries,
            'raised_voice_segments': raised_voices,
            'loud_events': loud_events,
            'cnn_cry_segments': cnn_cry_segments,
        }