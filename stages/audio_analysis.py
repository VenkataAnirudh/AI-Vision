import os
import librosa
import numpy as np
from pathlib import Path
from utils.audio_utils import extract_audio

class AudioAnalyzer:
    def __init__(self, config):
        self.config = config['models']['audio']
        self.output_dir = Path(config['output']['base_dir']) / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, video_path: str):
        """
        Analyzes the audio of a video to detect cries and loud events.
        """
        temp_wav = str(self.output_dir / f"temp_{Path(video_path).stem}.wav")
        
        # Extract audio using ffmpeg helper
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

        # Sliding window analysis parameters
        window_samples = int(self.config['window_seconds'] * sr)
        hop_samples = int(self.config['hop_seconds'] * sr)
        
        cries = []
        raised_voices = []
        loud_events = []

        # Simple rolling smoothing list for stress calculation
        rms_history = []

        for i in range(0, len(y) - window_samples, hop_samples):
            window = y[i:i + window_samples]
            start_ts = i / sr
            end_ts = (i + window_samples) / sr
            
            # Root Mean Square energy (loudness)
            rms = float(librosa.feature.rms(y=window)[0].mean())
            rms_history.append(rms)

            # Pitch tracking
            pitches, mags = librosa.piptrack(y=window, sr=sr)
            if mags.max() > 0:
                dominant_pitch = float(pitches[mags.argmax()])
            else:
                dominant_pitch = 0.0
                
            # Heuristic 1: Cry detection (pitch range + volume)
            is_cry = (
                self.config['cry_freq_min_hz'] <= dominant_pitch <= self.config['cry_freq_max_hz']
                and rms > self.config['cry_rms_threshold']
            )
            if is_cry:
                cries.append({
                    'start_ts': start_ts,
                    'end_ts': end_ts,
                    'timestamp': start_ts,  # For overlap compatibility
                    'probability': 0.85
                })

            # Heuristic 2: Raised Voice (extended high volume)
            if rms > self.config['raised_voice_threshold']:
                raised_voices.append({
                    'start_ts': start_ts,
                    'end_ts': end_ts,
                    'timestamp': start_ts
                })
                    
            # Heuristic 3: Loud Event (sudden impact/screaming)
            if rms > self.config['loud_event_rms_threshold']:
                loud_events.append({
                    'timestamp': start_ts,
                    'start_ts': start_ts,
                    'end_ts': end_ts,
                    'severity': 'HIGH'
                })

        # Cleanup temporary audio file
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception as e:
                print(f"[AudioAnalyzer] Warning: could not delete temporary audio file: {e}")
            
        return {
            'has_audio': True,
            'cry_segments': cries,
            'raised_voice_segments': raised_voices,
            'loud_events': loud_events
        }