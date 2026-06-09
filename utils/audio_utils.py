import subprocess
import json
import os

def check_audio_track(video_path: str) -> bool:
    """
    Use ffprobe to check if the video file contains an audio stream.
    """
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams', video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout)
        return any(s.get('codec_type') == 'audio' for s in data.get('streams', []))
    except Exception as e:
        print(f"[AudioUtils] FFprobe check failed: {e}")
        return False

def extract_audio(video_path: str, output_wav: str, sample_rate: int = 22050) -> bool:
    """
    Extracts audio track from video and saves it as a 16-bit PCM mono WAV file.
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', '1',
        output_wav
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return os.path.exists(output_wav)
    except Exception as e:
        print(f"[AudioUtils] FFmpeg extraction failed: {e}")
        return False
