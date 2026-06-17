import subprocess
import json
import os
import shutil

from utils.debug_log import log_exception, log_subprocess_result, log_subprocess_start


# Project root = the directory above utils/ (resolved from this file, not cwd).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _candidate_dirs():
    """Local ffmpeg build locations, checked before PATH."""
    return [
        os.path.join(_PROJECT_ROOT, 'ffmpeg-gpu'),
        os.path.join(_PROJECT_ROOT, 'ffmpeg-gpu', 'bin'),
        os.path.join(_PROJECT_ROOT, 'ffmpeg', 'bin'),
        os.path.join(_PROJECT_ROOT, 'ffmpeg'),
    ]


def _resolve(tool: str):
    """Locate an ffmpeg-family tool: bundled build → PATH. Returns a path or None."""
    exe = tool + ('.exe' if os.name == 'nt' else '')
    for d in _candidate_dirs():
        p = os.path.join(d, exe)
        if os.path.isfile(p):
            return p
    return shutil.which(tool)


def resolve_ffmpeg() -> str:
    """Path to ffmpeg: bundled build → PATH → imageio-ffmpeg → bare 'ffmpeg'."""
    p = _resolve('ffmpeg')
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def resolve_ffprobe() -> str:
    """Path to ffprobe: bundled build → PATH → bare 'ffprobe' (imageio has no ffprobe)."""
    return _resolve('ffprobe') or 'ffprobe'


def check_audio_track(video_path: str) -> bool:
    """
    Use ffprobe to check if the video file contains an audio stream.
    """
    cmd = [
        resolve_ffprobe(), '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams', video_path
    ]
    try:
        log_subprocess_start(cmd, source="audio")
        result = subprocess.run(cmd, capture_output=True, text=True)
        log_subprocess_result(cmd, result.returncode, result.stdout, result.stderr, source="audio")
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout)
        return any(s.get('codec_type') == 'audio' for s in data.get('streams', []))
    except Exception as e:
        log_exception("FFprobe check failed", e, source="audio")
        print(f"[AudioUtils] FFprobe check failed: {e}")
        return False

def extract_audio(video_path: str, output_wav: str, sample_rate: int = 22050) -> bool:
    """
    Extracts audio track from video and saves it as a 16-bit PCM mono WAV file.
    """
    cmd = [
        resolve_ffmpeg(), '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', '1',
        output_wav
    ]
    try:
        log_subprocess_start(cmd, source="audio")
        result = subprocess.run(cmd, capture_output=True, text=True)
        log_subprocess_result(cmd, result.returncode, result.stdout, result.stderr, source="audio")
        result.check_returncode()
        return os.path.exists(output_wav)
    except Exception as e:
        log_exception("FFmpeg extraction failed", e, source="audio")
        print(f"[AudioUtils] FFmpeg extraction failed: {e}")
        return False
