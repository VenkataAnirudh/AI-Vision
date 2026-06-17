"""
VisionAI — Event Clip Extractor
────────────────────────────────
Extracts short video clips around detected incidents using ffmpeg.
Supports dynamic per-run output directories.
"""

import subprocess
from pathlib import Path

from utils.debug_log import log_exception, log_subprocess_result, log_subprocess_start


class ClipExtractor:
    def __init__(self, config, run_dir=None):
        self.config = config['clips']
        if run_dir:
            self.output_dir = Path(run_dir) / "events"
        else:
            self.output_dir = Path(config['output']['base_dir']) / "events"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, video_path, event_type, timestamp):
        if not self.config['enabled']:
            return None

        pre = self.config['pre_event_seconds']
        post = self.config['post_event_seconds']

        start = max(0, timestamp - pre)
        duration = pre + post

        # Clean filename characters
        safe_type = event_type.replace(' ', '_').replace('/', '_')
        out_file = self.output_dir / f"{safe_type}_{timestamp:.1f}s.mp4"

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', video_path,
            '-t', str(duration),
            '-c:v', 'libx264', '-crf', '23',
            '-c:a', 'aac',
            str(out_file)
        ]

        try:
            log_subprocess_start(cmd, source="clip-extractor")
            result = subprocess.run(cmd, capture_output=True, text=True)
            log_subprocess_result(
                cmd,
                result.returncode,
                result.stdout,
                result.stderr,
                source="clip-extractor",
            )
            return str(out_file) if result.returncode == 0 and out_file.exists() else None
        except Exception as e:
            log_exception("Clip extraction failed", e, source="clip-extractor")
            return None
