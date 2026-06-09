import subprocess
from pathlib import Path

class ClipExtractor:
    def __init__(self, config):
        self.config = config['clips']
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
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out_file)