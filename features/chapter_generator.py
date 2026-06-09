import json
from pathlib import Path

class ChapterGenerator:
    def __init__(self, config):
        self.min_segment = config['chapters']['min_segment_seconds']
        self.output_dir = Path(config['output']['base_dir']) / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, video_duration, events, action_detections):
        chapters = []
        # Dividing a short clip into thirds for logical structure
        boundaries = [0.0, video_duration / 3, (video_duration / 3) * 2, video_duration]
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            
            # Filter events happening in this specific time block
            segment_events = [e for e in events if start <= e.get('timestamp', 0) < end]
            segment_actions = [a for a in action_detections if start <= a.get('timestamp', 0) < end]
            
            alert_level = "GREEN"
            if any(e.get('severity') == 'HIGH' for e in segment_events):
                alert_level = "RED"
            elif segment_events:
                alert_level = "YELLOW"
                
            # Dynamic titling based on what the models saw
            if segment_events:
                title = f"Activity Detected: {segment_events[0]['type'].title()}"
            elif segment_actions:
                title = f"Routine: {segment_actions[0]['action'].replace('_', ' ').title()}"
            else:
                title = "Calm / No Significant Activity"

            chapters.append({
                "index": i + 1,
                "start_ts": round(start, 1),
                "end_ts": round(end, 1),
                "start_fmt": self._fmt_ts(start),
                "end_fmt": self._fmt_ts(end),
                "title": title,
                "alert_level": alert_level,
                "event_count": len(segment_events)
            })
            
        return chapters

    def _fmt_ts(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}:{s:02d}"