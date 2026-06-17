"""
VisionAI — Report Generator
────────────────────────────
Generates JSON, HTML, and SRT reports for video analysis results.
Supports dynamic per-run output directories.
"""

import json
import os
from pathlib import Path
from datetime import datetime


class ReportGenerator:
    def __init__(self, config):
        self.config = config

    def _build_emotion_summary(self, emotions, individuals):
        """Aggregate per-person emotion data for the report."""
        if not emotions:
            return []

        
        from collections import Counter, defaultdict
        track_emotions = defaultdict(list)
        track_stress = defaultdict(list)

        for emo in emotions:
            tid = emo.get('track_id')
            if tid is not None:
                track_emotions[tid].append(emo.get('emotion', 'neutral'))
                track_stress[tid].append(emo.get('visual_stress_score', 0.0))

        
        identity_map = {ind['track_id']: ind.get('face_id', 'Unknown') for ind in (individuals or [])}

        summary = []
        for tid in track_emotions:
            emotion_counts = Counter(track_emotions[tid])
            dominant = emotion_counts.most_common(1)[0][0] if emotion_counts else 'neutral'
            avg_stress = sum(track_stress[tid]) / max(1, len(track_stress[tid]))
            summary.append({
                'track_id': tid,
                'identity': identity_map.get(tid, 'Unknown'),
                'dominant_emotion': dominant,
                'emotion_distribution': dict(emotion_counts),
                'avg_stress_score': round(avg_stress, 3),
                'readings_count': len(track_emotions[tid]),
            })
        return summary

    def _format_time_srt(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def build_srt(self, video_stem, events, run_dir=None):
        """Generates a standard SubRip (.srt) file for video players like VLC."""
        if run_dir:
            out_dir = Path(run_dir) / "reports"
        else:
            out_dir = Path(self.config['output']['base_dir']) / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        srt_path = out_dir / f"events_{video_stem}.srt"

        sorted_events = sorted(events, key=lambda x: x.get('timestamp', 0))

        with open(srt_path, 'w', encoding='utf-8') as f:
            for idx, evt in enumerate(sorted_events):
                ts = evt.get('timestamp', 0.0)
                start = max(0.0, ts - 1.5)
                end = ts + 3.5

                f.write(f"{idx + 1}\n")
                f.write(f"{self._format_time_srt(start)} --> {self._format_time_srt(end)}\n")
                f.write(f"ALERT: {evt['type'].upper()} ({evt.get('severity', 'MEDIUM')} - Conf: {evt.get('confidence', 0.0):.2f})\n\n")

        print(f"[Reporter] SRT Subtitles saved to {srt_path}")
        return str(srt_path)

    def build_html(self, video_name, duration, report_json_path, run_dir=None):
        """Generates a premium, browser-viewable HTML report."""
        with open(report_json_path, 'r') as f:
            data = json.load(f)

        if run_dir:
            out_dir = Path(run_dir) / "reports"
        else:
            out_dir = Path(self.config['output']['base_dir']) / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        html_path = out_dir / f"report_{Path(video_name).stem}.html"

        
        events_html = ""
        for e in data.get('events', []):
            sev_color = "#FF4B4B" if e.get('severity') == "HIGH" else "#FF9F43"
            events_html += f"""
            <div class="card" style="border-left: 4px solid {sev_color};">
                <div class="row" style="justify-content: space-between;">
                    <div>
                        <strong style="color: #F8F9FA;">{e.get('type', 'Unknown').upper()}</strong>
                        <span style="font-size: 0.85em; color: #8C98A4; margin-left: 10px;">({e.get('severity', 'MEDIUM')} alert)</span>
                    </div>
                    <span style="color: #00D4FF; font-family: monospace; font-size: 1.1em;">{e.get('timestamp', 0.0):.1f}s</span>
                </div>
                <div class="row" style="margin-top: 5px; font-size: 0.9em; color: #ADB5BD;">
                    Confidence: {e.get('confidence', 0.0):.2%}
                </div>
            </div>
            """

        
        people_html = ""
        individuals = data.get('people', {}).get('individuals', [])
        for p in individuals:
            people_html += f"""
            <div class="card" style="border-left: 4px solid #00D4FF;">
                <div class="row" style="justify-content: space-between;">
                    <strong style="color: #FFF;">Track ID: {p.get('track_id')}</strong>
                    <span class="badge" style="background: rgba(0, 212, 255, 0.15); color: #00D4FF;">Identity: {p.get('face_id', 'Unknown')}</span>
                </div>
                <div style="font-size: 0.85em; color: #8C98A4; margin-top: 5px;">
                    Face confidence: {f"{p.get('face_confidence'):.2%}" if p.get('face_confidence') else 'N/A'}
                </div>
            </div>
            """

        
        chapters_html = ""
        for c in data.get('chapters', []):
            badge_bg = "rgba(255, 75, 75, 0.2)" if c.get('alert_level') == "RED" else "rgba(46, 204, 113, 0.2)"
            badge_fg = "#FF4B4B" if c.get('alert_level') == "RED" else "#2ECC71"
            chapters_html += f"""
            <div class="card">
                <div class="row" style="justify-content: space-between;">
                    <strong>Chapter {c.get('index')}: {c.get('title')}</strong>
                    <span class="badge" style="background: {badge_bg}; color: {badge_fg};">{c.get('alert_level')}</span>
                </div>
                <div style="font-size: 0.85em; color: #8C98A4; margin-top: 5px;">
                    Timeline: {c.get('start_fmt')} - {c.get('end_fmt')} ({c.get('event_count')} events)
                </div>
            </div>
            """

        
        behavioral_html = ""
        behavioral_scores = data.get('analytics', {}).get('behavioral_scores', [])
        for bs in behavioral_scores:
            anomaly_pct = bs.get('anomaly_score', 0) * 100
            bar_color = "#FF4B4B" if anomaly_pct > 60 else ("#FF9F43" if anomaly_pct > 30 else "#2ECC71")
            behavioral_html += f"""
            <div class="card">
                <div class="row" style="justify-content: space-between;">
                    <strong>Track ID {bs.get('track_id')}</strong>
                    <span style="color: {bar_color}; font-weight: 600;">Anomaly: {anomaly_pct:.0f}%</span>
                </div>
                <div style="margin-top: 8px; background: #2C313E; border-radius: 4px; height: 6px;">
                    <div style="width: {anomaly_pct:.0f}%; background: {bar_color}; height: 100%; border-radius: 4px;"></div>
                </div>
                <div style="font-size: 0.8em; color: #8C98A4; margin-top: 5px;">
                    Movement: {bs.get('movement_score', 0):.2f} | Emotion: {bs.get('emotion_score', 0):.2f} | Events: {bs.get('event_score', 0):.2f} | Dwell: {bs.get('dwell_time_seconds', 0):.1f}s
                </div>
            </div>
            """

        
        timeline = data.get('threat_timeline', [])
        svg_points = ""
        if timeline:
            max_val = max([t.get('score', 0) for t in timeline] + [1.0])
            width = 800
            height = 150
            padding = 10
            x_step = (width - 2 * padding) / max(1, len(timeline) - 1)
            y_scale = (height - 2 * padding) / max_val

            points_list = []
            for i, t in enumerate(timeline):
                x = padding + i * x_step
                y = height - padding - (t.get('score', 0.0) * y_scale)
                points_list.append(f"{x},{y}")

            svg_points = " ".join(points_list)

        
        llm_provider = data.get('llm_provider', 'openai').upper()

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>VisionAI Security Analysis - {video_name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@500;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #0E1117;
            color: #E0E0E0;
            margin: 0;
            padding: 40px;
        }}
        h1, h2, h3, header {{
            font-family: 'Outfit', sans-serif;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #1A1C24;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .title-logo {{
            font-size: 2.2em;
            font-weight: 700;
            background: linear-gradient(135deg, #00D4FF 0%, #0984E3 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        @media(max-width: 900px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .panel {{
            background: #161920;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            border: 1px solid #232731;
        }}
        .card {{
            background: #1F232D;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            border: 1px solid #2C313E;
            transition: transform 0.2s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
        }}
        .row {{
            display: flex;
            align-items: center;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .metric-box {{
            background: #1F232D;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border-left: 5px solid #00D4FF;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: 700;
            color: #00D4FF;
            margin-top: 5px;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        svg {{
            background: #1F232D;
            border-radius: 8px;
            border: 1px solid #2C313E;
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <div class="title-logo">VISIONAI FORENSICS</div>
            <div style="color: #8C98A4; margin-top: 5px;">Surveillance Feed Analysis Report</div>
        </div>
        <div style="text-align: right;">
            <div><strong>Processed At:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            <div style="color: #8C98A4; margin-top: 5px;">LLM Provider: {llm_provider} | Fusion: {data.get('fusion_mode', 'visual_only')}</div>
        </div>
    </header>

    <div class="grid" style="grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
        <div class="metric-box">
            <div style="color: #8C98A4; font-size: 0.9em;">Asset Tracked</div>
            <div class="metric-value" style="font-size: 1.2em;">{Path(video_name).name}</div>
        </div>
        <div class="metric-box" style="border-left-color: #2ECC71;">
            <div style="color: #8C98A4; font-size: 0.9em;">Duration</div>
            <div class="metric-value">{duration:.1f}s</div>
        </div>
        <div class="metric-box" style="border-left-color: #E74C3C;">
            <div style="color: #8C98A4; font-size: 0.9em;">Threat Level</div>
            <div class="metric-value" style="color: #E74C3C;">{data.get('threat_level', 'GREEN')}</div>
        </div>
        <div class="metric-box" style="border-left-color: #F1C40F;">
            <div style="color: #8C98A4; font-size: 0.9em;">Anomalous Events</div>
            <div class="metric-value" style="color: #F1C40F;">{len(data.get('events', []))}</div>
        </div>
    </div>

    <div class="panel" style="margin-bottom: 30px;">
        <h3 style="margin-top: 0; color: #FFF; border-bottom: 1px solid #232731; padding-bottom: 10px;">Natural Language Scene Narrative</h3>
        <p style="font-size: 1.1em; line-height: 1.6; color: #ECEFF1; font-style: italic;">
            "{data.get('summary', 'No summary available.')}"
        </p>
    </div>

    <div class="panel" style="margin-bottom: 30px;">
        <h3 style="margin-top: 0; color: #FFF; border-bottom: 1px solid #232731; padding-bottom: 10px;">Threat Score Timeline</h3>
        <div class="chart-container">
            <svg width="100%" height="150" viewBox="0 0 800 150" preserveAspectRatio="none">
                <polyline fill="none" stroke="#00D4FF" stroke-width="3" points="{svg_points}" />
                <line x1="0" y1="75" x2="800" y2="75" stroke="rgba(255,255,255,0.05)" stroke-dasharray="5,5" />
            </svg>
            <div class="row" style="justify-content: space-between; font-size: 0.8em; color: #8C98A4; margin-top: 5px; padding: 0 10px;">
                <span>0.0s (Start)</span>
                <span>Threat Score Trend</span>
                <span>{duration:.1f}s (End)</span>
            </div>
        </div>
    </div>

    <div class="grid">
        <div class="panel">
            <h3 style="margin-top: 0; color: #FFF; border-bottom: 1px solid #232731; padding-bottom: 10px;">Detected Incidents</h3>
            {events_html if events_html else '<p style="color: #8C98A4;">No high-priority security incidents detected.</p>'}
        </div>

        <div class="panel">
            <h3 style="margin-top: 0; color: #FFF; border-bottom: 1px solid #232731; padding-bottom: 10px;">Identified Targets</h3>
            {people_html if people_html else '<p style="color: #8C98A4;">No individuals tracked in this sequence.</p>'}
        </div>
    </div>

    <div class="grid" style="margin-top: 30px;">
        <div class="panel">
            <h3 style="margin-top: 0; color: #FFF; border-bottom: 1px solid #232731; padding-bottom: 10px;">Behavioral Anomaly Scores</h3>
            {behavioral_html if behavioral_html else '<p style="color: #8C98A4;">No behavioral data available.</p>'}
        </div>

        <div class="panel">
            <h3 style="margin-top: 0; color: #FFF; border-bottom: 1px solid #232731; padding-bottom: 10px;">Chronological Chapters</h3>
            {chapters_html if chapters_html else '<p style="color: #8C98A4;">No chapters generated.</p>'}
        </div>
    </div>
</body>
</html>
"""

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[Reporter] HTML Dashboard saved to {html_path}")
        return str(html_path)

    def build_json(self, video_name, duration, aggregated_results, run_dir=None):
        """Generates the main structured JSON report file."""
        if run_dir:
            out_dir = Path(run_dir) / "reports"
        else:
            out_dir = Path(self.config['output']['base_dir']) / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        
        threat_timeline = aggregated_results.get('threat_timeline', [])
        max_threat = "GREEN"
        if threat_timeline:
            scores = [t['score'] for t in threat_timeline]
            max_score = max(scores) if scores else 0.0
            if max_score >= 0.80:
                max_threat = "RED"
            elif max_score >= 0.60:
                max_threat = "ORANGE"
            elif max_score >= 0.30:
                max_threat = "YELLOW"

        report = {
            "video": video_name,
            "duration_seconds": duration,
            "processed_at": datetime.now().isoformat(),
            "fusion_mode": aggregated_results.get('fusion_mode', 'visual_only'),
            "llm_provider": aggregated_results.get('llm_provider', 'openai'),
            "threat_level": max_threat,
            "summary": aggregated_results.get('description', "No VLM description generated."),
            "people": {
                "unique_count": aggregated_results.get('unique_count', 0),
                "individuals": aggregated_results.get('individuals', []),
            },
            "events": aggregated_results.get('events', []),
            "emotions": aggregated_results.get('emotions', []),
            "emotion_summary": self._build_emotion_summary(aggregated_results.get('emotions', []),
                                                            aggregated_results.get('individuals', [])),
            "chapters": aggregated_results.get('chapters', []),
            "threat_timeline": threat_timeline,
            "analytics": aggregated_results.get('analytics', {}),
            "processing_meta": {
                "total_frames_in_video": aggregated_results.get('total_frames', 0),
                "frames_sampled": aggregated_results.get('frames_sampled', 0),
                "fp16_enabled": self.config['hardware']['fp16'],
            },
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"report_{timestamp}.json"

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"[Reporter] JSON Report saved to {out_path}")
        return str(out_path)