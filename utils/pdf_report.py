"""
VisionAI — Professional PDF Report Generator
──────────────────────────────────────────────
Generates a multi-page PDF report with executive summary,
metrics, event tables, and embedded analytics images.
Uses fpdf2 for lightweight PDF generation (no heavy deps).
"""

import json
from pathlib import Path
from datetime import datetime

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False


# fpdf2's core fonts (Helvetica) are latin-1 only; common unicode punctuation (em/en dashes,
# curly quotes, ellipsis) raises/garbles otherwise. Map the frequent offenders to ASCII, then
# drop anything still outside latin-1 so VLM-generated narratives never crash the report.
_UNICODE_MAP = {
    '—': '-', '–': '-', '‒': '-', '−': '-',
    '‘': "'", '’': "'", '‚': "'", '‛': "'",
    '“': '"', '”': '"', '„': '"',
    '…': '...', '•': '*', ' ': ' ',
}


def _safe(text) -> str:
    """Sanitize a string for fpdf2 core (latin-1) fonts."""
    s = str(text)
    for u, a in _UNICODE_MAP.items():
        s = s.replace(u, a)
    return s.encode('latin-1', 'replace').decode('latin-1')


class PDFReportGenerator:
    """Generates a branded PDF security analysis report."""

    def __init__(self):
        if not HAS_FPDF:
            raise ImportError("fpdf2 is required for PDF reports. Install with: pip install fpdf2")

    def generate(self, json_report_path: str, run_dir, heatmap_path: str = None, trajectory_path: str = None) -> str:
        """
        Generate a PDF report from a JSON analysis report.

        Args:
            json_report_path: Path to the JSON report file.
            run_dir: Path to the run directory for saving the PDF.
            heatmap_path: Optional path to the heatmap image.
            trajectory_path: Optional path to the trajectory image.

        Returns:
            str: Path to the generated PDF file.
        """
        with open(json_report_path, 'r') as f:
            data = json.load(f)

        run_dir = Path(run_dir)
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        video_name = data.get('video', 'Unknown')
        pdf_path = reports_dir / f"report_{Path(video_name).stem}.pdf"

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        
        pdf.add_page()
        self._draw_header(pdf, "VISIONAI FORENSICS REPORT")

        
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(200, 200, 200)
        pdf.cell(0, 8, _safe(f"Video Asset: {video_name}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, _safe(f"Duration: {data.get('duration_seconds', 0):.1f} seconds"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, _safe(f"Processed: {data.get('processed_at', 'N/A')}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, _safe(f"Fusion Mode: {data.get('fusion_mode', 'visual_only')}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, _safe(f"LLM Provider: {data.get('llm_provider', 'openai').upper()}"), new_x="LMARGIN", new_y="NEXT")

        
        threat_level = data.get('threat_level', 'GREEN')
        threat_colors = {'GREEN': (46, 204, 113), 'YELLOW': (241, 196, 15), 'ORANGE': (255, 159, 67), 'RED': (255, 75, 75)}
        color = threat_colors.get(threat_level, (200, 200, 200))
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(*color)
        pdf.cell(0, 12, f"OVERALL THREAT LEVEL: {threat_level}", new_x="LMARGIN", new_y="NEXT")

        
        pdf.ln(5)
        pdf.set_text_color(200, 200, 200)
        pdf.set_font("Helvetica", "", 10)
        people_data = data.get('people', {})
        meta = data.get('processing_meta', {})
        pdf.cell(0, 7, f"Unique Persons Tracked: {people_data.get('unique_count', 0)}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 7, f"Total Events Detected: {len(data.get('events', []))}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 7, f"Frames Sampled: {meta.get('frames_sampled', 0)} / {meta.get('total_frames_in_video', 0)}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 7, f"FP16 Pipeline: {'Enabled' if meta.get('fp16_enabled') else 'Disabled'}", new_x="LMARGIN", new_y="NEXT")

        
        pdf.ln(8)
        self._section_title(pdf, "Scene Narrative")
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(180, 180, 180)
        summary = data.get('summary', 'No summary available.')
        pdf.multi_cell(0, 6, _safe(summary))

        
        events = data.get('events', [])
        if events:
            pdf.add_page()
            self._draw_header(pdf, "DETECTED INCIDENTS")

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(200, 200, 200)
            pdf.set_fill_color(30, 35, 45)

            
            col_widths = [55, 30, 30, 30, 45]
            headers = ["Event Type", "Time (s)", "Confidence", "Severity", "Details"]
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 8, h, border=1, fill=True)
            pdf.ln()

            
            pdf.set_font("Helvetica", "", 8)
            for evt in sorted(events, key=lambda x: x.get('timestamp', 0)):
                pdf.set_text_color(180, 180, 180)
                pdf.cell(col_widths[0], 7, _safe(evt.get('type', 'N/A')[:25]), border=1)
                pdf.cell(col_widths[1], 7, f"{evt.get('timestamp', 0):.1f}", border=1)
                pdf.cell(col_widths[2], 7, f"{evt.get('confidence', 0):.2%}", border=1)

                sev = evt.get('severity', 'MEDIUM')
                if sev == 'HIGH':
                    pdf.set_text_color(255, 75, 75)
                elif sev == 'MEDIUM':
                    pdf.set_text_color(255, 159, 67)
                else:
                    pdf.set_text_color(180, 180, 180)
                pdf.cell(col_widths[3], 7, sev, border=1)

                pdf.set_text_color(180, 180, 180)
                details = str(evt.get('details', ''))[:20]
                pdf.cell(col_widths[4], 7, _safe(details), border=1)
                pdf.ln()

        
        individuals = people_data.get('individuals', [])
        behavioral = data.get('analytics', {}).get('behavioral_scores', [])

        if individuals or behavioral:
            pdf.add_page()
            self._draw_header(pdf, "PERSONNEL ANALYSIS")

            if individuals:
                self._section_title(pdf, "Identified Targets")
                pdf.set_font("Helvetica", "", 9)
                for p in individuals:
                    pdf.set_text_color(200, 200, 200)
                    face_conf = f"{p.get('face_confidence'):.2%}" if p.get('face_confidence') else 'N/A'
                    pdf.cell(0, 7, _safe(f"Track ID {p.get('track_id')} - Identity: {p.get('face_id', 'Unknown')} (Confidence: {face_conf})"), new_x="LMARGIN", new_y="NEXT")

            if behavioral:
                pdf.ln(5)
                self._section_title(pdf, "Behavioral Anomaly Scores")
                pdf.set_font("Helvetica", "", 9)
                for bs in behavioral:
                    anomaly = bs.get('anomaly_score', 0) * 100
                    pdf.set_text_color(200, 200, 200)
                    pdf.cell(0, 7,
                        _safe(f"Track {bs.get('track_id')} - Anomaly: {anomaly:.0f}% | "
                              f"Movement: {bs.get('movement_score', 0):.2f} | "
                              f"Emotion: {bs.get('emotion_score', 0):.2f} | "
                              f"Dwell: {bs.get('dwell_time_seconds', 0):.1f}s"),
                        new_x="LMARGIN", new_y="NEXT"
                    )

        
        has_images = (heatmap_path and Path(heatmap_path).exists()) or (trajectory_path and Path(trajectory_path).exists())
        if has_images:
            pdf.add_page()
            self._draw_header(pdf, "SPATIAL INTELLIGENCE")

            if heatmap_path and Path(heatmap_path).exists():
                self._section_title(pdf, "Activity Heatmap")
                try:
                    pdf.image(heatmap_path, x=10, w=190)
                except Exception:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.cell(0, 7, "[Heatmap image could not be embedded]", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(5)

            if trajectory_path and Path(trajectory_path).exists():
                self._section_title(pdf, "Movement Trajectories")
                try:
                    pdf.image(trajectory_path, x=10, w=190)
                except Exception:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.cell(0, 7, "[Trajectory image could not be embedded]", new_x="LMARGIN", new_y="NEXT")

        
        chapters = data.get('chapters', [])
        if chapters:
            pdf.add_page()
            self._draw_header(pdf, "VIDEO CHAPTERS")
            pdf.set_font("Helvetica", "", 10)
            for c in chapters:
                pdf.set_text_color(200, 200, 200)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 8, _safe(f"Chapter {c.get('index')}: {c.get('title')}"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(150, 150, 150)
                pdf.cell(0, 6, _safe(f"  {c.get('start_fmt')} - {c.get('end_fmt')} | Alert: {c.get('alert_level')} | Events: {c.get('event_count')}"), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)

        
        pdf.output(str(pdf_path))
        return str(pdf_path)

    

    def _draw_header(self, pdf, title):
        """Draw a branded page header."""
        pdf.set_fill_color(14, 17, 23)
        pdf.rect(0, 0, 210, 297, 'F')  

        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(0, 212, 255)
        pdf.cell(0, 15, title, new_x="LMARGIN", new_y="NEXT")

        pdf.set_draw_color(0, 212, 255)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)

    def _section_title(self, pdf, title):
        """Draw a section title."""
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 212, 255)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(200, 200, 200)
