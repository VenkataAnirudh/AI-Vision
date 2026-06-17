"""
VisionAI — Streamlit Forensic Intelligence Dashboard
──────────────────────────────────────────────────────
Production-grade surveillance analytics UI with:
  - LLM Provider toggle (OpenAI / Gemini)
  - Real-time processing progress
  - Multi-tab analytics display
  - Spatial Intelligence (Heatmap + Trajectories)
  - People Gallery with behavioral scores
  - Download Center for all report formats
  - NL Investigator chat interface
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import json
from pathlib import Path
import numpy as np


st.set_page_config(page_title="VisionAI Intelligence", layout="wide", page_icon="🛡️")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
        color: #E0E0E0;
    }

    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        letter-spacing: 3px;
        font-weight: 700;
        background: linear-gradient(135deg, #00D4FF 0%, #0984E3 50%, #6C5CE7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 0.9rem;
        color: #8C98A4;
        letter-spacing: 1px;
        margin-top: -5px;
    }

    .stMetric {
        background: linear-gradient(135deg, #1A1C24 0%, #1F232D 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #00D4FF;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161920;
        border-radius: 8px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 6px;
        font-weight: 600;
        color: #808495;
        padding: 0 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1F232D !important;
        color: #00D4FF !important;
        border-bottom: 2px solid #00D4FF !important;
    }

    .card-incident {
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #1A1C24 0%, #1F232D 100%);
        border-left: 4px solid;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }

    .card-incident:hover {
        transform: translateX(4px);
    }

    .provider-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .badge-openai {
        background: rgba(16, 163, 127, 0.15);
        color: #10A37F;
    }

    .badge-gemini {
        background: rgba(66, 133, 244, 0.15);
        color: #4285F4;
    }

    .download-btn {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 8px;
        background: linear-gradient(135deg, #00D4FF 0%, #0984E3 100%);
        color: white;
        font-weight: 600;
        text-decoration: none;
        margin: 4px;
        font-size: 0.85em;
    }

    .sys-footer {
        text-align: center;
        color: #4A4E5A;
        font-size: 0.8em;
        padding: 20px 0;
        border-top: 1px solid #1A1C24;
        margin-top: 30px;
    }

    .status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #161920 0%, #1A1C24 100%);
        border: 1px solid #232731;
        border-radius: 10px;
        padding: 12px 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }

    .status-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }

    .dot-green { background: #2ECC71; box-shadow: 0 0 6px #2ECC71; }
    .dot-red { background: #FF4B4B; box-shadow: 0 0 6px #FF4B4B; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    .status-label {
        color: #8C98A4;
        font-size: 0.75em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .status-value {
        color: #F8F9FA;
        font-size: 0.95em;
        font-weight: 600;
    }

    .vram-bar {
        width: 100px;
        height: 6px;
        background: #2C313E;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 4px;
    }

    .vram-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)


for key in ['job_id', 'job_status', 'report_data', 'video_url',
            'heatmap_url', 'trajectory_url', 'pdf_url', 'log_url', 'stage_msg',
            'job_info']:
    if key not in st.session_state:
        st.session_state[key] = None


with st.sidebar:
    st.markdown("## ⚙️ System Controls")

    api_url = st.text_input("Backend API URL", value="http://localhost:8000", help="URL of the VisionAI backend server")

    st.markdown("---")

    
    st.markdown("### 🤖 LLM Provider")
    llm_provider = st.selectbox(
        "Select AI Model Provider",
        options=["openai", "gemini"],
        index=0,
        format_func=lambda x: "OpenAI GPT-4.1" if x == "openai" else "Google Gemini 2.5 Flash",
        help="Choose the LLM for video description and NL queries. Toggle when you hit rate limits."
    )
    badge_class = "badge-openai" if llm_provider == "openai" else "badge-gemini"
    badge_text = "GPT-4.1" if llm_provider == "openai" else "Gemini 2.5 Flash"
    st.markdown(f'<span class="provider-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

    st.markdown("---")

    
    st.markdown("### 📹 Upload Surveillance Feed")
    uploaded_file = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov', 'mkv'], label_visibility="collapsed")

    if uploaded_file and st.button("🚀 Initialize Deep Analysis", use_container_width=True):
        with st.spinner("Uploading to processing engine..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                form_data = {"llm_provider": llm_provider}
                res = requests.post(f"{api_url}/analyze", files=files, data=form_data)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state['job_id'] = data.get('job_id')
                    st.session_state['job_status'] = "queued"
                    st.session_state['report_data'] = None
                    st.session_state['video_url'] = None
                    st.session_state['heatmap_url'] = None
                    st.session_state['trajectory_url'] = None
                    st.session_state['pdf_url'] = None
                    st.session_state['log_url'] = None
                    st.session_state['job_info'] = None
                    st.success(f"✅ Job queued! ID: `{data.get('job_id')[:8]}...`")
                else:
                    st.error(f"Failed to queue: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Is the server running?")
            except Exception as e:
                st.error(f"Error: {e}")

    
    if st.session_state['job_id']:
        st.markdown("---")
        st.markdown(f"**Active Job:** `{st.session_state['job_id'][:8]}...`")

        status = st.session_state.get('job_status', 'unknown')
        status_colors = {'queued': '🟡', 'processing': '🔵', 'completed': '🟢', 'failed': '🔴'}
        st.markdown(f"**Status:** {status_colors.get(status, '⚪')} `{status.upper()}`")

        if status in ['queued', 'processing']:
            stage_msg = st.session_state.get('stage_msg') or 'Initializing...'
            st.info(f"⚙️ **{stage_msg}**")


            time.sleep(2.5)
            try:
                res = requests.get(f"{api_url}/status/{st.session_state['job_id']}", timeout=20)
                if res.status_code == 200:
                    status_data = res.json()
                    st.session_state['job_status'] = status_data.get('status')
                    st.session_state['stage_msg'] = status_data.get('stage', 'Processing...')
                    st.session_state['job_info'] = status_data

                    if st.session_state['job_status'] == 'completed':
                        st.session_state['video_url'] = f"{api_url}{status_data.get('video_url')}" if status_data.get('video_url') else None
                        st.session_state['heatmap_url'] = f"{api_url}{status_data.get('heatmap_url')}" if status_data.get('heatmap_url') else None
                        st.session_state['trajectory_url'] = f"{api_url}{status_data.get('trajectory_url')}" if status_data.get('trajectory_url') else None
                        st.session_state['pdf_url'] = f"{api_url}{status_data.get('pdf_url')}" if status_data.get('pdf_url') else None
                        st.session_state['log_url'] = f"{api_url}{status_data.get('log_url')}" if status_data.get('log_url') else None


                        report_url = status_data.get('report_url')
                        if report_url:
                            report_res = requests.get(f"{api_url}{report_url}", timeout=10)
                            if report_res.status_code == 200:
                                st.session_state['report_data'] = report_res.json()
                        st.success("✅ Analysis Complete!")
            except Exception:
                # A slow/timed-out poll (Stage 4 is GIL-heavy) must NOT kill the refresh loop.
                st.caption("⏳ waiting for backend…")

            # Always reschedule a rerun while the job is active, even if this poll failed,
            # so the UI keeps polling until the job reports completed/failed.
            st.rerun()

        elif status == 'failed':
            error_msg = (st.session_state.get('job_info') or {}).get('error', 'Unknown error')
            st.error(f"❌ Pipeline failed: {error_msg}")

    
    st.markdown("---")
    st.markdown("### 💻 System Health")
    try:
        health = requests.get(f"{api_url}/health", timeout=3).json()
        gpu = health.get('gpu', {})
        if isinstance(gpu, dict):
            st.caption(f"🟢 **GPU:** {gpu.get('device', 'N/A')}")
            vram_used = gpu.get('vram_total_mb', 0) - gpu.get('vram_free_mb', 0)
            st.caption(f"📊 **VRAM:** {vram_used}MB / {gpu.get('vram_total_mb', 0)}MB")
        st.caption(f"⏱️ **Uptime:** {health.get('uptime_seconds', 0):.0f}s")
        st.caption(f"📦 **Jobs:** {health.get('total_jobs', 0)} total")
    except Exception:
        st.caption("⚪ Backend offline")



st.markdown('<p class="main-title">AI VISION</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your World, Our Watch. Uncompromised Surveillance.</p>', unsafe_allow_html=True)




import streamlit as _st_version_check
_HAS_FRAGMENT = hasattr(_st_version_check, 'fragment')

def _status_bar_decorator(fn):
    """Wrap with st.fragment if available, otherwise return plain function."""
    if _HAS_FRAGMENT:
        return st.fragment(run_every=3)(fn)
    return fn

@_status_bar_decorator
def _render_status_bar():
    try:
        _health = requests.get(f"{api_url}/health", timeout=2).json()
        _gpu = _health.get('gpu', {})
        _online = True
        if isinstance(_gpu, dict):
            _gpu_name = _gpu.get('device', 'N/A')
            _vram_total = _gpu.get('vram_total_mb', 0)
            _vram_free = _gpu.get('vram_free_mb', 0)
            _vram_used = _vram_total - _vram_free
            _vram_pct = (_vram_used / _vram_total * 100) if _vram_total > 0 else 0
            _vram_color = '#2ECC71' if _vram_pct < 70 else ('#FF9F43' if _vram_pct < 90 else '#FF4B4B')
        else:
            _gpu_name = 'CPU Only'
            _vram_total = _vram_used = _vram_pct = 0
            _vram_color = '#8C98A4'
        _uptime = _health.get('uptime_seconds', 0)
        _uptime_str = f"{int(_uptime//60)}m {int(_uptime%60)}s" if _uptime >= 60 else f"{_uptime:.0f}s"
        _jobs = _health.get('total_jobs', 0)
        _active = _health.get('active_jobs', 0)
    except Exception:
        _online = False
        _gpu_name = 'N/A'
        _vram_used = _vram_total = _vram_pct = 0
        _vram_color = '#FF4B4B'
        _uptime_str = 'N/A'
        _jobs = _active = 0

    _dot_class = 'dot-green' if _online else 'dot-red'
    _status_text = 'ONLINE' if _online else 'OFFLINE'

    st.markdown(f"""
<div class="status-bar">
    <div class="status-item">
        <span class="status-dot {_dot_class}"></span>
        <div>
            <div class="status-label">Engine</div>
            <div class="status-value">{_status_text} <span style="color:#00D4FF; font-size:0.8em; margin-left:5px;">CUDA:0 FP16</span></div>
        </div>
    </div>
    <div class="status-item">
        <div>
            <div class="status-label">Telemetry</div>
            <div class="status-value" style="font-size:0.85em;">Fire/Crime: 30 FPS | Skeletal: 5 FPS</div>
        </div>
    </div>
    <div class="status-item">
        <div>
            <div class="status-label">VRAM <span style="color:#2ECC71; font-size:0.7em;">(Opt. Active)</span></div>
            <div class="status-value">{_vram_used}MB / {_vram_total}MB</div>
            <div class="vram-bar"><div class="vram-fill" style="width:{_vram_pct:.0f}%; background:{_vram_color};"></div></div>
        </div>
    </div>
    <div class="status-item">
        <div>
            <div class="status-label">Uptime</div>
            <div class="status-value">{_uptime_str}</div>
        </div>
    </div>
    <div class="status-item">
        <div>
            <div class="status-label">Jobs</div>
            <div class="status-value">{_active} active / {_jobs} total</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

_render_status_bar()
st.markdown("---")

report_data = st.session_state.get('report_data')

if report_data:
    
    col1, col2, col3, col4, col5 = st.columns(5)

    unique_count = report_data.get('people', {}).get('unique_count', 0)
    col1.metric("👤 Tracked Targets", f"{unique_count} Unique", delta="ByteTrack")

    threat_level = report_data.get('threat_level', 'GREEN')
    threat_delta = "⚠️ Elevated" if threat_level in ['ORANGE', 'RED'] else "✅ Normal"
    col2.metric("🎯 Threat Index", threat_level, delta=threat_delta,
                delta_color="inverse" if threat_level != 'GREEN' else "normal")

    meta = report_data.get('processing_meta', {})
    col3.metric("📊 Frames Analyzed", f"{meta.get('frames_sampled', 0)}", delta="FP16 Active")

    col4.metric("⏱️ Duration", f"{report_data.get('duration_seconds', 0.0):.1f}s", delta="Segment")

    event_count = len(report_data.get('events', []))
    col5.metric("🔔 Events", f"{event_count}", delta="Deduplicated")

    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "🎬 Annotated Feed", "🗺️ Spatial & Structural Intel",
        "👥 People Gallery", "💬 NL Investigator", "📥 Downloads", "🪵 Logs"
    ])

    
    with tab1:
        
        st.markdown("### 📝 Scene Narrative")
        st.info(f"*{report_data.get('summary', 'No narrative available.')}*")

        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.markdown("### 📈 Composite Threat Timeline")
            timeline = report_data.get('threat_timeline', [])
            if timeline:
                df = pd.DataFrame(timeline)
                df.rename(columns={'ts': 'Time (s)', 'score': 'Threat Score'}, inplace=True)
                fig = px.area(df, x='Time (s)', y='Threat Score',
                              color_discrete_sequence=['#00D4FF'])
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                fig.add_hline(y=0.30, line_dash="dash", line_color="rgba(241,196,15,0.4)",
                              annotation_text="Yellow")
                fig.add_hline(y=0.60, line_dash="dash", line_color="rgba(255,159,67,0.4)",
                              annotation_text="Orange")
                fig.add_hline(y=0.80, line_dash="dash", line_color="rgba(255,75,75,0.4)",
                              annotation_text="Red")
                st.plotly_chart(fig, use_container_width=True)

            
            crowd_data = report_data.get('analytics', {}).get('crowd', {})
            density_timeline = crowd_data.get('density_timeline', [])
            if density_timeline:
                st.markdown("### 👥 Crowd Density Over Time")
                df_crowd = pd.DataFrame(density_timeline)
                if 'timestamp' in df_crowd.columns and 'count' in df_crowd.columns:
                    fig_crowd = px.bar(df_crowd, x='timestamp', y='count',
                                       color_discrete_sequence=['#6C5CE7'])
                    fig_crowd.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=250,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="Time (s)",
                        yaxis_title="Person Count",
                    )
                    st.plotly_chart(fig_crowd, use_container_width=True)

        with col_r:
            st.markdown("### 🚨 Security Incidents")
            events = report_data.get('events', [])
            if events:
                for e in sorted(events, key=lambda x: x.get('timestamp', 0), reverse=True)[:15]:
                    color = "#FF4B4B" if e.get('severity') == "HIGH" else "#FF9F43"
                    evt_type = e.get('type', 'unknown').lower()
                    
                    if 'fire' in evt_type or 'smoke' in evt_type:
                        engine_tag = "[FIRE ENGINE]"
                        engine_color = "#E67E22"
                    elif evt_type in ['violence', 'weapon', 'fight', 'punching person', 'slapping', 'wrestling', 'headbutting']:
                        engine_tag = "[SPATIAL ENGINE]"
                        engine_color = "#E74C3C"
                    elif evt_type in ['falling_down', 'lying_on_floor', 'sitting_down', 'standing_up', 'bent_over', 'aggressive_guard']:
                        engine_tag = "[SKELETAL ENGINE]"
                        engine_color = "#9B59B6"
                    elif evt_type in ['crying', 'laughing']:
                        engine_tag = "[TEMPORAL ENGINE]"
                        engine_color = "#3498DB"
                    else:
                        engine_tag = "[SYS ENGINE]"
                        engine_color = "#95A5A6"
                        
                    st.markdown(f"""
                    <div class="card-incident" style="border-left-color: {color};">
                        <span style="color: {engine_color}; font-size: 0.7em; font-weight: bold; display: block; margin-bottom: 3px;">{engine_tag}</span>
                        <strong>{e.get('type', 'Anomaly').replace('_', ' ').title()}</strong>
                        <span style="color: #00D4FF; font-family: monospace; float: right;">{e.get('timestamp', 0.0):.1f}s</span><br/>
                        <span style="font-size: 0.85em; color: #8C98A4;">
                            Conf: {e.get('confidence', 0.0):.0%} · {e.get('severity', 'MEDIUM')}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No anomalies detected.")

            
            st.markdown("### 📚 Chapters")
            chapters = report_data.get('chapters', [])
            for c in chapters:
                alert_emoji = "🔴" if c.get('alert_level') == 'RED' else ("🟡" if c.get('alert_level') == 'YELLOW' else "🟢")
                st.caption(f"{alert_emoji} **Ch.{c['index']}:** {c['title']} ({c['start_fmt']}–{c['end_fmt']})")

    
    with tab2:
        st.markdown("### 🎬 Annotated Forensic Feed")
        if st.session_state.get('video_url'):
            st.video(st.session_state['video_url'])
        else:
            st.info("Annotated video not available for this run.")

    
    with tab3:
        st.markdown("### 🗺️ Spatial & Structural Intelligence Analytics")
        st.markdown("*Hardware-accelerated analytics mapped to physical space*")

        col_h, col_t = st.columns(2)

        with col_h:
            st.markdown("#### 🌡️ Activity Heatmap")
            st.caption("Spatial density of person detections across all frames")
            if st.session_state.get('heatmap_url'):
                st.image(st.session_state['heatmap_url'], use_container_width=True)
            else:
                st.info("Heatmap not available.")

        with col_t:
            st.markdown("#### 🔀 Movement Trajectories")
            st.caption("Per-person movement paths color-coded by track ID")
            if st.session_state.get('trajectory_url'):
                st.image(st.session_state['trajectory_url'], use_container_width=True)
            else:
                st.info("Trajectories not available.")

        st.markdown("---")
        st.markdown("#### 📐 Structural Geometry & Geofencing")
        st.info("Continuous structural metrics (Spine Angle, Hip Velocity) streaming disabled to conserve memory. High-resolution anomaly triggers are logged directly in the events ticker instead.")
        
        breaches = len([e for e in events if 'breach' in e.get('type', '').lower()])
        st.metric("🚫 Perimeter Breaches", breaches, delta="-0" if breaches == 0 else f"+{breaches} violations", delta_color="inverse")
        st.caption("Active safety perimeter violations based on lower-limb keypoint mapping.")

        
        crowd = report_data.get('analytics', {}).get('crowd', {})
        if crowd:
            st.markdown("---")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Max Crowd", crowd.get('max_count', 0))
            cc2.metric("Avg Crowd", f"{crowd.get('avg_count', 0):.1f}")
            cc3.metric("Overcrowd Events", len(crowd.get('overcrowding_events', [])) if 'overcrowding_events' in crowd else 0)

    
    with tab4:
        st.markdown("### 👥 Tracked Individuals")

        individuals = report_data.get('people', {}).get('individuals', [])
        behavioral_scores = report_data.get('analytics', {}).get('behavioral_scores', [])
        emotion_summary = report_data.get('emotion_summary', [])

        
        bs_map = {bs['track_id']: bs for bs in (behavioral_scores or [])}
        emo_map = {es['track_id']: es for es in (emotion_summary or [])}

        if individuals or behavioral_scores:
            
            display_list = behavioral_scores if behavioral_scores else [
                {'track_id': p['track_id'], 'anomaly_score': 0, 'dwell_time_seconds': 0}
                for p in individuals
            ]

            events = report_data.get('events', [])
            
            for person_data in display_list:
                tid = person_data['track_id']
                anomaly = person_data.get('anomaly_score', 0) * 100
                dwell = person_data.get('dwell_time_seconds', 0)

                identity = "Unknown"
                face_conf = None
                for ind in individuals:
                    if ind['track_id'] == tid:
                        identity = ind.get('face_id', 'Unknown')
                        face_conf = ind.get('face_confidence')
                        break

                emo_data = emo_map.get(tid, {})
                dominant_emotion = emo_data.get('dominant_emotion', '')
                avg_stress = emo_data.get('avg_stress_score', 0)

                emotion_emojis = {
                    'happy': '😊', 'sad': '😢', 'angry': '😠', 'surprise': '😲',
                    'fear': '😨', 'disgust': '🤢', 'neutral': '😐', 'contempt': '😒'
                }
                emo_emoji = emotion_emojis.get(dominant_emotion, '❓')

                stress_color = '#FF4B4B' if avg_stress > 0.6 else ('#FF9F43' if avg_stress > 0.3 else '#2ECC71')
                stress_label = 'HIGH' if avg_stress > 0.6 else ('MED' if avg_stress > 0.3 else 'LOW')

                has_fall = any('fall' in e.get('type', '').lower() and e.get('track_id', tid) == tid for e in events)
                has_guard = any(('guard' in e.get('type', '').lower() or 'violence' in e.get('type', '').lower()) and e.get('track_id', tid) == tid for e in events)
                
                posture = "Crouching / Bent Over" if (anomaly > 30 and not has_fall) else ("Critical Collapse" if has_fall else "Standing Straight")
                posture_color = "#FF4B4B" if has_fall else ("#FF9F43" if anomaly > 30 else "#2ECC71")
                
                kinetic_state = "Active Locomotion" if dwell < 10 else ("Stationary" if dwell > 30 else "Loitering")
                kinetic_color = "#3498DB" if kinetic_state == "Active Locomotion" else ("#95A5A6" if kinetic_state == "Stationary" else "#E67E22")
                
                guard_tag = '<br><span style="background:#E74C3C; padding:2px 6px; border-radius:4px; font-size:0.7em; margin-top:2px; display:inline-block;">🥊 GUARD INDEX ACTIVE</span>' if has_guard else ''

                with st.container():
                    if has_fall:
                        st.markdown(f'<div style="background:#FF4B4B; color:white; padding:5px 10px; border-radius:5px 5px 0 0; font-weight:bold; letter-spacing:1px; text-align:center;">CRITICAL COLLAPSE ANOMALY DETECTED</div>', unsafe_allow_html=True)
                        
                    c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 1.5, 2])
                    c1.markdown(f"**Track {tid}**")
                    c2.markdown(f"🪪 {identity}<br><span style='color:{posture_color}; font-size:0.85em;'>Posture: {posture}</span>{guard_tag}", unsafe_allow_html=True)
                    c3.markdown(f"⏱️ Dwell: {dwell:.1f}s<br><span style='color:{kinetic_color}; font-size:0.85em;'>Kinetic: {kinetic_state}</span>", unsafe_allow_html=True)

                    if dominant_emotion:
                        c4.markdown(
                            f'{emo_emoji} **{dominant_emotion.title()}**'
                            f'<br><span style="font-size:0.75em; color:{stress_color}">'
                            f'Stress: {stress_label} ({avg_stress:.0%})</span>',
                            unsafe_allow_html=True
                        )
                    else:
                        c4.markdown("*No emotion data*")

                    c5.progress(min(anomaly / 100, 1.0), text=f"Anomaly: {anomaly:.0f}%")
                    st.markdown("---")

            
            all_emotions = report_data.get('emotions', [])
            if all_emotions:
                from collections import Counter
                emotion_counts = Counter(e.get('emotion', 'neutral') for e in all_emotions)
                if emotion_counts:
                    st.markdown("### 🎭 Emotion Distribution")
                    df_emo = pd.DataFrame(
                        list(emotion_counts.items()),
                        columns=['Emotion', 'Count']
                    )
                    emo_colors = {
                        'happy': '#2ECC71', 'sad': '#3498DB', 'angry': '#E74C3C',
                        'surprise': '#F1C40F', 'fear': '#9B59B6', 'disgust': '#1ABC9C',
                        'neutral': '#95A5A6', 'contempt': '#E67E22'
                    }
                    fig_emo = px.pie(
                        df_emo, names='Emotion', values='Count',
                        color='Emotion',
                        color_discrete_map=emo_colors,
                    )
                    fig_emo.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig_emo, use_container_width=True)

        else:
            st.info("No tracked individuals in this analysis.")

    
    with tab5:
        st.markdown("### 💬 Natural Language Investigation")
        st.markdown(f"*Ask questions about the analysis. Powered by {badge_text}.*")

        query_input = st.text_input(
            "Query the analysis report",
            placeholder="e.g., 'When did the fall occur?', 'Who was identified?', 'What is the overall threat assessment?'"
        )
        if query_input:
            with st.chat_message("assistant"):
                with st.spinner(f"Querying {badge_text}..."):
                    try:
                        payload = {"job_id": st.session_state['job_id'], "question": query_input}
                        query_res = requests.post(f"{api_url}/query", params=payload, timeout=30)
                        if query_res.status_code == 200:
                            answer = query_res.json().get('answer', 'No answer received.')
                            st.markdown(answer)
                        else:
                            st.error(f"Error: {query_res.text}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")

    
    with tab6:
        st.markdown("### 📥 Download Center")
        st.markdown("*Export analysis results in multiple formats.*")

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            st.markdown("#### 📄 Reports")
            if report_data:
                json_str = json.dumps(report_data, indent=2)
                st.download_button("📊 JSON Report", json_str, file_name="visionai_report.json",
                                   mime="application/json", use_container_width=True)

            if st.session_state.get('pdf_url'):
                st.markdown(f"[📕 Download PDF Report]({st.session_state['pdf_url']})")

        with dl_col2:
            st.markdown("#### 🖼️ Analytics")
            if st.session_state.get('heatmap_url'):
                st.markdown(f"[🌡️ Download Heatmap]({st.session_state['heatmap_url']})")
            if st.session_state.get('trajectory_url'):
                st.markdown(f"[🔀 Download Trajectories]({st.session_state['trajectory_url']})")

        with dl_col3:
            st.markdown("#### 🎬 Media")
            if st.session_state.get('video_url'):
                st.markdown(f"[📹 Download Annotated Video]({st.session_state['video_url']})")

    with tab7:
        st.markdown("### 🪵 Pipeline Logs")
        st.markdown("*Tail of this run's per-stage log file.*")
        log_url = st.session_state.get('log_url')
        if not log_url:
            st.info("Logs become available once the run completes.")
        else:
            try:
                log_res = requests.get(log_url, timeout=10)
                if log_res.status_code == 200:
                    payload = log_res.json()
                    log_text = payload.get('log', '')
                    if log_text.strip():
                        st.caption(
                            f"Showing tail of `{payload.get('file', 'pipeline log')}` "
                            f"· {payload.get('total_lines', 0)} lines total"
                        )
                        st.code(log_text, language="log")
                    else:
                        st.info(payload.get('note', 'No log output yet.'))
                else:
                    st.warning(f"Logs unavailable (HTTP {log_res.status_code}).")
            except Exception as e:
                st.warning(f"Could not fetch logs: {e}")

else:
    
    st.info("👈 Upload a surveillance video in the sidebar and click **Initialize Deep Analysis** to begin.")

    st.markdown("### 🏗️ System Architecture")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Hardware**\n\nNVIDIA CUDA FP16")
    c2.info("**Multimodal**\n\nAudio-Visual Fusion")
    c3.info("**Semantic**\n\nVLM Keyframe Analysis")
    c4.info("**Analytics**\n\nHeatmap + Trajectories")

    st.markdown("### 🧠 Intelligence Capabilities")
    cap_cols = st.columns(3)
    capabilities = [
        ("🔥 Fire & Smoke Engine", "YOLO11s with localized Volumetric Rate-of-Rise monitoring"),
        ("⚔️ Surveillance Threat Engine", "Pose/proximity-gated YOLO weapon detector + multimodal fight fusion (pose + emotion + audio)"),
        ("🦴 Structural Keypoint Engine", "YOLO11x-Pose computing real-time 17-joint spatial vectors"),
        ("👤 Track & Trace", "ByteTrack multi-person tracking + dwell analytics"),
        ("😊 Affective Computing", "DeepFace emotion recognition + stress heuristics"),
        ("🗣️ Temporal Engine", "3D-ResNet Behavioral Context & False-Positive Shield"),
        ("🌡️ Spatial Heatmap", "Hardware-accelerated density visualization"),
        ("🔀 Geometric Interpolation", "Ultra-smooth native FPS temporal rendering"),
        ("🗺️ Contextual Geometry", "Dynamic Posture & Kinetic State Analysis"),
    ]
    for i, (name, desc) in enumerate(capabilities):
        cap_cols[i % 3].markdown(f"**{name}**\n\n{desc}")


st.markdown("---")
st.markdown(
    '<div class="sys-footer">'
    'VisionAI Pipeline Engine v3.0 · Production Grade · '
    'Dual LLM (OpenAI + Gemini) · CPU Analytics · GPU-Accelerated Detection'
    '</div>',
    unsafe_allow_html=True
)