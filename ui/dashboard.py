import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
import json
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="VisionAI Intelligence", layout="wide", page_icon="🛡️")

# --- CUSTOM CSS (Aesthetic Branding) ---
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
        color: #00D4FF;
        font-size: 3rem;
        letter-spacing: 2px;
        margin-bottom: 0px;
        font-weight: 700;
        background: linear-gradient(135deg, #00D4FF 0%, #0984E3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stMetric {
        background-color: #1A1C24;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00D4FF;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        font-weight: 600;
        color: #808495;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1A1C24 !important;
        color: #00D4FF !important;
        border-bottom: 2px solid #00D4FF !important;
    }
    
    .card-incident {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        background-color: #1A1C24;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'job_id' not in st.session_state:
    st.session_state['job_id'] = None
if 'job_status' not in st.session_state:
    st.session_state['job_status'] = None
if 'report_data' not in st.session_state:
    st.session_state['report_data'] = None
if 'video_url' not in st.session_state:
    st.session_state['video_url'] = None

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## System Controls")
    api_url = st.text_input("Backend API URL", value="http://localhost:8000")
    uploaded_file = st.file_uploader("Upload Surveillance Feed", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file and st.button("Initialize Deep Analysis"):
        with st.spinner("Uploading to GPU Cluster..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = requests.post(f"{api_url}/analyze", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state['job_id'] = data.get('job_id')
                    st.session_state['job_status'] = "queued"
                    st.session_state['report_data'] = None
                    st.session_state['video_url'] = None
                    st.success(f"Job queued successfully! ID: {st.session_state['job_id']}")
                else:
                    st.error(f"Failed to queue job: {res.text}")
            except Exception as e:
                st.error(f"Error communicating with backend: {e}")

    # Check status if a job is active
    if st.session_state['job_id']:
        st.markdown("---")
        st.markdown(f"**Active Job ID:** `{st.session_state['job_id']}`")
        st.markdown(f"**Current Status:** `{st.session_state['job_status'].upper()}`")
        
        if st.session_state['job_status'] in ['queued', 'processing']:
            stage_msg = st.session_state.get('stage_msg', 'Initializing...')
            st.info(f"⚙️ **{stage_msg}**")
            
            # Auto-poll status
            time.sleep(2.0)
            try:
                res = requests.get(f"{api_url}/status/{st.session_state['job_id']}")
                if res.status_code == 200:
                    status_data = res.json()
                    st.session_state['job_status'] = status_data.get('status')
                    st.session_state['stage_msg'] = status_data.get('stage', 'Processing...')
                    if st.session_state['job_status'] == 'completed':
                        st.session_state['video_url'] = f"{api_url}{status_data.get('video_url')}"
                        report_res = requests.get(f"{api_url}{status_data.get('report_url')}")
                        if report_res.status_code == 200:
                            st.session_state['report_data'] = report_res.json()
                        st.success("Analysis Completed!")
                    st.rerun()
                else:
                    st.error("Failed to query status.")
            except Exception as e:
                st.error(f"Status check error: {e}")
        elif st.session_state['job_status'] == 'failed':
            st.error("Pipeline analysis failed.")

# --- MAIN DASHBOARD ---
st.markdown('<p class="main-title">SECURITY INTELLIGENCE HUB</p>', unsafe_allow_html=True)
st.markdown("---")

report_data = st.session_state['report_data']

if report_data:
    # --- METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    
    unique_count = report_data.get('people', {}).get('unique_count', 0)
    col1.metric("Tracked Targets", f"{unique_count} Unique", delta="Person Bounding Box Tracker")
    
    threat_level = report_data.get('threat_level', 'GREEN')
    threat_delta = "Elevated Alert" if threat_level in ['ORANGE', 'RED'] else "Normal Activity"
    col2.metric("Threat Index", threat_level, delta=threat_delta, delta_color="inverse" if threat_level != 'GREEN' else "normal")
    
    meta = report_data.get('processing_meta', {})
    sampled = meta.get('frames_sampled', 0)
    col3.metric("Processed Samples", f"{sampled} frames", delta="FP16 Pipeline Active")
    
    col4.metric("Duration", f"{report_data.get('duration_seconds', 0.0):.1f} seconds", delta="Surveillance Feed Segment")

    # --- MAIN CONTENT TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎬 Annotated Feed", "✂️ Event Clips", "💬 LLM Investigator"])

    with tab1:
        col_l, col_r = st.columns([2, 1])
        
        with col_l:
            st.markdown("### Composite Threat Timeline")
            # Build chart from real threat score timeline
            timeline = report_data.get('threat_timeline', [])
            if timeline:
                df = pd.DataFrame(timeline)
                df.rename(columns={'ts': 'Time (s)', 'score': 'Threat Score'}, inplace=True)
                fig = px.area(df, x='Time (s)', y='Threat Score', color_discrete_sequence=['#00D4FF'])
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Threat score timeline unavailable.")
            
        with col_r:
            st.markdown("### Log of Security Incidents")
            events = report_data.get('events', [])
            if events:
                # Display in reverse chronological order
                for e in sorted(events, key=lambda x: x.get('timestamp', 0), reverse=True):
                    color = "#FF4B4B" if e.get('severity') == "HIGH" else "#FF9F43"
                    st.markdown(f"""
                    <div class="card-incident" style="border-left-color: {color};">
                        <strong>{e.get('type', 'Anomaly').upper()}</strong> (Confidence: {e.get('confidence', 0.0):.1%})<br/>
                        <span style="color: #00D4FF; font-family: monospace;">{e.get('timestamp', 0.0):.1f}s</span> - Severity: {e.get('severity', 'MEDIUM')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No anomalies detected in the current segment.")

    with tab2:
        st.markdown("### High-Fidelity Forensic Feed (Annotated)")
        if st.session_state['video_url']:
            st.video(st.session_state['video_url'])
        else:
            st.info("Annotated video URL is not available.")

    with tab3:
        st.markdown("### Chronological Chapters")
        chapters = report_data.get('chapters', [])
        if chapters:
            for c in chapters:
                st.markdown(f"**Chapter {c['index']}: {c['title']}** ({c['start_fmt']} - {c['end_fmt']})")
                st.caption(f"Alert level: {c['alert_level']} | Events: {c['event_count']}")
                st.markdown("---")
        else:
            st.info("No chapter segments generated.")

    with tab4:
        st.markdown("### Natural Language Investigation")
        st.markdown("*Ask questions directly related to the events, timeline, or entities detected in this video.*")
        
        query_input = st.text_input("Query the analysis report (e.g., 'When did the fall occur?', 'Who was identified?')")
        if query_input:
            with st.chat_message("assistant"):
                st.write("Querying backend VLM narrative engine...")
                try:
                    payload = {"job_id": st.session_state['job_id'], "question": query_input}
                    query_res = requests.post(f"{api_url}/query", params=payload)
                    if query_res.status_code == 200:
                        st.write(query_res.json().get('answer'))
                    else:
                        st.error(f"Error querying backend: {query_res.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
else:
    # Prompt user to select/initialize a job
    st.info("👈 Please upload a surveillance video feed in the sidebar and click 'Initialize Deep Analysis' to begin.")
    
    # Showcase system specs placeholder
    st.markdown("### System Architecture")
    c1, c2, c3 = st.columns(3)
    c1.info("**Hardware Acceleration:** NVIDIA CUDA Compute 7.5 Enabled")
    c2.info("**Multimodal Processing:** Joint Audio-Visual Signal Analysis")
    c3.info("**Semantic Analysis:** VLM Keyframe Synthesis & Querying")

st.markdown("---")
st.caption("VisionAI Pipeline Engine v2.5 | NVIDIA GTX 1650 Ti (4GB VRAM)")