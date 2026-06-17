"""
VisionAI — Core Video Intelligence Pipeline
─────────────────────────────────────────────
Orchestrates all analysis stages:
  Stage 0: Adaptive Frame Sampling
  Stage 1: Person Detection & Tracking (YOLOv8n + ByteTrack)
  Stage 2: Face Identification (InsightFace + FAISS)
  Stage 3: Emotion Analysis (DeepFace + EAR heuristics)
  Stage 4: Event Detection (Fire/Smoke, Violence, Indoor Action)
  Stage 5: Audio Analysis (Librosa — conditional)
  Stage 6: Audio-Visual Fusion
  Stage 7: Video Narrative Description (OpenAI GPT-4.1 / Gemini 2.5 Flash)
  Analytics: Heatmap, Trajectories, Loitering, Crowd, Behavioral Scoring
"""

import os
import re
import cv2
import yaml
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime


from core.model_manager import ModelManager
from core.frame_sampler import FrameSampler
from stages.person_detection import PersonDetector
from stages.face_pipeline import FacePipeline
from stages.emotion_analysis import EmotionAnalyzer
from stages.event_detection import EventDetector
from stages.weapon_detection import WeaponDetector
from stages.violence_detection import ViolenceClassifier, ViolenceSTGCN
from stages.indoor_action import IndoorActionDetector, IndoorActionSTGCN
from stages.dual_head_model import DualHeadR3D18
from stages.behavioral_context import BehavioralContextEngine
from stages.audio_analysis import AudioAnalyzer
from stages.video_description import VideoDescriber


from fusion.av_fusion import AudioVisualFuser
from features.proximity_analyzer import ProximityAnalyzer
from features.anonymizer import FaceAnonymizer
from features.threat_scorer import ThreatScorer
from features.chapter_generator import ChapterGenerator
from features.clip_extractor import ClipExtractor
from features.heatmap_generator import HeatmapGenerator
from features.trajectory_visualizer import TrajectoryVisualizer
from features.loitering_detector import LoiteringDetector
from features.crowd_analyzer import CrowdAnalyzer
from features.behavioral_scorer import BehavioralScorer
from utils.reporter import ReportGenerator
from utils.metrics import TemporalSmoother
from utils.audio_utils import check_audio_track, resolve_ffmpeg
from utils.drawing import Annotator, ui_scale
from utils.logger import get_logger, attach_run_logfile


class VideoPipeline:
    def __init__(self, stages=None, llm_provider="openai"):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        
        self.log = get_logger(level=self.config.get('logging', {}).get('level', 'INFO'))

        
        device_name = self.config['hardware']['device']
        fp16_enabled = self.config['hardware']['fp16']
        self.manager = ModelManager(device=device_name, fp16=fp16_enabled)

        
        self.llm_provider = llm_provider

        
        self.frame_sampler = FrameSampler(self.config)
        self.smoother = TemporalSmoother(window_size=5)
        self.proximity_analyzer = ProximityAnalyzer()
        self.anonymizer = FaceAnonymizer()
        self.threat_scorer = ThreatScorer(self.config)
        self.chapter_generator = ChapterGenerator(self.config)
        self.annotator = Annotator()

        
        self.heatmap_gen = HeatmapGenerator(self.config)
        self.trajectory_viz = TrajectoryVisualizer(self.config)
        self.loitering_detector = LoiteringDetector(self.config)
        self.crowd_analyzer = CrowdAnalyzer(self.config)
        self.behavioral_scorer = BehavioralScorer(self.config)

        self.stages_to_run = stages if stages else ['all']

    def _should_run(self, stage_name):
        if 'all' in self.stages_to_run:
            return self.config['stages'].get(stage_name, True)
        return stage_name in self.stages_to_run

    def _create_run_directory(self, video_path):
        """Create a unique output directory for this analysis run."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_stem = Path(video_path).stem
        safe_stem = re.sub(r'[^a-zA-Z0-9_\-]', '_', video_stem)
        base_dir = Path(self.config['output']['base_dir'])
        run_dir = base_dir / f"run_{run_id}_{safe_stem}"

        for subdir in ["reports", "annotated", "analytics", "events", "keyframes", "logs"]:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Attach a per-run log file (the singleton logger is created at import time with no
        # run dir, so we add the handler here once the directory exists).
        try:
            level = self.config.get('logging', {}).get('level', 'INFO')
            self._run_log_path = attach_run_logfile(str(run_dir / "logs"), level=level)
        except Exception as e:
            self._run_log_path = None
            self.log.warning(f"Per-run log file could not be attached: {e}")

        self.log.info(f"Run directory created: {run_dir}")
        return run_dir

    def _transcode_to_h264(self, raw_path, final_path):
        """Transcode an mp4v file to browser-playable H.264 via the resolved ffmpeg.

        Used only when the OpenCV build cannot open an 'avc1' writer directly. On success
        the raw mp4v is replaced by the H.264 file; if ffmpeg is missing or fails, the raw
        file is kept as the output so the run still produces a playable (if mp4v) video.
        """
        import subprocess
        raw_path = Path(raw_path)
        final_path = Path(final_path)
        ffmpeg = resolve_ffmpeg()
        cmd = [ffmpeg, '-y', '-i', str(raw_path),
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(final_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and final_path.exists():
                try:
                    raw_path.unlink()
                except Exception:
                    pass
                self.log.info(f"Transcoded annotated video to H.264 via {ffmpeg}.")
            else:
                # Keep the raw mp4v as the final output.
                raw_path.replace(final_path)
                self.log.warning(f"H.264 transcode failed (rc={result.returncode}); kept mp4v output.")
        except Exception as e:
            try:
                raw_path.replace(final_path)
            except Exception:
                pass
            self.log.warning(f"H.264 transcode error: {e}; kept mp4v output.")

    def _deduplicate_events(self, events_list, time_window=2.0, spatial_threshold=200):
        """
        Merges duplicate events of the same type within a time window.
        Now includes a spatial check: events must be physically close to merge.
        """
        if not events_list:
            return []

        sorted_events = sorted(events_list, key=lambda e: (e.get('type', ''), e.get('timestamp', 0)))
        deduped = []

        for event in sorted_events:
            merged = False
            for existing in deduped:
                if (existing['type'] == event['type'] and
                    abs(existing['timestamp'] - event['timestamp']) <= time_window):
                    
                    
                    
                    box_e = event.get('bbox')
                    box_ex = existing.get('bbox')
                    
                    if box_e and box_ex:
                        
                        cx_e, cy_e = (box_e[0] + box_e[2]) / 2, (box_e[1] + box_e[3]) / 2
                        cx_ex, cy_ex = (box_ex[0] + box_ex[2]) / 2, (box_ex[1] + box_ex[3]) / 2
                        distance = ((cx_e - cx_ex)**2 + (cy_e - cy_ex)**2)**0.5
                        
                        if distance > spatial_threshold:
                            continue 
                            
                    if event.get('confidence', 0) > existing.get('confidence', 0):
                        existing.update(event)
                    merged = True
                    break
            if not merged:
                deduped.append(event.copy())

        return deduped

    def _validate_fire_detection(self, fire_events, frame, persons_in_frame, faces_in_frame, recent_persons):
        """
        Multi-signal scoring validator for fire/smoke detections.

        Design rationale
        ────────────────
        Previous implementation used hard binary gates (HSV threshold, person-overlap
        threshold, confidence re-check).  Each gate was independently reasonable but
        their combination was over-fitted to bright, uncompressed, person-free footage.
        Real fire footage is compressed (crushing V-channel headroom), often contains
        persons near the fire, and includes dim/candle flames with low absolute luminance.

        Replacement strategy: every signal contributes a numeric penalty/bonus to a
        composite score.  No single signal can veto a detection on its own.  Only
        detections whose composite score falls below SCORE_THRESHOLD are rejected.

        Scoring bands (0.0 = maximum penalty, 1.0 = maximum support):
          model_conf_score   — raw YOLO confidence (already pre-filtered by config)
          hsv_fire_score     — how fire-like the crop is; soft, not binary
          hsv_smoke_score    — how smoke-like the crop is; soft
          person_penalty     — penalises overlap with persons but never hard-rejects
          area_score         — penalises implausibly tiny or face-sized detections

        Final score = weighted sum.  Gate = 0.35 (intentionally permissive;
        temporal consistency in the next stage is the real precision gate).
        """
        validated = []
        frame_h, frame_w = frame.shape[:2]

        
        
        
        min_area = max(1, int(frame_h * frame_w * 0.001))

        SCORE_THRESHOLD = 0.35   
        W_CONF   = 0.50          
        W_HSV    = 0.50          

        for fe in fire_events:
            bbox = fe.get('bbox', [0, 0, 0, 0])
            fx1, fy1, fx2, fy2 = bbox
            det_w  = max(1, fx2 - fx1)
            det_h  = max(1, fy2 - fy1)
            det_area = det_w * det_h

            
            if det_area < min_area:
                self.log.debug(f"[FireFilter] Skipped: area {det_area} < min {min_area}")
                continue

            
            
            
            
            conf_threshold = self.config['models']['fire'].get('confidence', 0.45)
            raw_conf = fe.get('confidence', 0.0)
            conf_score = min(1.0, (raw_conf - conf_threshold) / max(0.01, 1.0 - conf_threshold))
            conf_score = max(0.0, conf_score)

            
            cx1 = int(max(0, fx1))
            cy1 = int(max(0, fy1))
            cx2 = int(min(frame_w, fx2))
            cy2 = int(min(frame_h, fy2))
            crop = frame[cy1:cy2, cx1:cx2]

            hsv_score = 0.0
            v_mean = 0.0
            v_90th = 0.0
            s_median = 0.0
            hue_fire_ratio = 0.0

            if crop.size > 0 and crop.shape[0] > 2 and crop.shape[1] > 2:
                hsv_crop   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                h_channel  = hsv_crop[:, :, 0].astype(np.float32)
                s_channel  = hsv_crop[:, :, 1].astype(np.float32)
                v_channel  = hsv_crop[:, :, 2].astype(np.float32)

                v_mean    = float(np.mean(v_channel))
                v_90th    = float(np.percentile(v_channel, 90))
                s_median  = float(np.median(s_channel))
                s_mean    = float(np.mean(s_channel))

                
                fire_hue_mask = ((h_channel <= 30) | (h_channel >= 160))
                hue_fire_ratio = float(np.mean(fire_hue_mask))

                
                
                
                
                
                
                fire_v_score   = min(1.0, max(0.0, (v_90th - 100.0) / 100.0))   
                fire_s_score   = min(1.0, max(0.0, (s_mean - 20.0)  / 80.0))    
                fire_hue_score = min(1.0, max(0.0, (hue_fire_ratio - 0.15) / 0.45))  
                fire_composite = (fire_v_score * 0.35 + fire_s_score * 0.30 + fire_hue_score * 0.35)

                
                
                smoke_s_score = min(1.0, max(0.0, (60.0 - s_median) / 60.0))    
                smoke_v_score = min(1.0, max(0.0, 1.0 - abs(v_mean - 140.0) / 100.0))  
                smoke_composite = (smoke_s_score * 0.55 + smoke_v_score * 0.45)

                
                hsv_score = max(fire_composite, smoke_composite)
            else:
                
                hsv_score = 0.40

            fe['hsv_metrics'] = {
                'mean_v': v_mean, 'v_90th': v_90th, 's_median': s_median,
                'hue_fire_ratio': hue_fire_ratio, 'hsv_score': hsv_score,
            }

            
            
            
            composite = (
                W_CONF * conf_score +
                W_HSV  * hsv_score
            )

            self.log.debug(
                f"[FireFilter] conf={raw_conf:.2f} conf_score={conf_score:.2f} "
                f"hsv={hsv_score:.2f} "
                f"→ composite={composite:.2f} (gate={SCORE_THRESHOLD})"
            )

            if composite < SCORE_THRESHOLD:
                self.log.debug(f"[FireFilter] Rejected composite={composite:.2f}")
                continue

            fe['fire_score'] = composite
            validated.append(fe)

        return validated

    def _check_fire_temporal_consistency(self, fire_events, timestamp):
        """
        Confirms a fire detection only after it appears in N frames within a
        time window, with spatial continuity between consecutive frames.

        Design rationale
        ────────────────
        Previous implementation computed required_frames = max(3, int(fps*1.5)).
        At 5 FPS that yields 7–8 required frames inside a 2-second window that
        holds at most 10 frames — meaning any single missed detection (HSV flicker,
        motion blur) would reset the counter and fire would never confirm.

        Fixes applied:
          • required_frames comes directly from config (default 3), never FPS-scaled.
            The config value is the tunable knob; the FPS multiplier was a hidden trap.
          • Spatial continuity uses a max-distance-between-centres check instead of
            requiring a non-zero pixel intersection.  Fire bboxes grow/shrink frame to
            frame; requiring exact overlap resets on any size change.  We allow centres
            to drift up to half the diagonal of the smaller box per frame step.
          • Luminance variance threshold lowered from 5.0 → 0.8.  A candle flame in
            a steady environment has variance ~1–3; the old 5.0 threshold rejected it.
            0.8 only catches true static objects (lights, reflections, LEDs).
          • Cooldown reduced from 5 s → 2 s.  5 s is longer than many short video
            clips and causes zero events to be reported in brief footage.
          • On confirmation the tracker is NOT fully cleared — it keeps all frames
            inside the window so the next second also confirms quickly if fire persists.
        """
        if not hasattr(self, '_fire_frame_tracker'):
            self._fire_frame_tracker = []

        
        required_frames = max(2, self.config['models']['fire'].get('temporal_frames_required', 3))
        window_seconds  = self.config['models']['fire'].get('temporal_window_seconds', 4.0)

        
        if fire_events:
            self._fire_frame_tracker.append({
                'timestamp': timestamp,
                'events': fire_events,
            })

        
        self._fire_frame_tracker = [
            e for e in self._fire_frame_tracker
            if timestamp - e['timestamp'] <= window_seconds
        ]

        
        
        
        
        
        def _bbox_centre(b):
            return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

        def _bbox_diag(b):
            return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5

        def _centres_plausible(prev_events, curr_events):
            """
            Return True if ANY pair (prev_box, curr_box) has centres within
            max(half_diag_prev, half_diag_curr, 80) pixels.
            80 px floor handles tiny detections at any resolution.
            """
            for pe in prev_events:
                pb = pe.get('bbox')
                if not pb:
                    continue
                for ce in curr_events:
                    cb = ce.get('bbox')
                    if not cb:
                        continue
                    cx_p, cy_p = _bbox_centre(pb)
                    cx_c, cy_c = _bbox_centre(cb)
                    dist = ((cx_p - cx_c) ** 2 + (cy_p - cy_c) ** 2) ** 0.5
                    threshold = max(80.0, _bbox_diag(pb) * 0.6, _bbox_diag(cb) * 0.6)
                    if dist <= threshold:
                        return True
            return False

        
        
        if len(self._fire_frame_tracker) > 1:
            coherent = [self._fire_frame_tracker[-1]]
            for i in range(len(self._fire_frame_tracker) - 2, -1, -1):
                if _centres_plausible(
                    self._fire_frame_tracker[i]['events'],
                    coherent[0]['events'],
                ):
                    coherent.insert(0, self._fire_frame_tracker[i])
                else:
                    
                    break
            self._fire_frame_tracker = coherent

        
        if len(self._fire_frame_tracker) < required_frames:
            return []

        
        
        luminance_history = []
        for entry in self._fire_frame_tracker:
            for fe in entry['events']:
                mv = fe.get('hsv_metrics', {}).get('mean_v')
                if mv is not None:
                    luminance_history.append(mv)

        if len(luminance_history) >= required_frames:
            variance = float(np.var(luminance_history))
            mean_lum = float(np.mean(luminance_history))
            
            
            max_area = 0
            for entry in self._fire_frame_tracker:
                for fe in entry['events']:
                    b = fe.get('bbox', [0, 0, 0, 0])
                    max_area = max(max_area, (b[2] - b[0]) * (b[3] - b[1]))
                    
            
            
            
            is_overexposed = mean_lum > 240.0
            is_massive = max_area > 30000
            
            if variance < 0.8 and not (is_massive or is_overexposed):
                self.log.debug(
                    f"[FireTracker] Rejected static object: variance={variance:.3f} < 0.8"
                )
                
                self._fire_frame_tracker = self._fire_frame_tracker[1:]
                return []

        
        
        best_events = sorted(
            self._fire_frame_tracker[-1]['events'],
            key=lambda e: e.get('fire_score', e.get('confidence', 0.0)),
            reverse=True,
        )

        
        
        if not hasattr(self, '_last_fire_reported_ts'):
            self._last_fire_reported_ts = -999.0

        if timestamp - self._last_fire_reported_ts <= 2.0:
            return []

        self._last_fire_reported_ts = timestamp
        return best_events

    def _associate_weapon_to_person(self, wbbox, persons, pad_ratio, raised_ratio):
        """Associate a weapon bbox to the nearest person whose (padded) box overlaps it.
        Returns {'track_id', 'raised'} or None if no person is near the weapon.

        'raised' approximates "raised above head": the weapon's vertical center sits
        at/above the head zone of the person box (top raised_ratio of its height), or
        the weapon is entirely above the person box top.
        """
        wx1, wy1, wx2, wy2 = wbbox
        w_cx = (wx1 + wx2) / 2.0
        w_cy = (wy1 + wy2) / 2.0
        best = None
        best_overlap = 0.0
        for p in persons:
            px1, py1, px2, py2 = p['bbox']
            ph = py2 - py1
            pw = px2 - px1
            ex1, ey1 = px1 - pw * pad_ratio, py1 - ph * pad_ratio
            ex2, ey2 = px2 + pw * pad_ratio, py2 + ph * pad_ratio
            ix = max(0.0, min(wx2, ex2) - max(wx1, ex1))
            iy = max(0.0, min(wy2, ey2) - max(wy1, ey1))
            inter = ix * iy
            if inter > best_overlap:
                best_overlap = inter
                head_zone_y = py1 + raised_ratio * ph
                raised = (w_cy <= head_zone_y) or (wy2 < py1)
                best = {'track_id': p.get('track_id'), 'raised': bool(raised)}
        return best

    def _vote_weapon_events(self, weapon_hits, min_persist, persist_window, strong_conf=0.60):
        """Temporal persistence gate with a high-recall escalator.

        A weapon run becomes an event when it either (a) recurs across >= min_persist
        sampled frames within persist_window seconds, OR (b) contains a single hit at
        confidence >= strong_conf. A contiguous run merges into one event (CRITICAL if
        any frame had it raised, else HIGH)."""
        events = []
        by_class = {}
        for h in weapon_hits:
            by_class.setdefault(h['weapon_class'], []).append(h)

        def _flush(wclass, run):
            if not run:
                return
            if len(run) >= min_persist or any(h['confidence'] >= strong_conf for h in run):
                events.append(self._build_weapon_event(wclass, run))

        for wclass, hits in by_class.items():
            hits.sort(key=lambda x: x['timestamp'])
            run = []
            for h in hits:
                if run and (h['timestamp'] - run[-1]['timestamp'] > persist_window):
                    _flush(wclass, run)
                    run = []
                run.append(h)
            _flush(wclass, run)
        return events

    def _build_weapon_event(self, wclass, run):
        raised = any(h.get('raised') for h in run)
        return {
            'type': 'weapon',
            'weapon_class': wclass,
            'timestamp': run[len(run) // 2]['timestamp'],
            'confidence': max(h['confidence'] for h in run),
            'severity': 'CRITICAL' if raised else 'HIGH',
            'raised': raised,
            'track_id': run[len(run) // 2].get('track_id'),
            't_start': run[0]['timestamp'],
            't_end': run[-1]['timestamp'],
        }

    def _apply_combinatorial_logic(self, events_list, fused_emotions):
        """
        Applies cross-modal and cross-feature logic rules to escalate or filter events:
        1. Contextual Violence: If violence is detected but faces are 'happy', downgrade.
                                If 'angry' or 'fear', escalate.
        2. Verified Distress: If audio cry + visual fear/sad, create CRITICAL alert.
        3. Suspicious Behavior: If sustained anger/stress over 5 seconds, trigger MEDIUM alert.
        """
        new_events = list(events_list)
        
        
        def get_emotions_at(ts, window=2.0):
            return [fe for fe in fused_emotions if abs(fe['timestamp'] - ts) <= window]

        # Derive fused violence/fighting from the pose 'aggressive_guard' seed corroborated by
        # emotion (anger/fear) or audio (loud shout). A single weak signal (a striking stance
        # alone) is never a standalone violence alert; the corroboration is what gives confidence.
        audio_shouts = [e for e in new_events if e.get('type') in ('loud_shout_impact', 'raised_voice')]
        for guard in [e for e in new_events if e.get('type') == 'aggressive_guard']:
            ts = guard['timestamp']
            emos = get_emotions_at(ts)
            sources = []
            if any(e.get('dominant_emotion') in ('angry', 'fear') for e in emos):
                sources.append('emotion')
            if any(abs(s['timestamp'] - ts) <= 2.0 for s in audio_shouts):
                sources.append('audio')
            if sources:
                new_events.append({
                    'type': 'violence',
                    'timestamp': ts,
                    'confidence': max(guard.get('confidence', 0.0), 0.85),
                    'severity': 'HIGH',
                    'track_id': guard.get('track_id'),
                    'context': 'fused_' + '_'.join(sources),
                })

        # Weapon escalation: a weapon raised above head, or with anger/fear nearby, is CRITICAL.
        for wevt in [e for e in new_events if e.get('type') == 'weapon']:
            emos = get_emotions_at(wevt['timestamp'])
            if wevt.get('raised') or any(e.get('dominant_emotion') in ('angry', 'fear') for e in emos):
                wevt['severity'] = 'CRITICAL'
                wevt['context'] = 'armed_threat'


        for event in new_events:
            if event.get('type') == 'violence':
                emos = get_emotions_at(event['timestamp'])
                if emos:
                    
                    all_dom = [e.get('dominant_emotion') for e in emos]
                    if 'happy' in all_dom or 'surprise' in all_dom:
                        
                        event['context'] = 'context: happy/surprise_detected_nearby'
                    elif 'angry' in all_dom or 'fear' in all_dom:
                        
                        event['severity'] = 'CRITICAL'
                        event['context'] = 'confirmed_assault'

        
        audio_cries = [e for e in new_events if e.get('type') == 'audio_cry']
        for cry in audio_cries:
            emos = get_emotions_at(cry['timestamp'])
            for fe in emos:
                if fe.get('dominant_emotion') in ['fear', 'sad'] or fe.get('visual_stress_score', 0) > 0.8:
                    new_events.append({
                        'type': 'verified_distress',
                        'timestamp': cry['timestamp'],
                        'confidence': max(cry.get('confidence', 0), 0.95),
                        'severity': 'CRITICAL',
                        'context': 'audio_and_visual_distress_fused'
                    })
                    break 

        
        
        person_stress_history = {}
        for fe in fused_emotions:
            tid = fe['track_id']
            if tid not in person_stress_history:
                person_stress_history[tid] = []
            
            is_stressed = fe.get('dominant_emotion') == 'angry' or fe.get('visual_stress_score', 0) > 0.8
            if is_stressed:
                person_stress_history[tid].append(fe['timestamp'])
                
        
        for tid, timestamps in person_stress_history.items():
            timestamps.sort()
            current_streak_start = None
            for i, ts in enumerate(timestamps):
                if current_streak_start is None:
                    current_streak_start = ts
                
                
                if i > 0 and ts - timestamps[i-1] > 2.0:
                    current_streak_start = ts
                    
                
                if ts - current_streak_start >= 5.0:
                    
                    if not any(e.get('type') == 'suspicious_behavior' and e.get('track_id') == tid and abs(e['timestamp'] - ts) < 5.0 for e in new_events):
                        new_events.append({
                            'type': 'suspicious_behavior',
                            'timestamp': ts,
                            'confidence': 0.85,
                            'severity': 'MEDIUM',
                            'track_id': tid,
                            'context': 'sustained_anger_or_stress'
                        })

        return new_events

    def _detect_fall_lying_escalation(self, events_list):
        """
        Post-processing: If falling_down at time T is followed by lying_on_floor
        at time T+delta (delta <= window) with NO standing_up between them,
        escalate to 'fall_and_unresponsive' with CRITICAL severity.
        """
        escalation_config = self.config['models'].get('escalation', {})
        window = escalation_config.get('fall_lying_window_seconds', 15.0)
        severity = escalation_config.get('fall_lying_severity', 'CRITICAL')

        falls = sorted(
            [e for e in events_list if e.get('type') == 'falling_down' or e.get('action') == 'falling_down'],
            key=lambda e: e.get('timestamp', 0)
        )
        lyings = sorted(
            [e for e in events_list if e.get('type') == 'lying_on_floor' or e.get('action') == 'lying_on_floor'],
            key=lambda e: e.get('timestamp', 0)
        )
        standups = sorted(
            [e for e in events_list if e.get('type') == 'standing_up' or e.get('action') == 'standing_up'],
            key=lambda e: e.get('timestamp', 0)
        )

        compound_events = []
        for fall in falls:
            fall_ts = fall.get('timestamp', 0)
            for lying in lyings:
                lying_ts = lying.get('timestamp', 0)
                delta = lying_ts - fall_ts
                if 0 < delta <= window:
                    
                    stood_up = any(
                        fall_ts < su.get('timestamp', 0) < lying_ts
                        for su in standups
                    )
                    if not stood_up:
                        compound_events.append({
                            'type': 'fall_and_unresponsive',
                            'timestamp': fall_ts,
                            'end_timestamp': lying_ts,
                            'confidence': max(fall.get('confidence', 0), lying.get('confidence', 0)),
                            'severity': severity,
                            'details': f'Fall at {fall_ts:.1f}s followed by lying at {lying_ts:.1f}s ({delta:.1f}s gap, no recovery)',
                        })
                        break  

        return compound_events

    def _apply_audio_threat_escalation(self, events_list):
        """Audio-driven RED escalations, emitted as CRITICAL 'loud_shout_panic' events:
          (1) a burst of >= N shouts within a rolling window (repeated shouting = an incident), and
          (2) a shout coinciding with crowd proximity / overcrowding / violence.
        Each escalation pushes the per-second threat index into the RED band via ThreatScorer.
        Must run AFTER proximity + crowd events are appended so rule (2) can see them.
        """
        cfg = self.config['threat'].get('audio_escalation', {})
        n_for_red = int(cfg.get('loud_shout_count_for_red', 3))
        burst_win = float(cfg.get('loud_shout_window_seconds', 10.0))
        fuse_win = float(cfg.get('fusion_window_seconds', 3.0))

        shout_types = ('loud_shout_impact', 'raised_voice')
        shouts = [e for e in events_list if e.get('type') in shout_types]
        if not shouts:
            return []

        escalations = []
        emitted = set()  # dedupe escalations that land on the same second

        def _emit(ts, ctx, conf):
            key = round(ts, 1)
            if key in emitted:
                return
            emitted.add(key)
            escalations.append({
                'type': 'loud_shout_panic',
                'timestamp': ts,
                'confidence': conf,
                'severity': 'CRITICAL',
                'context': ctx,
                'details': f'Loud-shout escalation ({ctx})',
            })

        # Rule 1: >= N distinct shout windows within any rolling burst_win span -> RED.
        # Dedupe near-identical timestamps so overlapping detectors don't inflate the count.
        ts_list = sorted({round(s['timestamp'], 1) for s in shouts})
        for i in range(len(ts_list)):
            j = i
            while j < len(ts_list) and ts_list[j] - ts_list[i] <= burst_win:
                j += 1
            count = j - i
            if count >= n_for_red:
                _emit(ts_list[j - 1], f'{count}_shouts_in_{int(burst_win)}s', 0.95)

        # Rule 2: a shout co-occurring with crowd proximity / overcrowding / violence -> RED.
        companions = [e for e in events_list
                      if e.get('type') in ('proximity_alert', 'overcrowding', 'violence', 'aggressive_guard')]
        for s in shouts:
            st = s['timestamp']
            hit = next((c for c in companions if abs(c.get('timestamp', 0.0) - st) <= fuse_win), None)
            if hit is not None:
                _emit(st, f'shout_with_{hit["type"]}', 0.95)

        return escalations

    def process_video(self, video_path, progress_callback=None):
        """
        Main pipeline entry point. Processes a video through all stages.

        Returns:
            dict: Paths to all generated outputs, or None on failure.
        """
        pipeline_start = time.time()
        self.log.info(f"Starting Full Video Pipeline for: {Path(video_path).name}")
        if progress_callback:
            progress_callback("Initializing Pipeline")

        
        run_dir = self._create_run_directory(video_path)
        run_dir_name = run_dir.name

        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frame_count / fps
        cap.release()

        
        if progress_callback:
            progress_callback("Checking Audio Streams")
        has_audio = check_audio_track(video_path)
        self.log.info(f"Audio: {'detected' if has_audio else 'not found (video-only mode)'}")

        
        if progress_callback:
            progress_callback("Stage 0: Adaptive Frame Sampling")
        stage_start = time.time()
        sampled_frames, motion_clips = self.frame_sampler.extract(video_path)
        if not sampled_frames:
            self.log.error("No frames sampled from video.")
            return None
        self.log.info(f"Stage 0 complete: {len(sampled_frames)} frames, {len(motion_clips)} clips [{time.time()-stage_start:.1f}s]")

        
        aggregated_results = {
            'total_frames': total_frame_count,
            'frames_sampled': len(sampled_frames),
            'fusion_mode': 'audio_visual' if (has_audio and self._should_run('audio_analysis')) else 'visual_only',
            'unique_count': 0,
            'individuals': [],
            'events': [],
            'chapters': [],
            'threat_timeline': [],
            'description': "",
            'emotions': [],
            'analytics': {},
            'llm_provider': self.llm_provider,
        }

        faces_per_frame_map = {}
        fire_per_frame_map = {}
        weapon_per_frame_map = {}


        person_results = {'unique_count': 0, 'per_frame': []}
        if self._should_run('person_detection'):
            if progress_callback:
                progress_callback("Stage 1: Person Tracking (YOLOv8n + ByteTrack)")
            stage_start = time.time()
            try:
                detector = PersonDetector(self.manager, self.config)
                person_results = detector.process_frames(sampled_frames)
                aggregated_results['unique_count'] = person_results['unique_count']
                self.log.info(f"Stage 1 complete: {person_results['unique_count']} unique persons [{time.time()-stage_start:.1f}s]")
            except Exception as e:
                self.log.error(f"Stage 1 (Person Detection) failed: {e}")
            finally:
                self.manager.unload('yolo_person')

        
        track_identity_map = {}
        track_face_conf_map = {}

        
        face_detections_timeline = []
        emotion_results = []

        if self._should_run('face_pipeline') or self._should_run('emotion_analysis'):
            if progress_callback:
                progress_callback("Stage 2 & 3: Face Recognition & Emotion Analysis")
            stage_start = time.time()

            face_pipeline = None
            if self._should_run('face_pipeline'):
                try:
                    face_pipeline = FacePipeline(self.config)
                except Exception as e:
                    self.log.error(f"Face pipeline init failed: {e}")

            emotion_analyzer = None
            if self._should_run('emotion_analysis'):
                try:
                    emotion_analyzer = EmotionAnalyzer(self.config, self.manager)
                except Exception as e:
                    self.log.error(f"Emotion analyzer init failed: {e}")

            for idx, frame_data in enumerate(sampled_frames):
                frame = frame_data['frame']
                ts = frame_data['timestamp']

                persons_in_frame = []
                for pf in person_results['per_frame']:
                    if pf['frame_idx'] == frame_data['frame_idx']:
                        persons_in_frame = pf['persons']
                        break

                if not persons_in_frame:
                    continue

                
                faces_in_frame = []
                if face_pipeline:
                    try:
                        faces_in_frame = face_pipeline.process(frame, [p['bbox'] for p in persons_in_frame])
                    except Exception as e:
                        self.log.warning(f"Face detection error at {ts:.1f}s: {e}")

                faces_per_frame_map[frame_data['frame_idx']] = faces_in_frame

                
                for face in faces_in_frame:
                    fx1, fy1, fx2, fy2 = face['bbox']
                    best_track_id = None
                    best_overlap = 0.0

                    for person in persons_in_frame:
                        px1, py1, px2, py2 = person['bbox']
                        ix1 = max(fx1, px1)
                        iy1 = max(fy1, py1)
                        ix2 = min(fx2, px2)
                        iy2 = min(fy2, py2)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        if inter > best_overlap:
                            best_overlap = inter
                            best_track_id = person['track_id']

                    if best_track_id is not None:
                        face['track_id'] = best_track_id
                        track_identity_map[best_track_id] = face['identity']
                        track_face_conf_map[best_track_id] = face['confidence']

                        
                        if emotion_analyzer and face_pipeline:
                            try:
                                matched_raw_face = face.get('face_obj')

                                if matched_raw_face:
                                    emo_data = emotion_analyzer.analyze_face(frame, matched_raw_face, track_id=best_track_id)

                                    smoothed_cry = self.smoother.smooth(best_track_id, 'cry', emo_data['visual_cry_prob'])
                                    smoothed_stress = self.smoother.smooth(best_track_id, 'stress', emo_data['visual_stress_score'])

                                    emotion_results.append({
                                        'track_id': best_track_id,
                                        'timestamp': ts,
                                        'emotion': emo_data['dominant_emotion'],
                                        'emotion_scores': emo_data['emotion_scores'],
                                        'visual_cry_prob': smoothed_cry,
                                        'visual_stress_score': smoothed_stress,
                                    })
                            except Exception as e:
                                self.log.warning(f"Emotion analysis error at {ts:.1f}s: {e}")

            if face_pipeline:
                self.manager.unload('buffalo_l')
            self.log.info(f"Stage 2&3 complete: {len(track_identity_map)} faces, {len(emotion_results)} emotion readings [{time.time()-stage_start:.1f}s]")

        
        for tid in track_identity_map.keys():
            aggregated_results['individuals'].append({
                'track_id': tid,
                'face_id': track_identity_map[tid],
                'face_confidence': track_face_conf_map.get(tid),
            })

        
        events_list = []

        if self._should_run('event_detection') or self._should_run('indoor_action') or self._should_run('violence_detection') or self._should_run('weapon_detection'):
            if progress_callback:
                progress_callback("Stage 4: Incident & Action Detection (YOLO11x + Dual-Head R3D-18)")
            stage_start = time.time()

            event_detector = None
            if self._should_run('event_detection'):
                try:
                    event_detector = EventDetector(self.manager, self.config)
                except Exception as e:
                    self.log.error(f"Event detector init failed: {e}")

            weapon_detector = None
            if self._should_run('weapon_detection'):
                try:
                    weapon_detector = WeaponDetector(self.manager, self.config)
                except Exception as e:
                    self.log.error(f"Weapon detector init failed: {e}")

            violence_classifier = None
            if self._should_run('violence_detection'):
                try:
                    if self.config['models']['violence'].get('method') == 'stgcn':
                        violence_classifier = ViolenceSTGCN(self.manager, self.config)
                    else:
                        violence_classifier = ViolenceClassifier(self.manager, self.config)
                except Exception as e:
                    self.log.error(f"Violence classifier init failed: {e}")

            action_detector = None
            action_stgcn = None
            if self._should_run('indoor_action'):
                try:
                    action_detector = IndoorActionDetector(self.manager, self.config)
                except Exception as e:
                    self.log.error(f"Indoor action detector init failed: {e}")
                if self.config['models']['indoor_action'].get('stgcn_model_path'):
                    try:
                        action_stgcn = IndoorActionSTGCN(self.manager, self.config)
                    except Exception as e:
                        self.log.error(f"Indoor action ST-GCN init failed: {e}")

            
            context_history = []
            try:
                def _load_dual_head():
                    model = DualHeadR3D18()
                    ckpt = torch.load(
                        self.config['models']['indoor_action'].get('dualhead_model_path', 'models/weights/indoor_action_dualhead_best.pt'),
                        map_location='cpu'
                    )
                    model.load_state_dict(ckpt['model_state_dict'])
                    model.half()
                    return model
                
                shared_dual_head = self.manager.load_torch_model('dual_head_r3d', _load_dual_head)
                behavioral_engine = BehavioralContextEngine(self.manager, self.config, shared_model=shared_dual_head)
                
                
                for i in range(0, len(sampled_frames) - 15, 8):
                    clip = sampled_frames[i:i+16]
                    frames = [f['frame'] for f in clip]
                    mid_ts = clip[8]['timestamp']
                    states = behavioral_engine.analyze_clip(frames)
                    context_history.append({'timestamp': mid_ts, 'states': states})
                    
                
                self.manager.unload('dual_head_r3d')
                self.log.info(f"Behavioral Context computed for {len(context_history)} clips.")
                
            except Exception as e:
                self.log.error(f"Behavioral Context Engine failed: {e}")
                
            
            for ctx in context_history:
                ts = ctx['timestamp']
                st = ctx['states']
                
                
                has_occupants = False
                for pf in person_results['per_frame']:
                    
                    pf_ts = pf.get('timestamp', pf['frame_idx'] / self.config['sampling']['base_fps'])
                    if abs(pf_ts - ts) <= 1.0:
                        if len(pf.get('persons', [])) > 0:
                            has_occupants = True
                            break
                            
                if not has_occupants:
                    continue  
                
                for es in st.get('essential', []):
                    events_list.append({'type': es['type'], 'timestamp': ts, 'confidence': es['confidence'], 'severity': 'MEDIUM'})
                for ind in st.get('indoor_states', []):
                    sev = 'HIGH' if ind['type'] in ['falling_down', 'lying_on_floor'] else 'LOW'
                    events_list.append({'type': ind['type'], 'timestamp': ts, 'confidence': ind['confidence'], 'severity': sev})
                for en in st.get('enrichment', []):
                    events_list.append({'type': en['type'], 'timestamp': ts, 'confidence': en['confidence'], 'severity': 'INFO'})

            
            self._fire_frame_tracker = []  
            self._recent_persons = []      
            self._lying_timestamps = []    

            # Precompute persons and faces per frame for easy access
            frame_persons_map = {}
            for pf in person_results['per_frame']:
                frame_persons_map[pf['frame_idx']] = pf['persons']

            # 1. Fire Detection Pass (Batch process all frames)
            if event_detector:
                self.log.info("Running Fire Detection pass...")
                for idx, frame_data in enumerate(sampled_frames):
                    try:
                        raw_fire_events = event_detector.detect_fire(frame_data['frame'])
                        persons_in_this_frame = frame_persons_map.get(frame_data['frame_idx'], [])
                        faces_in_this_frame = faces_per_frame_map.get(frame_data['frame_idx'], [])
                        
                        curr_ts = frame_data['timestamp']
                        for p in persons_in_this_frame:
                            self._recent_persons.append({'bbox': p['bbox'], 'timestamp': curr_ts})
                        self._recent_persons = [p for p in self._recent_persons if curr_ts - p['timestamp'] <= 3.0]

                        validated_fire = self._validate_fire_detection(
                            raw_fire_events, frame_data['frame'], persons_in_this_frame, faces_in_this_frame, self._recent_persons
                        )
                        fire_per_frame_map[frame_data['frame_idx']] = validated_fire

                        confirmed_fire = self._check_fire_temporal_consistency(validated_fire, curr_ts)
                        for fe in confirmed_fire:
                            fe['timestamp'] = curr_ts
                            fe['severity'] = 'HIGH'
                            events_list.append(fe)
                    except Exception as e:
                        self.log.warning(f"Fire detection error at frame {idx}: {e}")
                self.manager.unload('fire_yolo')

            # 1b. Weapon Detection Pass (gated escalator: confidence + proximity + persistence)
            if weapon_detector:
                self.log.info("Running Weapon Detection pass...")
                w_cfg = self.config['models']['weapon']
                pad = float(w_cfg.get('proximity_pad_ratio', 0.15))
                raised_ratio = float(w_cfg.get('raised_head_ratio', 0.12))
                min_persist = int(w_cfg.get('min_persist_frames', 3))
                persist_window = float(w_cfg.get('persist_window_seconds', 2.0))
                strong_conf = float(w_cfg.get('strong_confidence', 0.60))

                weapon_hits = []
                try:
                    for frame_data in sampled_frames:
                        persons_in_this_frame = frame_persons_map.get(frame_data['frame_idx'], [])
                        dets = weapon_detector.detect(frame_data['frame'])
                        if not dets:
                            continue
                        kept = []
                        for d in dets:
                            assoc = (self._associate_weapon_to_person(d['bbox'], persons_in_this_frame, pad, raised_ratio)
                                     if persons_in_this_frame else None)
                            if assoc is not None:
                                d['track_id'] = assoc['track_id']
                                d['raised'] = assoc['raised']
                            elif d['confidence'] >= strong_conf:
                                # High-recall: a strong detection with no nearby person still
                                # flags (HIGH, never "raised") instead of being dropped.
                                d['track_id'] = None
                                d['raised'] = False
                            else:
                                continue  # weak + unassociated → drop to avoid FP noise
                            d['timestamp'] = frame_data['timestamp']
                            kept.append(d)
                        if kept:
                            weapon_per_frame_map[frame_data['frame_idx']] = kept
                            weapon_hits.extend(kept)
                except Exception as e:
                    self.log.warning(f"Weapon detection error: {e}")
                finally:
                    weapon_detector.unload()

                weapon_events = self._vote_weapon_events(weapon_hits, min_persist, persist_window, strong_conf)
                events_list.extend(weapon_events)
                self.log.info(f"Weapon pass: {len(weapon_events)} event(s) from {len(weapon_hits)} gated detections.")

            # Shared pose cache: extract COCO-17 skeletons ONCE for the ST-GCN passes
            # (violence + home action) instead of re-running yolo11x-pose per overlapping
            # clip window. Collapses ~8x redundant pose inference into one batched pass.
            pose_cache = None
            pose_H = pose_W = 1.0
            use_shared_pose = isinstance(violence_classifier, ViolenceSTGCN) or action_stgcn is not None
            if use_shared_pose and sampled_frames:
                try:
                    from ultralytics import YOLO
                    pose_path = (self.config['models']['violence'].get('pose_model_path')
                                 or self.config['models']['indoor_action'].get('model_path',
                                                                               'models/weights/yolo11x-pose.pt'))
                    # FP32 pose: half-precision batched pose inference on this GPU intermittently
                    # returns ZERO detections (degenerate FP16), which silently emptied the cache and
                    # killed skeleton overlays + ST-GCN input. FP32 is reliable and the cost over one
                    # batched pass on the sampled frames is negligible.
                    pose_model = self.manager.load_torch_model(
                        'stage4_pose', lambda: YOLO(pose_path), keep_previous=True, skip_fp16=True)
                    use_half = False
                    pose_H, pose_W = sampled_frames[0]['frame'].shape[:2]

                    # Batch size scales with free VRAM (user runs ~3.6GB headroom on a 4GB GPU).
                    free_mb = self.manager.get_free_vram_mb()
                    batch = 32 if free_mb > 3000 else (16 if free_mb > 1500 else 8)

                    pose_cache = {}
                    for b in range(0, len(sampled_frames), batch):
                        chunk = sampled_frames[b:b + batch]
                        res_list = pose_model.predict([f['frame'] for f in chunk], conf=0.30,
                                                      classes=[0], verbose=False, half=use_half)
                        for fd, res in zip(chunk, res_list):
                            kp_arr = np.zeros((0, 17, 3), dtype=np.float32)
                            if res.keypoints is not None and res.keypoints.data is not None:
                                kp = res.keypoints.data.cpu().numpy()
                                if kp.shape[0] > 0:
                                    # Store ALL persons, sorted best-first. skeleton_clip_from_cache
                                    # still picks the top-max_persons internally (ST-GCN unchanged),
                                    # while annotation can now draw everyone in the frame.
                                    order = np.argsort(-kp[:, :, 2].mean(axis=1))
                                    kp_arr = kp[order]
                            pose_cache[fd['frame_idx']] = kp_arr
                    self.log.info(f"Shared pose cache built: {len(pose_cache)} frames "
                                  f"(batch={batch}, free={free_mb:.0f}MB).")
                    # Skeletons are now CPU numpy; the pose model is no longer needed.
                    self.manager.unload('stage4_pose')
                except Exception as e:
                    self.log.warning(f"Shared pose cache failed ({e}); ST-GCN passes fall back to per-clip pose.")
                    pose_cache = None

            # 2. Violence Detection Pass (Temporal VideoMAE classifier)
            if violence_classifier:
                self.log.info(f"Running Violence Detection pass ({self.config['models']['violence'].get('method', 'videomae')})...")
                v_cfg = self.config['models']['violence']
                stride = int(v_cfg.get('clip_stride', 8))
                num_frames = int(v_cfg.get('num_frames', 16))
                conf_thresh = float(v_cfg.get('confidence_threshold', 0.65))
                min_consec = int(v_cfg.get('min_consecutive_clips', 2))

                clip_results = []
                try:
                    for i in range(0, len(sampled_frames) - (num_frames - 1), stride):
                        clip = sampled_frames[i:i + num_frames]
                        mid_ts = clip[num_frames // 2]['timestamp']

                        # Occupancy gate: only score clips where persons are present
                        has_occupants = False
                        for pf in person_results['per_frame']:
                            pf_ts = pf.get('timestamp', pf['frame_idx'] / self.config['sampling']['base_fps'])
                            if abs(pf_ts - mid_ts) <= 1.0 and len(pf.get('persons', [])) > 0:
                                has_occupants = True
                                break
                        if not has_occupants:
                            continue

                        if pose_cache is not None and isinstance(violence_classifier, ViolenceSTGCN):
                            prob = violence_classifier.classify_clip(
                                None, frame_skeletons=pose_cache,
                                frame_keys=[f['frame_idx'] for f in clip], frame_hw=(pose_H, pose_W))
                        else:
                            prob = violence_classifier.classify_clip([f['frame'] for f in clip])
                        clip_results.append({
                            'prob': prob,
                            'mid_ts': mid_ts,
                            't_start': clip[0]['timestamp'],
                            't_end': clip[-1]['timestamp'],
                        })
                except Exception as e:
                    self.log.warning(f"Violence detection error: {e}")
                finally:
                    violence_classifier.unload()

                # Temporal voting: merge contiguous runs of positive clips into one event.
                # The trailing sentinel forces the final run to flush.
                run = []
                violence_events = []
                for c in clip_results + [{'prob': -1.0}]:
                    if c['prob'] >= conf_thresh:
                        run.append(c)
                    else:
                        if len(run) >= min_consec:
                            violence_events.append({
                                'type': 'violence',
                                'timestamp': run[len(run) // 2]['mid_ts'],
                                'confidence': max(x['prob'] for x in run),
                                'severity': 'HIGH',
                                't_start': run[0]['t_start'],
                                't_end': run[-1]['t_end'],
                            })
                        run = []
                events_list.extend(violence_events)
                self.log.info(f"Violence pass: {len(violence_events)} event(s) from {len(clip_results)} scored clips.")

            # 3. Action / Pose Pass (Batch process all frames)
            if action_detector:
                self.log.info("Running Indoor Action pass...")
                for idx, frame_data in enumerate(sampled_frames):
                    try:
                        persons_in_this_frame = frame_persons_map.get(frame_data['frame_idx'], [])
                        act_events = action_detector.process_frame(frame_data['frame'], persons_in_this_frame)
                        for ae in act_events:
                            ae['timestamp'] = frame_data['timestamp']
                            ae['action'] = ae['type']
                            
                            if ae['action'] == 'lying_on_floor':
                                self._lying_timestamps.append(frame_data['timestamp'])
                                self._lying_timestamps = [t for t in self._lying_timestamps if frame_data['timestamp'] - t <= 15.0]
                                
                                if len(self._lying_timestamps) >= 3:
                                    ae['severity'] = "CRITICAL"
                                    ae['is_alert'] = True
                                else:
                                    ae['severity'] = "LOW"
                                    ae['is_alert'] = False
                            else:
                                ae['is_alert'] = ae.get('severity', 'LOW') in ['HIGH', 'CRITICAL']
                                
                            events_list.append(ae)
                    except Exception as e:
                        self.log.warning(f"Indoor action error at frame {idx}: {e}")
                self.manager.unload('pose_yolo')

            # 4. Indoor Action ST-GCN Clip Pass (fine-tuned skeleton classifier; runs
            #    alongside the geometric pass above and feeds the same consumers).
            if action_stgcn:
                self.log.info("Running Indoor Action ST-GCN pass...")
                num_frames = int(self.config['models']['indoor_action'].get('stgcn_clip_frames', 32))
                stride = max(1, num_frames // 4)
                stgcn_events = 0
                try:
                    for i in range(0, len(sampled_frames) - (num_frames - 1), stride):
                        clip = sampled_frames[i:i + num_frames]
                        mid = clip[num_frames // 2]
                        mid_ts = mid['timestamp']

                        has_occupants = False
                        for pf in person_results['per_frame']:
                            pf_ts = pf.get('timestamp', pf['frame_idx'] / self.config['sampling']['base_fps'])
                            if abs(pf_ts - mid_ts) <= 1.0 and len(pf.get('persons', [])) > 0:
                                has_occupants = True
                                break
                        if not has_occupants:
                            continue

                        if pose_cache is not None:
                            det = action_stgcn.classify_clip(
                                None, frame_skeletons=pose_cache,
                                frame_keys=[f['frame_idx'] for f in clip], frame_hw=(pose_H, pose_W))
                        else:
                            det = action_stgcn.classify_clip([f['frame'] for f in clip])
                        if det is None:
                            continue

                        ae = {'type': det['class_name'], 'action': det['class_name'],
                              'timestamp': mid_ts, 'confidence': det['confidence'], 'source': 'stgcn'}
                        if det['class_name'] == 'lying_on_floor':
                            self._lying_timestamps.append(mid_ts)
                            self._lying_timestamps = [t for t in self._lying_timestamps if mid_ts - t <= 15.0]
                            if len(self._lying_timestamps) >= 3:
                                ae['severity'] = "CRITICAL"
                                ae['is_alert'] = True
                            else:
                                ae['severity'] = "LOW"
                                ae['is_alert'] = False
                        else:  # falling_down
                            ae['severity'] = "CRITICAL"
                            ae['is_alert'] = True
                        events_list.append(ae)
                        stgcn_events += 1
                except Exception as e:
                    self.log.warning(f"Indoor action ST-GCN error: {e}")
                finally:
                    action_stgcn.unload()
                self.log.info(f"Indoor Action ST-GCN pass: {stgcn_events} alert event(s).")
            self.log.info(f"Stage 4 complete: {len(events_list)} raw events [{time.time()-stage_start:.1f}s]")

        
        audio_results = None
        if has_audio and self._should_run('audio_analysis'):
            if progress_callback:
                progress_callback("Stage 5: Audio Analysis (Librosa)")
            stage_start = time.time()
            try:
                audio_analyzer = AudioAnalyzer(self.config, self.manager)
                audio_results = audio_analyzer.process(video_path)

                if audio_results:
                    for cry in audio_results.get('cry_segments', []):
                        events_list.append({
                            'type': 'audio_cry',
                            'timestamp': cry['timestamp'],
                            'confidence': cry['probability'],
                            'severity': 'MEDIUM',
                        })
                    # AudioCNN-confirmed cries (fine-tuned model) — higher-confidence corroboration.
                    for cry in audio_results.get('cnn_cry_segments', []):
                        events_list.append({
                            'type': 'audio_cry',
                            'timestamp': cry['timestamp'],
                            'confidence': cry['probability'],
                            'severity': 'MEDIUM',
                            'source': 'audio_cnn',
                        })
                    for loud in audio_results.get('loud_events', []):
                        events_list.append({
                            'type': 'loud_shout_impact',
                            'timestamp': loud['timestamp'],
                            'confidence': 0.85,
                            'severity': 'HIGH',
                        })
                    # Sustained raised voice (librosa z-RMS spike). Was computed but discarded;
                    # now surfaced so it counts toward shout escalation and the violence fusion.
                    for rv in audio_results.get('raised_voice_segments', []):
                        events_list.append({
                            'type': 'raised_voice',
                            'timestamp': rv['timestamp'],
                            'confidence': 0.75,
                            'severity': 'MEDIUM',
                        })
                self.log.info(f"Stage 5 complete [{time.time()-stage_start:.1f}s]")
            except Exception as e:
                self.log.error(f"Audio analysis failed: {e}")

        
        fused_emotions = emotion_results
        if has_audio and audio_results and self._should_run('audio_analysis'):
            try:
                fuser = AudioVisualFuser(self.config)
                fused_emotions = fuser.fuse(emotion_results, audio_results)
            except Exception as e:
                self.log.warning(f"Audio-visual fusion error: {e}")
        else:
            for fe in fused_emotions:
                fe['fused_cry_prob'] = fe.get('visual_cry_prob', 0.0)

        
        for fe in fused_emotions:
            
            threshold = 0.50 if not (has_audio and audio_results and self._should_run('audio_analysis')) else 0.60
            if fe.get('fused_cry_prob', 0.0) >= threshold:
                events_list.append({
                    'type': 'person_cry',
                    'timestamp': fe['timestamp'],
                    'confidence': fe['fused_cry_prob'],
                    'severity': 'MEDIUM',
                    'track_id': fe['track_id'],
                })

        
        events_list = self._apply_combinatorial_logic(events_list, fused_emotions)

        
        seen_proximity_pairs = {}
        for idx, frame_data in enumerate(sampled_frames):
            persons_in_frame = []
            for pf in person_results['per_frame']:
                if pf['frame_idx'] == frame_data['frame_idx']:
                    persons_in_frame = pf['persons']
                    break

            prox_events = self.proximity_analyzer.analyze(persons_in_frame)
            for pe in prox_events:
                pair_key = tuple(sorted(pe['track_ids']))
                window_key = (pair_key, int(frame_data['timestamp'] / 2.0))
                if window_key in seen_proximity_pairs:
                    continue
                seen_proximity_pairs[window_key] = True

                events_list.append({
                    'type': 'proximity_alert',
                    'timestamp': frame_data['timestamp'],
                    'confidence': 1.0,
                    'severity': pe['severity'],
                    'details': f"Track IDs {pe['track_ids']} too close",
                })

        
        if self.config.get('analytics', {}).get('loitering_detection', True):
            if progress_callback:
                progress_callback("Analytics: Loitering Detection")
            try:
                loiter_events = self.loitering_detector.detect(person_results['per_frame'])
                events_list.extend(loiter_events)
                self.log.info(f"Loitering detection: {len(loiter_events)} events")
            except Exception as e:
                self.log.warning(f"Loitering detection error: {e}")

        
        crowd_data = None
        if self.config.get('analytics', {}).get('crowd_analysis', True):
            if progress_callback:
                progress_callback("Analytics: Crowd Density Analysis")
            try:
                crowd_data = self.crowd_analyzer.analyze(person_results['per_frame'])
                events_list.extend(crowd_data.get('overcrowding_events', []))
                aggregated_results['analytics']['crowd'] = {
                    'max_count': crowd_data['max_count'],
                    'avg_count': crowd_data['avg_count'],
                    'density_timeline': crowd_data['density_timeline'],
                }
                self.log.info(f"Crowd analysis: max={crowd_data['max_count']}, avg={crowd_data['avg_count']:.1f}")
            except Exception as e:
                self.log.warning(f"Crowd analysis error: {e}")

        
        events_list = self._deduplicate_events(events_list, time_window=2.0)
        self.log.info(f"Total events after deduplication: {len(events_list)}")

        
        compound_events = self._detect_fall_lying_escalation(events_list)
        if compound_events:
            events_list.extend(compound_events)
            self.log.info(f"Escalation: {len(compound_events)} fall_and_unresponsive compound events detected")

        aggregated_results['events'] = events_list

        
        aggregated_results['emotions'] = emotion_results

        
        description = "No narrative summary generated."
        if self._should_run('video_description'):
            if progress_callback:
                progress_callback(f"Stage 6: VLM Narrative Engine ({self.llm_provider.upper()})")
            stage_start = time.time()

            try:
                keyframe_indices = [0]
                if len(sampled_frames) > 1:
                    keyframe_indices.append(len(sampled_frames) - 1)
                motion_scores = [f['motion_score'] for f in sampled_frames]
                if motion_scores:
                    keyframe_indices.append(int(np.argmax(motion_scores)))
                if events_list:
                    highest_conf_event = max(events_list, key=lambda x: x.get('confidence', 0.0))
                    evt_ts = highest_conf_event.get('timestamp', duration / 2.0)
                    diffs = [abs(f['timestamp'] - evt_ts) for f in sampled_frames]
                    keyframe_indices.append(int(np.argmin(diffs)))
                else:
                    keyframe_indices.append(len(sampled_frames) // 2)
                person_counts = [len(pf['persons']) for pf in person_results['per_frame']]
                if person_counts:
                    keyframe_indices.append(int(np.argmax(person_counts)))

                unique_indices = sorted(list(set(keyframe_indices)))
                
                highest_motion_idx = int(np.argmax(motion_scores)) if motion_scores else 0
                highest_motion_frame = None
                if highest_motion_idx < len(sampled_frames):
                    highest_motion_frame = cv2.cvtColor(sampled_frames[highest_motion_idx]['frame'], cv2.COLOR_BGR2RGB)

                describer = VideoDescriber(self.manager, self.config, llm_provider_name=self.llm_provider)

                frame_descriptions = []
                keyframes_dir = run_dir / "keyframes"
                for k_idx in unique_indices:
                    if k_idx < len(sampled_frames):
                        frame_data = sampled_frames[k_idx]
                        rgb_frame = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)

                        
                        kf_path = keyframes_dir / f"keyframe_{k_idx:04d}_{frame_data['timestamp']:.1f}s.jpg"
                        cv2.imwrite(str(kf_path), frame_data['frame'], [cv2.IMWRITE_JPEG_QUALITY, 90])

                        desc = describer.describe_keyframe(rgb_frame)
                        frame_descriptions.append(f"At {frame_data['timestamp']:.1f}s: {desc}")

                event_summary = ", ".join([f"{e['type']} at {e['timestamp']:.1f}s" for e in events_list[:5]])
                if frame_descriptions:
                    description = describer.synthesize_summary(frame_descriptions, event_summary, context_image=highest_motion_frame)

                aggregated_results['description'] = description

                if describer.mode == 'local':
                    self.manager.unload('moondream')
                self.log.info(f"Stage 6 complete [{time.time()-stage_start:.1f}s]")
            except Exception as e:
                self.log.error(f"Video description failed: {e}")
                aggregated_results['description'] = f"Narrative generation failed: {str(e)}"

        
        # Audio-driven RED escalation: burst of shouts, or a shout fused with crowd/violence.
        # Runs here so proximity + overcrowding events are already in events_list.
        audio_escalations = self._apply_audio_threat_escalation(events_list)
        if audio_escalations:
            events_list.extend(audio_escalations)
            self.log.info(f"Audio escalation: {len(audio_escalations)} RED 'loud_shout_panic' event(s) raised.")

        self.log.info("Computing per-second threat index timeline...")
        timeline = []
        for sec in range(int(np.ceil(duration))):
            ts = float(sec)
            events_in_window = [e for e in events_list if abs(e['timestamp'] - ts) <= 1.0]
            emotions_in_window = [e for e in fused_emotions if abs(e['timestamp'] - ts) <= 1.0]
            emo_res = emotions_in_window[0] if emotions_in_window else None
            score, level = self.threat_scorer.calculate_score(events_in_window, emo_res)
            timeline.append({'ts': ts, 'score': score, 'level': level})
        aggregated_results['threat_timeline'] = timeline

        
        self.log.info("Generating video chapter divisions...")
        action_detections = [e for e in events_list if e['type'] in self.config['models']['indoor_action']['classes']]
        chapters = self.chapter_generator.generate(duration, events_list, action_detections)
        aggregated_results['chapters'] = chapters

        
        analytics_dir = run_dir / "analytics"
        heatmap_path = None
        trajectory_path = None
        behavioral_data = None

        if progress_callback:
            progress_callback("Analytics: Generating Heatmap, Trajectories & Behavioral Scores")

        
        if self.config.get('analytics', {}).get('heatmap', True) and sampled_frames:
            try:
                frame_shape = sampled_frames[0]['frame'].shape
                heatmap_out = str(analytics_dir / "heatmap.png")
                self.heatmap_gen.generate(person_results['per_frame'], frame_shape, heatmap_out)
                heatmap_path = heatmap_out
                self.log.info("Heatmap generated.")
            except Exception as e:
                self.log.warning(f"Heatmap generation failed: {e}")

        
        if self.config.get('analytics', {}).get('trajectories', True) and sampled_frames:
            try:
                frame_shape = sampled_frames[0]['frame'].shape
                traj_out = str(analytics_dir / "trajectories.png")
                self.trajectory_viz.generate(person_results['per_frame'], frame_shape, traj_out)
                trajectory_path = traj_out
                self.log.info("Trajectory visualization generated.")
            except Exception as e:
                self.log.warning(f"Trajectory generation failed: {e}")

        
        if self.config.get('analytics', {}).get('behavioral_scoring', True):
            try:
                behavioral_data = self.behavioral_scorer.score(
                    person_results['per_frame'], fused_emotions, events_list
                )
                aggregated_results['analytics']['behavioral_scores'] = behavioral_data
                self.log.info(f"Behavioral scoring: {len(behavioral_data)} persons scored")
            except Exception as e:
                self.log.warning(f"Behavioral scoring failed: {e}")

        
        if progress_callback:
            progress_callback("Finalizing: Saving Reports & Annotated Feeds")

        reporter = ReportGenerator(self.config)
        json_path = reporter.build_json(Path(video_path).name, duration, aggregated_results, run_dir)
        html_path = reporter.build_html(Path(video_path).name, duration, json_path, run_dir)
        srt_path = reporter.build_srt(Path(video_path).stem, events_list, run_dir)

        
        pdf_path = None
        if self.config['output'].get('pdf_report', True):
            try:
                from utils.pdf_report import PDFReportGenerator
                pdf_gen = PDFReportGenerator()
                pdf_path = pdf_gen.generate(json_path, run_dir, heatmap_path, trajectory_path)
                self.log.info(f"PDF report generated: {pdf_path}")
            except Exception as e:
                self.log.warning(f"PDF report generation failed: {e}")

        
        clip_extractor = ClipExtractor(self.config, run_dir=run_dir)
        if self.config['clips']['enabled']:
            self.log.info("Extracting video evidence clips around incidents...")
            for idx, e in enumerate(events_list):
                if e.get('severity') in ["HIGH", "CRITICAL"] or e.get('type') in ['fire/smoke', 'violence', 'falling_down', 'fall_and_unresponsive']:
                    clip_extractor.extract(video_path, e['type'], e['timestamp'])

        
        if self.config['output']['annotated_video']:
            if progress_callback:
                progress_callback("Stage 8: Generating Smooth Annotated Video")
            try:
                annotated_dir = run_dir / "annotated"
                annotated_dir.mkdir(parents=True, exist_ok=True)
                out_video_path = annotated_dir / f"annotated_{Path(video_path).stem}.mp4"

                cap = cv2.VideoCapture(str(video_path))
                orig_fps = cap.get(cv2.CAP_PROP_FPS)
                if orig_fps <= 0: orig_fps = 30.0
                
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                
                # Prefer H.264 (avc1) so the feed plays in browsers / Streamlit st.video.
                # If this OpenCV build has no H.264 encoder, fall back to mp4v into a temp
                # file and transcode to H.264 with ffmpeg after the render completes.
                write_path = out_video_path
                needs_transcode = False
                out_writer = cv2.VideoWriter(str(out_video_path),
                                             cv2.VideoWriter_fourcc(*'avc1'), orig_fps, (w, h))
                if not out_writer.isOpened():
                    needs_transcode = True
                    write_path = annotated_dir / f"_raw_{Path(video_path).stem}.mp4"
                    out_writer = cv2.VideoWriter(str(write_path),
                                                 cv2.VideoWriter_fourcc(*'mp4v'), orig_fps, (w, h))

                sampled_idxs = sorted([f['frame_idx'] for f in sampled_frames])
                if not sampled_idxs:
                    sampled_idxs = [0]
                    
                person_map = {pf['frame_idx']: pf['persons'] for pf in person_results['per_frame']}

                recent_logs = []
                last_seen_events = set()
                
                orig_frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    t = orig_frame_idx / orig_fps
                    
                    prev_s = sampled_idxs[0]
                    next_s = sampled_idxs[-1]
                    for i in range(len(sampled_idxs)-1):
                        if sampled_idxs[i] <= orig_frame_idx < sampled_idxs[i+1]:
                            prev_s = sampled_idxs[i]
                            next_s = sampled_idxs[i+1]
                            break
                            
                    if orig_frame_idx >= sampled_idxs[-1]:
                        prev_s = sampled_idxs[-1]
                        next_s = sampled_idxs[-1]
                        
                    fraction = 0.0
                    if next_s != prev_s:
                        fraction = (orig_frame_idx - prev_s) / (next_s - prev_s)
                        
                    prev_persons = person_map.get(prev_s, [])
                    next_persons = person_map.get(next_s, [])
                    next_dict = {p['track_id']: p for p in next_persons}
                    
                    # Stabilized boxes: only interpolate tracks present in BOTH endpoints.
                    # A track missing in `next` is dropped (not held at its raw box) — this
                    # removes the "frozen then jump" artifact. A prev→next displacement larger
                    # than 1.5×box-diagonal is treated as a track-id swap, so we draw the prev
                    # box without interpolating into a far-away one.
                    interp_persons = []
                    for p in prev_persons:
                        tid = p['track_id']
                        if tid not in next_dict:
                            continue
                        np_p = next_dict[tid]
                        pcx = (p['bbox'][0] + p['bbox'][2]) / 2.0
                        pcy = (p['bbox'][1] + p['bbox'][3]) / 2.0
                        ncx = (np_p['bbox'][0] + np_p['bbox'][2]) / 2.0
                        ncy = (np_p['bbox'][1] + np_p['bbox'][3]) / 2.0
                        box_diag = float(np.hypot(p['bbox'][2] - p['bbox'][0], p['bbox'][3] - p['bbox'][1]))
                        disp = float(np.hypot(ncx - pcx, ncy - pcy))
                        if box_diag > 0 and disp > 1.5 * box_diag:
                            bx1, by1, bx2, by2 = p['bbox']  # likely track swap → no interpolation
                        else:
                            bx1 = p['bbox'][0] + fraction * (np_p['bbox'][0] - p['bbox'][0])
                            by1 = p['bbox'][1] + fraction * (np_p['bbox'][1] - p['bbox'][1])
                            bx2 = p['bbox'][2] + fraction * (np_p['bbox'][2] - p['bbox'][2])
                            by2 = p['bbox'][3] + fraction * (np_p['bbox'][3] - p['bbox'][3])
                        interp_persons.append({
                            'track_id': tid,
                            'bbox': [bx1, by1, bx2, by2],
                            'confidence': p['confidence']
                        })

                    prev_t = prev_s / orig_fps
                    current_events = [e for e in events_list if abs(e['timestamp'] - prev_t) <= 0.5]
                    
                    for e in current_events:
                        evt_id = f"{e['type']}_{e['timestamp']}"
                        if evt_id not in last_seen_events:
                            last_seen_events.add(evt_id)
                            log_msg = f"[{orig_frame_idx}] - {e.get('severity', 'WARN')}: {e['type'].replace('_', ' ').title()}"
                            recent_logs.append(log_msg)
                            
                    det_list = []
                    for p in interp_persons:
                        state = 'normal'
                        p_bbox = p['bbox']
                        for e in current_events:
                            e_sev = e.get('severity', 'LOW')
                            e_bbox = e.get('bbox')
                            is_match = False
                            if e_bbox:
                                ixA = max(p_bbox[0], e_bbox[0])
                                iyA = max(p_bbox[1], e_bbox[1])
                                ixB = min(p_bbox[2], e_bbox[2])
                                iyB = min(p_bbox[3], e_bbox[3])
                                if max(0, ixB - ixA) * max(0, iyB - iyA) > 0:
                                    is_match = True
                                    
                            if is_match or (e['type'] in ['lying_on_floor', 'bent_over', 'aggressive_guard']):
                                if e_sev in ['CRITICAL', 'HIGH']:
                                    state = 'critical'
                                elif e_sev in ['MEDIUM', 'LOW']:
                                    if state != 'critical': state = 'caution'
                                    
                        det_list.append({
                            'bbox': p['bbox'],
                            'identity': f"ID {p['track_id']}",
                            'confidence': p['confidence'],
                            'state': state
                        })
                        
                    annotated_frame = frame.copy()
                    annotated_frame = self.annotator.draw_bboxes(annotated_frame, det_list, label_key='identity', color_key='person', frame_idx=orig_frame_idx)
                    # Pose skeletons (confidence-gated) replace the old chaotic centroid trails.
                    # Held across interpolated frames from the shared pose cache (same pattern
                    # as faces/fire). Cache stores raw pixel-space keypoints for ALL persons.
                    if pose_cache is not None and prev_s in pose_cache:
                        annotated_frame = self.annotator.draw_skeletons(annotated_frame, pose_cache[prev_s])

                    # Resolution-adaptive metrics for the inline overlay labels below.
                    ui_font, ui_t, _ = ui_scale(h)
                    lbl_font = max(0.45, ui_font * 0.6)

                    if prev_s in faces_per_frame_map:
                        for f in faces_per_frame_map[prev_s]:
                            fx1, fy1, fx2, fy2 = [int(v) for v in f['bbox']]
                            cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), ui_t)
                            emo = next((e['emotion'] for e in emotion_results if e.get('track_id') == f.get('track_id') and abs(e.get('timestamp', 0) - t) < 0.5), f.get('identity', 'Face'))
                            cv2.putText(annotated_frame, emo, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, lbl_font, (0, 255, 0), ui_t, cv2.LINE_AA)

                    if prev_s in fire_per_frame_map:
                        for f in fire_per_frame_map[prev_s]:
                            fx1, fy1, fx2, fy2 = [int(v) for v in f['bbox']]
                            overlay = annotated_frame.copy()
                            cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (0, 100, 255), -1)
                            cv2.addWeighted(overlay, 0.18, annotated_frame, 0.82, 0, annotated_frame)
                            cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (0, 165, 255), ui_t + 1)

                            det_conf  = f.get('confidence', 0.0)
                            det_type  = 'Smoke' if f.get('type', '').lower() == 'smoke' else 'Fire'
                            label = f"{det_type} {det_conf:.0%}"
                            cv2.putText(annotated_frame, label, (fx1, max(fy1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, lbl_font, (0, 165, 255), ui_t, cv2.LINE_AA)

                    if prev_s in weapon_per_frame_map:
                        for d in weapon_per_frame_map[prev_s]:
                            wx1, wy1, wx2, wy2 = [int(v) for v in d['bbox']]
                            raised = d.get('raised', False)
                            box_t = (ui_t + 2) if raised else (ui_t + 1)
                            cv2.rectangle(annotated_frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), box_t)
                            wlabel = d.get('weapon_class', 'weapon').title()
                            if raised:
                                wlabel += " RAISED"
                            label = f"{wlabel} {d.get('confidence', 0.0):.0%}"
                            cv2.putText(annotated_frame, label, (wx1, max(wy1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, lbl_font, (0, 0, 255), ui_t, cv2.LINE_AA)

                    hud_metrics = {
                        "Pipeline FPS": f"{orig_fps:.1f}",
                        "Inference Rate": f"{self.config['sampling'].get('base_fps', 5.0)} FPS",
                        "Occupants": str(len(interp_persons)),
                        "Active Threats": str(len(current_events))
                    }
                    annotated_frame = self.annotator.draw_hud(annotated_frame, hud_metrics)
                    annotated_frame = self.annotator.draw_event_ticker(annotated_frame, recent_logs)
                    
                    out_writer.write(annotated_frame)
                    orig_frame_idx += 1

                cap.release()
                out_writer.release()

                if needs_transcode:
                    self._transcode_to_h264(write_path, out_video_path)

                self.log.info(f"Smooth annotated video saved to {out_video_path}")
            except Exception as e:
                self.log.error(f"Annotated video creation failed: {e}")

        total_time = time.time() - pipeline_start
        self.log.info(f"Pipeline finished in {total_time:.1f}s. All outputs in {run_dir}")

        
        return {
            'json_path': json_path,
            'html_path': html_path,
            'srt_path': srt_path,
            'pdf_path': pdf_path,
            'heatmap_path': heatmap_path,
            'trajectory_path': trajectory_path,
            'run_dir': str(run_dir),
            'run_dir_name': run_dir_name,
            'log_path': getattr(self, '_run_log_path', None),
            'processing_time_seconds': round(total_time, 1),
        }
