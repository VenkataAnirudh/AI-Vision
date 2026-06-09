import os
import cv2
import yaml
import json
import torch
import numpy as np
from pathlib import Path

# Core and Stages Imports
from core.model_manager import ModelManager
from core.frame_sampler import FrameSampler
from stages.person_detection import PersonDetector
from stages.face_pipeline import FacePipeline
from stages.emotion_analysis import EmotionAnalyzer
from stages.event_detection import EventDetector
from stages.indoor_action import IndoorActionDetector
from stages.audio_analysis import AudioAnalyzer
from stages.video_description import VideoDescriber

# Feature and Utility Imports
from fusion.av_fusion import AudioVisualFuser
from features.proximity_analyzer import ProximityAnalyzer
from features.anonymizer import FaceAnonymizer
from features.threat_scorer import ThreatScorer
from features.chapter_generator import ChapterGenerator
from features.clip_extractor import ClipExtractor
from utils.reporter import ReportGenerator
from utils.metrics import TemporalSmoother
from utils.audio_utils import check_audio_track
from utils.drawing import Annotator

class VideoPipeline:
    def __init__(self, stages=None):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        # Force CUDA if config specifies and device is available
        device_name = self.config['hardware']['device']
        fp16_enabled = self.config['hardware']['fp16']
        self.manager = ModelManager(device=device_name, fp16=fp16_enabled)

        # Initialize samplers and features
        self.frame_sampler = FrameSampler(self.config)
        self.smoother = TemporalSmoother(window_size=5)
        self.proximity_analyzer = ProximityAnalyzer()
        self.anonymizer = FaceAnonymizer()
        self.threat_scorer = ThreatScorer(self.config)
        self.chapter_generator = ChapterGenerator(self.config)
        self.clip_extractor = ClipExtractor(self.config)
        self.reporter = ReportGenerator(self.config)
        self.annotator = Annotator()

        self.stages_to_run = stages if stages else ['all']

    def _should_run(self, stage_name):
        if 'all' in self.stages_to_run:
            return self.config['stages'].get(stage_name, True)
        return stage_name in self.stages_to_run

    def _deduplicate_events(self, events_list, time_window=2.0):
        """
        Merges duplicate events of the same type within a time window.
        Keeps only the highest-confidence instance per type per window.
        """
        if not events_list:
            return []

        # Sort by timestamp
        sorted_events = sorted(events_list, key=lambda e: (e.get('type', ''), e.get('timestamp', 0)))
        deduped = []
        
        for event in sorted_events:
            merged = False
            for existing in deduped:
                # Same type and within time window → keep the higher confidence one
                if (existing['type'] == event['type'] and
                    abs(existing['timestamp'] - event['timestamp']) <= time_window):
                    if event.get('confidence', 0) > existing.get('confidence', 0):
                        existing.update(event)
                    merged = True
                    break
            if not merged:
                deduped.append(event.copy())
        
        return deduped

    def _validate_fire_detection(self, fire_events, frame, persons_in_frame):
        """
        Cross-validates fire/smoke detections against person detections
        and frame geometry to suppress false positives.
        
        Common false positives: warm-colored clothing, skin tones, 
        motion blur during falls, indoor lighting.
        """
        validated = []
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        
        for fe in fire_events:
            bbox = fe.get('bbox', [0, 0, 0, 0])
            fx1, fy1, fx2, fy2 = bbox
            det_area = max(1, (fx2 - fx1) * (fy2 - fy1))
            
            # FILTER 1: Reject detections covering >40% of the frame
            # Real fire/smoke in surveillance is localized, not full-frame
            if det_area / frame_area > 0.40:
                print(f"[FireFilter] Rejected: bbox covers {det_area/frame_area:.0%} of frame (too large)")
                continue
            
            # FILTER 2: Reject detections with very small area (noise)
            if det_area < 400:  # Less than ~20x20 pixels
                continue
            
            # FILTER 3: Cross-check against person bounding boxes
            # If fire bbox heavily overlaps a detected person, it's likely
            # a false positive (warm clothing, skin tone, motion blur)
            is_person_overlap = False
            for person in persons_in_frame:
                px1, py1, px2, py2 = person['bbox']
                # Calculate IoU between fire and person bbox
                ix1 = max(fx1, px1)
                iy1 = max(fy1, py1)
                ix2 = min(fx2, px2)
                iy2 = min(fy2, py2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                
                person_area = max(1, (px2 - px1) * (py2 - py1))
                # If >50% of the fire bbox overlaps a person → false positive
                overlap_ratio = inter / det_area
                if overlap_ratio > 0.50:
                    is_person_overlap = True
                    print(f"[FireFilter] Rejected: fire bbox overlaps person (IoU={overlap_ratio:.2f})")
                    break
            
            if is_person_overlap:
                continue
            
            # FILTER 4: Require higher effective confidence (0.55+)
            if fe.get('confidence', 0) < 0.55:
                continue
            
            validated.append(fe)
        
        return validated

    def process_video(self, video_path, progress_callback=None):
        print(f"\n--- Starting Full Video Pipeline for: {Path(video_path).name} ---")
        if progress_callback: progress_callback("Initializing Pipeline")
        
        # Open video to query base properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frame_count / fps
        cap.release()

        # Step 1: check audio track presence
        if progress_callback: progress_callback("Checking Audio Streams")
        has_audio = check_audio_track(video_path)
        print(f"[Audio Check] {'Audio stream detected' if has_audio else 'No audio stream found. Running in video-only mode.'}")

        # Step 2: Stage 0 - Adaptive Frame Sampling
        if progress_callback: progress_callback("Stage 0: Adaptive Frame Sampling")
        sampled_frames, motion_clips = self.frame_sampler.extract(video_path)
        if not sampled_frames:
            print("[Error] No frames sampled from video.")
            return None

        # Data structure to accumulate results
        aggregated_results = {
            'total_frames': total_frame_count,
            'frames_sampled': len(sampled_frames),
            'fusion_mode': 'audio_visual' if (has_audio and self._should_run('audio_analysis')) else 'visual_only',
            'unique_count': 0,
            'individuals': [],
            'events': [],
            'chapters': [],
            'threat_timeline': [],
            'description': ""
        }

        # Step 3: Stage 1 - Person Detection + Tracking
        person_results = {'unique_count': 0, 'per_frame': []}
        if self._should_run('person_detection'):
            print("\n--- Running Stage 1: Person Detection & Tracking ---")
            if progress_callback: progress_callback("Stage 1: Person Tracking (YOLOv8n + ByteTrack)")
            detector = PersonDetector(self.manager, self.config)
            person_results = detector.process_frames(sampled_frames)
            aggregated_results['unique_count'] = person_results['unique_count']
            self.manager.unload('yolo_person')

        # Map tracked IDs to identity info
        track_identity_map = {} # track_id -> face_id
        track_face_conf_map = {}
        
        # Step 4: Stage 2 & 3 - Face Identification & Emotion Analysis
        face_detections_timeline = []
        emotion_results = []
        
        if self._should_run('face_pipeline') or self._should_run('emotion_analysis'):
            print("\n--- Running Stage 2 & 3: Face & Emotion Analysis ---")
            if progress_callback: progress_callback("Stage 2 & 3: Recognizing Faces & Emotion Analysis")
            
            face_pipeline = None
            if self._should_run('face_pipeline'):
                face_pipeline = FacePipeline(self.config)
                
            emotion_analyzer = None
            if self._should_run('emotion_analysis'):
                emotion_analyzer = EmotionAnalyzer(self.config)

            # We process face pipeline on sampled frames
            # Subsampling optimization: process face and emotion every 2nd sampled frame
            for idx, frame_data in enumerate(sampled_frames):
                if idx % 2 != 0:
                    continue
                
                frame = frame_data['frame']
                ts = frame_data['timestamp']
                
                # Get tracked persons for this frame
                persons_in_frame = []
                for pf in person_results['per_frame']:
                    if pf['frame_idx'] == frame_data['frame_idx']:
                        persons_in_frame = pf['persons']
                        break

                if not persons_in_frame:
                    continue

                # Run Face Detection & Recognition
                faces_in_frame = []
                if face_pipeline:
                    faces_in_frame = face_pipeline.process(frame, [p['bbox'] for p in persons_in_frame])
                    
                # Correlate faces to person tracks via bounding box overlap
                for face in faces_in_frame:
                    fx1, fy1, fx2, fy2 = face['bbox']
                    best_track_id = None
                    best_overlap = 0.0
                    
                    # Compute overlap (simple intersection area)
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
                        
                        # Run Emotion Analyzer
                        if emotion_analyzer:
                            # Create a dummy container for landmarks
                            face_obj = type('', (), {})()
                            face_obj.bbox = face['bbox']
                            
                            # Retrieve the full detected face object containing landmarks
                            # The face_pipeline process returns simplified dicts, but we can access 
                            # the original landmarks if we match it from face_pipeline app output
                            # For safety, face_pipeline app.get() returns raw faces. Let's find the match.
                            raw_faces = face_pipeline.app.get(frame)
                            matched_raw_face = None
                            for rf in raw_faces:
                                rx1, ry1, rx2, ry2 = rf.bbox
                                if abs(rx1 - fx1) < 5 and abs(ry1 - fy1) < 5:
                                    matched_raw_face = rf
                                    break
                            
                            if matched_raw_face:
                                emo_data = emotion_analyzer.analyze_face(frame, matched_raw_face)
                                
                                # Smooth scores temporally
                                smoothed_cry = self.smoother.smooth(best_track_id, 'cry', emo_data['visual_cry_prob'])
                                smoothed_stress = self.smoother.smooth(best_track_id, 'stress', emo_data['visual_stress_score'])
                                
                                emotion_results.append({
                                    'track_id': best_track_id,
                                    'timestamp': ts,
                                    'emotion': emo_data['dominant_emotion'],
                                    'emotion_scores': emo_data['emotion_scores'],
                                    'visual_cry_prob': smoothed_cry,
                                    'visual_stress_score': smoothed_stress
                                })

            # Purge Face Analysis models from VRAM
            if face_pipeline:
                self.manager.unload('buffalo_l')

        # Populate individual target info
        for tid in track_identity_map.keys():
            aggregated_results['individuals'].append({
                'track_id': tid,
                'face_id': track_identity_map[tid],
                'face_confidence': track_face_conf_map.get(tid)
            })

        # Step 5: Stage 4 - Event Detection (Fire/Smoke & Violence) & Indoor Action (Falls)
        events_list = []
        
        if self._should_run('event_detection') or self._should_run('indoor_action'):
            print("\n--- Running Stage 4: Event & Action Detection ---")
            if progress_callback: progress_callback("Stage 4: Running Incident & Fall Classifiers (R3D-18)")
            
            event_detector = None
            if self._should_run('event_detection'):
                event_detector = EventDetector(self.manager, self.config)
                
            action_detector = None
            if self._should_run('indoor_action'):
                action_detector = IndoorActionDetector(self.manager, self.config)

            # 4a. Fire & Smoke: Run at 3 FPS on sampled frames, with cross-validation
            if event_detector:
                for idx, frame_data in enumerate(sampled_frames):
                    if idx % max(1, int(round(fps / self.config['sampling']['stage_fps']['fire_detection']))) == 0:
                        raw_fire_events = event_detector.detect_fire(frame_data['frame'])
                        
                        # Get persons detected in this frame for cross-validation
                        persons_in_this_frame = []
                        for pf in person_results['per_frame']:
                            if pf['frame_idx'] == frame_data['frame_idx']:
                                persons_in_this_frame = pf['persons']
                                break
                        
                        # Cross-validate fire detections against person bboxes
                        validated_fire = self._validate_fire_detection(
                            raw_fire_events, frame_data['frame'], persons_in_this_frame
                        )
                        for fe in validated_fire:
                            fe['timestamp'] = frame_data['timestamp']
                            events_list.append(fe)

            # 4b. Violence and Indoor Action: Process on motion_clips
            # Load both models together to optimize VRAM session
            if event_detector and motion_clips:
                event_detector._get_violence_model()
            if action_detector and motion_clips:
                action_detector._load_model()
                
            for clip in motion_clips:
                ts = clip['timestamp']
                
                # Check for Violence
                if event_detector:
                    # Construct clip tensor: shape [1, 3, 16, 112, 112]
                    processed_frames = []
                    for f in clip['frames']:
                        resized = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (112, 112))
                        # Scale to 0-1 and normalize
                        normalized = (resized.astype(np.float32) / 255.0 - np.array([0.43216, 0.394666, 0.37645])) / np.array([0.22803, 0.22145, 0.216989])
                        processed_frames.append(normalized)
                    clip_tensor = torch.from_numpy(np.stack(processed_frames)).permute(3, 0, 1, 2).unsqueeze(0)
                    
                    v_event = event_detector.detect_violence(clip_tensor)
                    if v_event:
                        v_event['timestamp'] = ts
                        events_list.append(v_event)

                # Check for Falls / Action Alerts
                if action_detector:
                    act_event = action_detector.process_clip(clip['frames'])
                    if act_event:
                        act_event['timestamp'] = ts
                        act_event['type'] = act_event['action']
                        act_event['severity'] = "HIGH" if act_event['is_alert'] else "LOW"
                        events_list.append(act_event)

            # Clean R3D-18 models from memory
            self.manager.unload('violence_r3d')
            self.manager.unload('indoor_action_r3d')
            self.manager.unload('fire_yolo')

        # Step 6: Stage 5 - Audio Analysis (Conditional)
        audio_results = None
        if has_audio and self._should_run('audio_analysis'):
            print("\n--- Running Stage 5: Conditional Audio Analysis ---")
            if progress_callback: progress_callback("Stage 5: Conditional Audio Analysis (Librosa)")
            audio_analyzer = AudioAnalyzer(self.config)
            audio_results = audio_analyzer.process(video_path)
            
            # Append audio events to the main event log
            if audio_results:
                for cry in audio_results.get('cry_segments', []):
                    events_list.append({
                        'type': 'audio_cry',
                        'timestamp': cry['timestamp'],
                        'confidence': cry['probability'],
                        'severity': 'MEDIUM'
                    })
                for loud in audio_results.get('loud_events', []):
                    events_list.append({
                        'type': 'loud_shout_impact',
                        'timestamp': loud['timestamp'],
                        'confidence': 0.85,
                        'severity': 'HIGH'
                    })

        # Step 7: Audio-Visual Fusion
        fused_emotions = emotion_results
        if has_audio and audio_results and self._should_run('audio_analysis'):
            print("\n--- Running Audio-Visual Fusion ---")
            fuser = AudioVisualFuser(self.config)
            fused_emotions = fuser.fuse(emotion_results, audio_results)

        # Map fused emotions/cries into events
        for fe in fused_emotions:
            if fe.get('fused_cry_prob', 0.0) >= 0.60:
                events_list.append({
                    'type': 'person_cry',
                    'timestamp': fe['timestamp'],
                    'confidence': fe['fused_cry_prob'],
                    'severity': 'MEDIUM',
                    'track_id': fe['track_id']
                })

        # Proximity analysis (frame-by-frame) with pair deduplication
        seen_proximity_pairs = {}  # (pair_key, window_ts) -> True
        for idx, frame_data in enumerate(sampled_frames):
            persons_in_frame = []
            for pf in person_results['per_frame']:
                if pf['frame_idx'] == frame_data['frame_idx']:
                    persons_in_frame = pf['persons']
                    break
            
            prox_events = self.proximity_analyzer.analyze(persons_in_frame)
            for pe in prox_events:
                # Deduplicate: only emit one proximity alert per unique pair per 2-second window
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
                    'details': f"Track IDs {pe['track_ids']} too close"
                })

        # Deduplicate events of the same type within close temporal proximity
        events_list = self._deduplicate_events(events_list, time_window=2.0)
        print(f"[Pipeline] {len(events_list)} events after deduplication")
        aggregated_results['events'] = events_list

        # Step 8: Stage 6 - Natural Language Video Description (VLM)
        description = "No narrative summary generated."
        if self._should_run('video_description'):
            print("\n--- Running Stage 6: Video Narrative Description ---")
            if progress_callback: progress_callback("Stage 6: Querying Gemini VLM Narrative Engine")
            
            # Select 5 keyframes
            keyframe_indices = []
            
            # 1. First frame
            keyframe_indices.append(0)
            
            # 2. Last frame
            if len(sampled_frames) > 1:
                keyframe_indices.append(len(sampled_frames) - 1)
                
            # 3. Highest motion score frame
            motion_scores = [f['motion_score'] for f in sampled_frames]
            if motion_scores:
                keyframe_indices.append(int(np.argmax(motion_scores)))
                
            # 4. Frame nearest to highest threat timestamp (approx center if no events)
            if events_list:
                highest_conf_event = max(events_list, key=lambda x: x.get('confidence', 0.0))
                evt_ts = highest_conf_event.get('timestamp', duration / 2.0)
                diffs = [abs(f['timestamp'] - evt_ts) for f in sampled_frames]
                keyframe_indices.append(int(np.argmin(diffs)))
            else:
                keyframe_indices.append(len(sampled_frames) // 2)

            # 5. Frame with most detected persons
            person_counts = [len(pf['persons']) for pf in person_results['per_frame']]
            if person_counts:
                keyframe_indices.append(int(np.argmax(person_counts)))

            # Unique keyframe indices
            unique_indices = sorted(list(set(keyframe_indices)))
            
            describer = VideoDescriber(self.manager, self.config)
            
            frame_descriptions = []
            for k_idx in unique_indices:
                if k_idx < len(sampled_frames):
                    frame_data = sampled_frames[k_idx]
                    rgb_frame = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)
                    desc = describer.describe_keyframe(
                        rgb_frame, 
                        prompt="Describe the actions, location, and people visible in this scene frame."
                    )
                    frame_descriptions.append(f"At {frame_data['timestamp']:.1f}s: {desc}")
            
            # Synthesize final narration paragraph
            event_summary = ", ".join([f"{e['type']} at {e['timestamp']:.1f}s" for e in events_list[:5]])
            description = describer.synthesize_summary(frame_descriptions, event_summary)
            aggregated_results['description'] = description

            # Clean local VLM if loaded
            if describer.mode == 'local':
                self.manager.unload('moondream')

        # Step 9: Threat Timeline Calculation
        print("[ThreatScorer] Computing per-second threat index timeline...")
        timeline = []
        for sec in range(int(np.ceil(duration))):
            ts = float(sec)
            # Find events occurring in the window (sec - 1 to sec + 1)
            events_in_window = [e for e in events_list if abs(e['timestamp'] - ts) <= 1.0]
            
            # Find emotions in this window
            emotions_in_window = [e for e in fused_emotions if abs(e['timestamp'] - ts) <= 1.0]
            emo_res = emotions_in_window[0] if emotions_in_window else None
            
            score, level = self.threat_scorer.calculate_score(events_in_window, emo_res)
            timeline.append({
                'ts': ts,
                'score': score,
                'level': level
            })
        aggregated_results['threat_timeline'] = timeline

        # Step 10: Chapter Segmentation
        print("[ChapterGenerator] Generating video chapter divisions...")
        # Simplistic scene changes fallback (can be computed by high motion thresholds)
        scene_changes = [f['timestamp'] for f in sampled_frames if f['motion_score'] >= 30.0]
        # Deduplicate scenes close to each other
        filtered_scenes = []
        last_s = -99.0
        for s in scene_changes:
            if s - last_s >= 5.0:
                filtered_scenes.append(s)
                last_s = s
                
        # Generate chapters
        action_detections = [e for e in events_list if e['type'] in self.config['models']['indoor_action']['classes']]
        chapters = self.chapter_generator.generate(duration, events_list, action_detections)
        aggregated_results['chapters'] = chapters

        # Step 11: Export Reports (JSON, HTML Dashboard, SRT Subtitles)
        if progress_callback: progress_callback("Finalizing: Saving Reports & Annotated Feeds")
        out_dir = Path("output")
        reports_dir = out_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = self.reporter.build_json(Path(video_path).name, duration, aggregated_results)
        self.reporter.build_html(Path(video_path).name, duration, json_path)
        self.reporter.build_srt(Path(video_path).stem, events_list)

        # Step 12: Clip Extraction
        if self.config['clips']['enabled']:
            print("[ClipExtractor] Extracting video evidence clips around incidents...")
            for idx, e in enumerate(events_list):
                if e.get('severity') == "HIGH" or e.get('type') in ['fire/smoke', 'violence', 'falling_down']:
                    self.clip_extractor.extract(video_path, e['type'], e['timestamp'])

        # Step 13: Annotated Video Output
        if self.config['output']['annotated_video']:
            annotated_dir = out_dir / "annotated"
            annotated_dir.mkdir(parents=True, exist_ok=True)
            out_video_path = annotated_dir / f"annotated_{Path(video_path).name}"
            
            h, w, _ = sampled_frames[0]['frame'].shape
            # Write annotated video at 3 FPS (matching our sampling speed)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(str(out_video_path), fourcc, 3.0, (w, h))
            
            for f_data in sampled_frames:
                frame_idx = f_data['frame_idx']
                annotated_frame = f_data['frame'].copy()
                
                # Fetch detections for this frame
                persons_in_frame = []
                for pf in person_results['per_frame']:
                    if pf['frame_idx'] == frame_idx:
                        persons_in_frame = pf['persons']
                        break
                        
                # GDPR Blur faces if enabled
                if self.config['output']['anonymize_faces']:
                    faces_to_blur = []
                    # Simple face bbox extraction using bounding box overlap logic
                    # To be clean, we can just run the anonymizer on face list we generated earlier
                    # But doing it on the fly is simpler:
                    for p in persons_in_frame:
                        # Estimate face region (upper 25% of body box)
                        px1, py1, px2, py2 = p['bbox']
                        face_h = (py2 - py1) * 0.25
                        faces_to_blur.append({'bbox': [px1, py1, px2, py1 + face_h]})
                    annotated_frame = self.anonymizer.anonymize(annotated_frame, faces_to_blur)

                # Draw person bboxes
                annotated_frame = self.annotator.draw_bboxes(
                    annotated_frame, 
                    [{'bbox': p['bbox'], 'identity': f"Track ID {p['track_id']}", 'confidence': p['confidence']} for p in persons_in_frame],
                    label_key='identity',
                    color_key='person'
                )
                
                # Draw high-priority alerts on top-left of the screen
                current_events = [e for e in events_list if abs(e['timestamp'] - f_data['timestamp']) <= 1.0]
                if current_events:
                    alert_text = f"ALERT: {current_events[0]['type'].upper()}"
                    cv2.putText(annotated_frame, alert_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                out_writer.write(annotated_frame)
                
            out_writer.release()
            print(f"[Reporter] Annotated video saved to {out_video_path}")

        print(f"--- Pipeline Finished. Saved reports and clips to output/ ---")
        return json_path
