import cv2
import numpy as np

class FrameSampler:
    def __init__(self, config):
        self.config = config['sampling']
        self.base_fps = self.config['base_fps']
        self.static_fps = self.config['static_fps']
        self.high_motion_fps = self.config['high_motion_fps']
        self.static_threshold = self.config['static_threshold']
        self.high_motion_threshold = self.config['high_motion_threshold']
        
        
        self.clip_fps = self.config['temporal']['clip_fps']
        self.clip_duration = self.config['temporal']['clip_duration_seconds']
        self.motion_trigger_threshold = self.config['temporal']['motion_trigger_threshold']
        self.min_clip_gap = self.config['temporal']['min_clip_gap_seconds']

    def extract(self, video_path):
        """
        Processes the video, performs adaptive frame sampling, and extracts
        16-frame action clips when motion exceeds the trigger threshold.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Could not open video: {video_path}")
            return [], []

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0:
            native_fps = 30.0

        sampled_frames = []
        motion_clips = []

        prev_gray = None
        last_sampled_time = -999.0
        last_clip_time = -999.0
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        
        
        
        frame_spacing = max(1, int(round(native_fps / self.clip_fps)))

        
        
        
        
        from collections import deque
        max_buffer_size = max(100, int(native_fps * 3)) 
        frame_buffer = deque(maxlen=max_buffer_size)

        pending_clip_trigger = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / native_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (160, 120)) 

            
            frame_buffer.append((timestamp, frame.copy()))

            
            if prev_gray is not None:
                diff = cv2.absdiff(gray_resized, prev_gray)
                
                k = max(1, int(diff.size * 0.05))
                motion_score = float(np.partition(diff.flatten(), -k)[-k:].mean())
            else:
                motion_score = 0.0

            prev_gray = gray_resized

            
            if motion_score < self.static_threshold:
                target_fps = max(3.0, self.static_fps) 
            elif motion_score > self.high_motion_threshold:
                target_fps = self.high_motion_fps
            else:
                target_fps = max(3.0, self.base_fps)

            
            sample_interval = 1.0 / target_fps
            if timestamp - last_sampled_time >= sample_interval:
                sampled_frames.append({
                    'frame': frame.copy(),
                    'timestamp': timestamp,
                    'frame_idx': frame_idx,
                    'motion_score': motion_score
                })
                last_sampled_time = timestamp

            
            if motion_score >= self.motion_trigger_threshold:
                if timestamp - last_clip_time >= self.min_clip_gap:
                    if pending_clip_trigger is None:
                        pending_clip_trigger = timestamp
                        last_clip_time = timestamp
                        print(f"[FrameSampler] Motion Clip triggered at {timestamp:.2f}s (motion_score: {motion_score:.2f}) - buffering...")

            
            if pending_clip_trigger is not None:
                frames_needed_after = 8 * frame_spacing
                time_needed_after = frames_needed_after / native_fps
                
                if timestamp >= pending_clip_trigger + time_needed_after:
                    total_span = 16 * frame_spacing
                    if len(frame_buffer) >= total_span:
                        clip_frames = [f for (_, f) in list(frame_buffer)[-total_span::frame_spacing]][:16]
                        if len(clip_frames) == 16:
                            motion_clips.append({
                                'timestamp': pending_clip_trigger,
                                'frames': clip_frames
                            })
                            print(f"[FrameSampler] Motion Clip extracted around {pending_clip_trigger:.2f}s")
                    pending_clip_trigger = None

            frame_idx += 1

        
        if pending_clip_trigger is not None:
            total_span = 16 * frame_spacing
            available_frames = [f for (_, f) in list(frame_buffer)[-total_span::frame_spacing]]
            while len(available_frames) < 16 and len(available_frames) > 0:
                available_frames.append(available_frames[-1])
            if len(available_frames) == 16:
                motion_clips.append({
                    'timestamp': pending_clip_trigger,
                    'frames': available_frames
                })

        cap.release()
        print(f"[FrameSampler] Adaptive extraction complete. Sampled {len(sampled_frames)} frames, extracted {len(motion_clips)} motion clips.")
        return sampled_frames, motion_clips
