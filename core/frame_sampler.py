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
        
        # Temporal clip parameters
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
        
        # We pre-read frames if we need to extract a clip
        # R3D-18 clip requires 16 frames. 16 frames at 12 FPS is ~1.3 seconds.
        # Spacing: native_fps / clip_fps. E.g., 30 / 12 = 2.5 frames.
        frame_spacing = max(1, int(round(native_fps / self.clip_fps)))

        # Cache frames to allow backward/forward extraction for clips
        # We can buffer the last few frames to construct a clip or read forward.
        # Since we are reading sequentially, we can read ahead when a trigger occurs.
        # We keep a sliding window of recent BGR frames
        frame_buffer = []
        max_buffer_size = int(native_fps * 2) # 2 seconds buffer

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / native_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (160, 120)) # Resize for fast diff computation

            # Keep sliding BGR frame buffer
            frame_buffer.append((timestamp, frame.copy()))
            if len(frame_buffer) > max_buffer_size:
                frame_buffer.pop(0)

            # Calculate motion score
            if prev_gray is not None:
                diff = cv2.absdiff(gray_resized, prev_gray)
                motion_score = float(diff.mean())
            else:
                motion_score = 0.0

            prev_gray = gray_resized

            # Determine adaptive target FPS
            if motion_score < self.static_threshold:
                target_fps = self.static_fps
            elif motion_score > self.high_motion_threshold:
                target_fps = self.high_motion_fps
            else:
                target_fps = self.base_fps

            # Check if we should sample this frame
            sample_interval = 1.0 / target_fps
            if timestamp - last_sampled_time >= sample_interval:
                sampled_frames.append({
                    'frame': frame.copy(),
                    'timestamp': timestamp,
                    'frame_idx': frame_idx,
                    'motion_score': motion_score
                })
                last_sampled_time = timestamp

            # Check if we should trigger an action clip (R3D-18)
            if motion_score >= self.motion_trigger_threshold:
                if timestamp - last_clip_time >= self.min_clip_gap:
                    # Extract 16 frames. We can look forward or look backward/forward.
                    # Let's read 16 frames from cap or buffer.
                    # Spacing is frame_spacing (e.g. 2). 16 frames at spacing 2 is 32 native frames.
                    # Let's collect 16 frames starting from the buffer or read ahead.
                    clip_frames = []
                    
                    # Store current position to restore it
                    curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    
                    # Read 16 frames spaced by frame_spacing
                    clip_success = True
                    clip_idx = frame_idx
                    for _ in range(16):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_idx)
                        r_ret, r_frame = cap.read()
                        if r_ret:
                            clip_frames.append(r_frame.copy())
                        else:
                            clip_success = False
                            break
                        clip_idx += frame_spacing
                    
                    # Restore original capture position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_pos)
                    
                    if clip_success and len(clip_frames) == 16:
                        motion_clips.append({
                            'timestamp': timestamp,
                            'frames': clip_frames
                        })
                        last_clip_time = timestamp
                        print(f"[FrameSampler] Motion Clip triggered at {timestamp:.2f}s (motion_score: {motion_score:.2f})")

            frame_idx += 1

        cap.release()
        print(f"[FrameSampler] Adaptive extraction complete. Sampled {len(sampled_frames)} frames, extracted {len(motion_clips)} motion clips.")
        return sampled_frames, motion_clips
