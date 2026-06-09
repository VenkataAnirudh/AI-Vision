class AudioVisualFuser:
    def __init__(self, config):
        # Based on config: threat -> cry_weight / stress_weight
        self.audio_cry_weight = 0.65
        self.visual_cry_weight = 0.35
        self.fusion_mode = "audio_visual"

    def fuse(self, emotion_results, audio_results, time_tolerance_seconds=1.5):
        """
        Combines visual probability with overlapping audio probability.
        Audio is weighted heavier because EAR (visual) is prone to false positives from blinking.
        """
        if not audio_results or not emotion_results:
            self.fusion_mode = "visual_only"
            return emotion_results
            
        fused_timeline = []
        audio_cries = audio_results.get('cry_segments', [])
        
        # This assumes emotion_results is a timeline of per-frame detections
        # Format mapping needs to align with your pipeline loop aggregation
        for visual_entry in emotion_results:
            v_ts = visual_entry['timestamp']
            v_cry = visual_entry['visual_cry_prob']
            
            # Find closest audio cry event
            matching_audio = [a for a in audio_cries if abs(a['timestamp'] - v_ts) <= time_tolerance_seconds]
            
            if matching_audio:
                a_cry = max([a['probability'] for a in matching_audio])
                fused_prob = (self.audio_cry_weight * a_cry) + (self.visual_cry_weight * v_cry)
            else:
                # If no audio supports it, dampen the visual signal
                fused_prob = v_cry * 0.5 
                
            visual_entry['fused_cry_prob'] = fused_prob
            visual_entry['fusion_mode'] = self.fusion_mode
            fused_timeline.append(visual_entry)
            
        return fused_timeline