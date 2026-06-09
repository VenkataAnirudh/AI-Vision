class ThreatScorer:
    def __init__(self, config):
        self.weights = config['threat']
        self.levels = self.weights['levels']

    def calculate_score(self, events_in_frame, emotion_result):
        score = 0.0
        
        # Base multiplier from distinct events
        for evt in events_in_frame:
            evt_type = evt.get('type', '')
            if evt_type == 'fire/smoke':
                score = max(score, self.weights['fire_weight'])
            elif evt_type == 'violence':
                score = max(score, self.weights['violence_weight'])
            elif evt_type in ('falling_down', 'lying_on_floor'):
                score = max(score, self.weights['fall_weight'])
                
        # Factor in visual/audio heuristics
        if emotion_result:
            cry_val = emotion_result.get('fused_cry_prob', emotion_result.get('visual_cry_prob', 0.0))
            stress_val = emotion_result.get('visual_stress_score', 0.0)
            
            score = max(score, cry_val * self.weights['cry_weight'])
            score = max(score, stress_val * self.weights['stress_weight'])
            
        return score, self._get_level(score)

    def _get_level(self, score):
        if score <= self.levels['green'][1]: return 'GREEN'
        if score <= self.levels['yellow'][1]: return 'YELLOW'
        if score <= self.levels['orange'][1]: return 'ORANGE'
        return 'RED'