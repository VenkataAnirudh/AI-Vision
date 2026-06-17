class ThreatScorer:
    def __init__(self, config):
        self.weights = config['threat']
        self.levels = self.weights['levels']

    def calculate_score(self, events_in_frame, emotion_result):
        score = 0.0
        
        esc = self.weights.get('audio_escalation', {})

        # Base multiplier from distinct events
        for evt in events_in_frame:
            evt_type = evt.get('type', '')
            if 'fire' in evt_type or 'smoke' in evt_type:
                score = max(score, self.weights['fire_weight'])
            elif 'weapon' in evt_type:
                score = max(score, self.weights['weapon_weight'])
            elif evt_type == 'violence':
                score = max(score, self.weights['violence_weight'])
            elif evt_type == 'aggressive_guard':
                score = max(score, self.weights['fight_weight'])
            elif evt_type in ('falling_down', 'lying_on_floor'):
                score = max(score, self.weights['fall_weight'])
            # Audio-driven RED escalation (burst of shouts, or shout fused with crowd/violence).
            elif evt_type == 'loud_shout_panic':
                score = max(score, esc.get('red_weight', 0.90))
            elif evt_type in ('loud_shout_impact', 'raised_voice'):
                score = max(score, self.weights.get('shout_weight', 0.45))
            elif evt_type == 'audio_cry':
                score = max(score, self.weights.get('cry_weight', 0.35))
            elif evt_type == 'overcrowding':
                score = max(score, self.weights.get('overcrowd_weight', 0.50))
                
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