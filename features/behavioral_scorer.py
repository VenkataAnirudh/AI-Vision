import numpy as np
from collections import defaultdict


class BehavioralScorer:
    """Computes per-person anomaly scores from movement, emotion, and events.

    Sub-scores
    ----------
    * **Movement erraticism** — standard deviation of frame-to-frame
      centroid velocity, normalised to 0-1.
    * **Emotion volatility** — standard deviation of dominant-emotion
      label changes, normalised to 0-1.
    * **Event involvement** — count of events referencing the track,
      normalised to 0-1 (capped at 3).

    The composite anomaly score is a weighted average of the three
    sub-scores, clamped to ``[0, 1]``.

    CPU-only — uses numpy exclusively.
    """

    def __init__(self, config=None):
        """Initialise the behavioural scorer.

        Args:
            config: Optional dict.  Recognised keys (under 'behavioral'):
                - weight_movement (float): Default 0.4.
                - weight_emotion (float): Default 0.3.
                - weight_events  (float): Default 0.3.
                - velocity_cap   (float): Velocity std-dev value that
                  maps to a score of 1.0.  Default 60.0.
        """
        config = config or {}
        beh_cfg = config.get('behavioral', {})
        self.w_movement = float(beh_cfg.get('weight_movement', 0.4))
        self.w_emotion = float(beh_cfg.get('weight_emotion', 0.3))
        self.w_events = float(beh_cfg.get('weight_events', 0.3))
        self.velocity_cap = float(beh_cfg.get('velocity_cap', 60.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, person_results_per_frame, emotion_results=None,
              events_list=None):
        """Compute per-track anomaly scores.

        Args:
            person_results_per_frame: List of per-frame dicts with
                ``'timestamp'`` and ``'persons'`` (each having
                ``'track_id'`` and ``'bbox'``).
            emotion_results: Optional list of dicts with ``'track_id'``,
                ``'timestamp'``, ``'emotion'``, and ``'emotion_scores'``.
            events_list: Optional list of event dicts (``'type'``,
                ``'timestamp'``, etc.).  If an event contains a
                ``'track_id'`` key it is counted for that track.

        Returns:
            List[dict]: Per-track result dicts sorted by
            ``anomaly_score`` descending.  Keys: ``track_id``,
            ``anomaly_score``, ``movement_score``, ``emotion_score``,
            ``event_score``, ``dwell_time_seconds``, ``first_seen``,
            ``last_seen``.
        """
        emotion_results = emotion_results or []
        events_list = events_list or []

        if not person_results_per_frame:
            return []

        # -- 1. Build per-track centroid + timestamp sequences ------------
        track_positions = defaultdict(list)   # tid → [(ts, cx, cy)]
        for frame_entry in person_results_per_frame:
            ts = frame_entry.get('timestamp', 0.0)
            for person in frame_entry.get('persons', []):
                tid = person.get('track_id')
                bbox = person.get('bbox')
                if tid is None or bbox is None or len(bbox) < 4:
                    continue
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                track_positions[tid].append((ts, cx, cy))

        if not track_positions:
            return []

        # -- 2. Index emotions and events by track_id ---------------------
        track_emotions = defaultdict(list)  # tid → [emotion_str, ...]
        for emo in emotion_results:
            tid = emo.get('track_id')
            if tid is not None:
                track_emotions[tid].append(emo.get('emotion', ''))

        track_event_count = defaultdict(int)
        for evt in events_list:
            tid = evt.get('track_id')
            if tid is not None:
                track_event_count[tid] += 1

        # -- 3. Score each track -----------------------------------------
        results = []
        for tid, pos_list in track_positions.items():
            pos_list.sort(key=lambda t: t[0])
            timestamps = [p[0] for p in pos_list]
            first_seen = timestamps[0]
            last_seen = timestamps[-1]
            dwell_time = last_seen - first_seen

            movement_score = self._movement_erraticism(pos_list)
            emotion_score = self._emotion_volatility(track_emotions.get(tid))
            event_score = self._event_involvement(track_event_count.get(tid, 0))

            composite = (self.w_movement * movement_score
                         + self.w_emotion * emotion_score
                         + self.w_events * event_score)
            composite = float(np.clip(composite, 0.0, 1.0))

            results.append({
                'track_id': tid,
                'anomaly_score': round(composite, 4),
                'movement_score': round(movement_score, 4),
                'emotion_score': round(emotion_score, 4),
                'event_score': round(event_score, 4),
                'dwell_time_seconds': round(dwell_time, 2),
                'first_seen': float(first_seen),
                'last_seen': float(last_seen),
            })

        # Sort descending by anomaly score
        results.sort(key=lambda r: r['anomaly_score'], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Internal sub-score helpers
    # ------------------------------------------------------------------

    def _movement_erraticism(self, pos_list):
        """Std-dev of frame-to-frame velocity, normalised to 0-1."""
        if len(pos_list) < 3:
            return 0.0

        velocities = []
        for i in range(1, len(pos_list)):
            ts_prev, cx_prev, cy_prev = pos_list[i - 1]
            ts_curr, cx_curr, cy_curr = pos_list[i]
            dt = ts_curr - ts_prev
            if dt <= 0:
                continue
            dx = cx_curr - cx_prev
            dy = cy_curr - cy_prev
            speed = np.sqrt(dx ** 2 + dy ** 2) / dt
            velocities.append(speed)

        if len(velocities) < 2:
            return 0.0

        std = float(np.std(velocities))
        return float(np.clip(std / self.velocity_cap, 0.0, 1.0))

    def _emotion_volatility(self, emotions):
        """Count of dominant-emotion changes, normalised to 0-1."""
        if not emotions or len(emotions) < 2:
            return 0.0

        changes = sum(1 for i in range(1, len(emotions))
                      if emotions[i] != emotions[i - 1])
        max_possible = len(emotions) - 1
        if max_possible <= 0:
            return 0.0
        return float(np.clip(changes / max_possible, 0.0, 1.0))

    @staticmethod
    def _event_involvement(count):
        """Normalise event count to 0-1, capped at 3."""
        return float(min(count, 3) / 3.0)
