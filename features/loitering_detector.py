import numpy as np
from collections import defaultdict


class LoiteringDetector:
    """Detects loitering — persons staying in one area for an extended time.

    For every ``track_id`` the detector records centroid positions over
    time.  When a person's centroid stays within ``distance_threshold_px``
    of their earliest recorded position for longer than
    ``time_threshold_seconds``, a loitering event is emitted.

    CPU-only — uses numpy exclusively.
    """

    def __init__(self, config=None):
        """Initialise the loitering detector.

        Args:
            config: Dict with optional ``'loitering'`` section containing:
                - time_threshold_seconds (float): Minimum dwell duration
                  to flag.  Default 15.0.
                - distance_threshold_px (float): Max centroid displacement
                  that still counts as "staying".  Default 80.0.
        """
        config = config or {}
        loiter_cfg = config.get('loitering', {})
        self.time_threshold = float(
            loiter_cfg.get('time_threshold_seconds', 15.0))
        self.dist_threshold = float(
            loiter_cfg.get('distance_threshold_px', 80.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, person_results_per_frame):
        """Scan tracking data for loitering behaviour.

        Args:
            person_results_per_frame: List of per-frame dicts.  Each dict
                must contain ``'timestamp'`` (float) and ``'persons'``
                (list of dicts with ``'track_id'`` and ``'bbox'``).

        Returns:
            List[dict]: Loitering events, each with keys ``type``,
            ``track_id``, ``timestamp``, ``duration_seconds``,
            ``confidence``, ``severity``.
        """
        if not person_results_per_frame:
            return []

        # -- Build per-track timeline: [(timestamp, cx, cy), ...] ---------
        track_timeline = defaultdict(list)
        for frame_entry in person_results_per_frame:
            ts = frame_entry.get('timestamp', 0.0)
            for person in frame_entry.get('persons', []):
                tid = person.get('track_id')
                bbox = person.get('bbox')
                if tid is None or bbox is None or len(bbox) < 4:
                    continue
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                track_timeline[tid].append((ts, cx, cy))

        events = []

        for tid, timeline in track_timeline.items():
            if len(timeline) < 2:
                continue

            # Sort chronologically
            timeline.sort(key=lambda t: t[0])
            event = self._check_loitering(tid, timeline)
            if event is not None:
                events.append(event)

        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_loitering(self, track_id, timeline):
        """Sliding-window check for stationary behaviour.

        Uses a two-pointer approach: advance the right pointer and check
        whether all positions between left and right stay within the
        distance threshold from the anchor (left pointer) position.
        If the time span exceeds the threshold, emit an event.
        """
        n = len(timeline)
        best_duration = 0.0
        best_ts = timeline[0][0]

        left = 0
        for right in range(n):
            t_r, cx_r, cy_r = timeline[right]
            t_l, cx_l, cy_l = timeline[left]

            dist = np.sqrt((cx_r - cx_l) ** 2 + (cy_r - cy_l) ** 2)

            if dist > self.dist_threshold:
                # Move anchor forward
                left = right
                continue

            duration = t_r - t_l
            if duration > best_duration:
                best_duration = duration
                best_ts = t_l

        if best_duration >= self.time_threshold:
            return {
                'type': 'loitering',
                'track_id': track_id,
                'timestamp': float(best_ts),
                'duration_seconds': float(round(best_duration, 2)),
                'confidence': 0.90,
                'severity': 'MEDIUM',
            }

        return None
