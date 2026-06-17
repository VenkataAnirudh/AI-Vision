import numpy as np


class CrowdAnalyzer:
    """Estimates per-frame crowd density and emits overcrowding events.

    Density levels are driven by configurable person-count ranges read
    from ``config['crowd']``.  An overcrowding event is generated
    whenever the count meets or exceeds ``overcrowd_threshold``.

    CPU-only — uses numpy exclusively.
    """

    def __init__(self, config=None):
        """Initialise the crowd analyser.

        Args:
            config: Dict with an optional ``'crowd'`` section:
                - overcrowd_threshold (int): Person count that triggers
                  an overcrowding alert.  Default 10.
                - density_levels (dict): Mapping of level name →
                  ``[min_count, max_count]``.  Defaults:
                  sparse 0-2, moderate 3-5, dense 6-10,
                  overcrowded 11-9999.
        """
        config = config or {}
        crowd_cfg = config.get('crowd', {})
        self.overcrowd_threshold = int(
            crowd_cfg.get('overcrowd_threshold', 10))
        self.density_levels = crowd_cfg.get('density_levels', {
            'sparse': [0, 2],
            'moderate': [3, 5],
            'dense': [6, 10],
            'overcrowded': [11, 9999],
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, person_results_per_frame):
        """Analyse crowd density across all frames.

        Args:
            person_results_per_frame: List of per-frame dicts, each with
                ``'timestamp'`` (float) and ``'persons'`` (list).

        Returns:
            dict with keys:
                - max_count (int)
                - avg_count (float)
                - density_timeline (list of {timestamp, count, level})
                - overcrowding_events (list of event dicts)
        """
        if not person_results_per_frame:
            return {
                'max_count': 0,
                'avg_count': 0.0,
                'density_timeline': [],
                'overcrowding_events': [],
            }

        counts = []
        density_timeline = []
        overcrowding_events = []

        for frame_entry in person_results_per_frame:
            ts = frame_entry.get('timestamp', 0.0)
            persons = frame_entry.get('persons', [])
            count = len(persons)
            counts.append(count)

            level = self._classify(count)
            density_timeline.append({
                'timestamp': float(ts),
                'count': count,
                'level': level,
            })

            if count >= self.overcrowd_threshold:
                overcrowding_events.append({
                    'type': 'overcrowding',
                    'timestamp': float(ts),
                    'count': count,
                    'confidence': 0.95,
                    'severity': 'HIGH',
                })

        counts_arr = np.array(counts, dtype=np.float64)

        return {
            'max_count': int(counts_arr.max()),
            'avg_count': float(round(counts_arr.mean(), 2)),
            'density_timeline': density_timeline,
            'overcrowding_events': overcrowding_events,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self, count):
        """Return the density level string for a given person *count*."""
        for level_name, (lo, hi) in self.density_levels.items():
            if lo <= count <= hi:
                return level_name
        # Fallback — count exceeds every configured range
        return 'overcrowded'
